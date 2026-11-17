"""财报 RAG 查询接口。"""

from __future__ import annotations

import argparse
import copy
import json
import threading
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config
from lite_rag.search import retrieve

Mode = Literal["hybrid", "bm25", "vector"]


class UTF8JSONResponse(JSONResponse):
    """中文按 UTF-8 原样返回，不转成 \\uXXXX。"""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


class QueryRequest(BaseModel):
    """POST /query 的请求体。未传的可选项继续使用 config.yaml。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="要查询的问题")
    mode: Mode | None = Field(default=None, description="检索模式。不传则使用 retrieval.mode")
    top_k: StrictInt | None = Field(default=None, ge=1, description="返回的 chunk 数。不传则使用 retrieval.top_k")
    answer: StrictBool | None = Field(default=None, description="是否调用大模型。不传则跟随 llm.enabled")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("不能为空")
        return text


class Hit(BaseModel):
    rank: int
    id: int
    source_name: str
    source_path: str
    chunk_idx: int
    chunk_text: str
    score: float | None
    score_name: str


class QueryResponse(BaseModel):
    question: str
    mode: Mode
    hits: list[Hit]
    answer: str | None = None


def _exit_message(exc: SystemExit) -> str:
    if isinstance(exc.code, str) and exc.code.strip():
        return exc.code.strip()
    return "查询失败"


def create_app(config_path: str | None = None) -> FastAPI:
    configure_stdio()
    base_cfg = load_config(config_path)
    app = FastAPI(title="财报 RAG", default_response_class=UTF8JSONResponse)
    query_lock = threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return UTF8JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return UTF8JSONResponse(status_code=exc.status_code, content={"detail": jsonable_encoder(exc.detail)})

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> QueryResponse:
        cfg = copy.deepcopy(base_cfg)
        if body.mode is not None:
            cfg["retrieval"]["mode"] = body.mode
        if body.top_k is not None:
            cfg["retrieval"]["top_k"] = body.top_k

        want_answer = body.answer if body.answer is not None else bool(cfg["llm"].get("enabled"))
        with query_lock:
            try:
                hits = retrieve(cfg, body.question)
            except SystemExit as exc:
                raise HTTPException(status_code=400, detail=_exit_message(exc)) from exc

            answer = None
            if want_answer:
                try:
                    answer = generate_answer(cfg, body.question, hits)
                except SystemExit as exc:
                    return UTF8JSONResponse(
                        status_code=502,
                        content={
                            "detail": _exit_message(exc),
                            "mode": cfg["retrieval"]["mode"],
                            "hits": hits,
                        },
                    )

        return QueryResponse(question=body.question, mode=cfg["retrieval"]["mode"], hits=hits, answer=answer)

    return app


def serve(config_path: str | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(create_app(config_path), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="财报 RAG 查询接口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认使用项目里的 config.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只接受本机访问")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    args = parser.parse_args()
    serve(args.config, args.host, args.port)


if __name__ == "__main__":
    main()
