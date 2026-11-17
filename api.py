"""财报 RAG 查询接口。"""

from __future__ import annotations

import argparse
import copy
import json
import threading
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config, log, resolve_path
from lite_rag.search import retrieve
from lite_rag.store import get_model, keep_model_loaded, model_is_loaded, release_model

Mode = Literal["hybrid", "bm25", "vector"]
Scope = Literal["company", "knowledge"]
DocType = Literal["filing", "ir_qa", "survey", "research", "news", "theory", "beginner"]


class UTF8JSONResponse(JSONResponse):
    """中文按 UTF-8 原样返回，不转成 \\uXXXX。"""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


class RetrieveRequest(BaseModel):
    """检索请求。未传的可选项继续使用 config.yaml。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="要查询的问题")
    mode: Mode | None = Field(default=None, description="检索模式。不传则使用 retrieval.mode")
    top_k: StrictInt | None = Field(default=None, ge=1, description="返回的 chunk 数。不传则使用 retrieval.top_k")
    scope: Scope | None = Field(default=None, description="company 只查公司材料，knowledge 只查投资通识")
    doc_type: DocType | None = Field(
        default=None,
        description="filing 定期报告，ir_qa 互动问答，survey 调研，research 研报，news 新闻，theory 投资理论，beginner 入门答疑",
    )
    company: str | None = Field(default=None, description="公司简称，精确匹配")
    stock_code: str | None = Field(default=None, description="6 位股票代码")
    year: StrictInt | None = Field(default=None, ge=1900, le=2100, description="报告期或发布年份")

    @field_validator("question", "company")
    @classmethod
    def text_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("不能为空")
        return text

    @field_validator("stock_code")
    @classmethod
    def code_is_six_digits(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if len(text) != 6 or not text.isdigit():
            raise ValueError("需要是 6 位数字")
        return text


class QueryRequest(RetrieveRequest):
    """POST /query 的请求体。未传的可选项继续使用 config.yaml。"""

    answer: StrictBool | None = Field(default=None, description="是否调用大模型。不传则跟随 llm.enabled")


class Hit(BaseModel):
    rank: int
    id: int
    source_name: str
    source_path: str
    chunk_idx: int
    chunk_text: str
    score: float | None
    score_name: str
    scope: str
    doc_type: str
    title: str
    company: str
    stock_code: str
    year: int
    period: str
    published_on: str
    publisher: str


class RetrieveResponse(BaseModel):
    question: str
    mode: Mode
    hits: list[Hit]


class QueryResponse(RetrieveResponse):
    answer: str | None = None


def _exit_message(exc: SystemExit) -> str:
    if isinstance(exc.code, str) and exc.code.strip():
        return exc.code.strip()
    return "查询失败"


def create_app(config_path: str | None = None, keep_model: bool | None = None) -> FastAPI:
    configure_stdio()
    base_cfg = load_config(config_path)
    if keep_model is not None:
        base_cfg.setdefault("embedding", {})["keep_in_memory"] = keep_model

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if keep_model_loaded(base_cfg):
            get_model(base_cfg)
            log.info("embedding 模型已常驻内存")
        yield
        release_model()

    app = FastAPI(title="财报 RAG", default_response_class=UTF8JSONResponse, lifespan=lifespan)
    query_lock = threading.Lock()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return UTF8JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return UTF8JSONResponse(status_code=exc.status_code, content={"detail": jsonable_encoder(exc.detail)})

    def _prepare(body: RetrieveRequest) -> dict:
        cfg = copy.deepcopy(base_cfg)
        if body.mode is not None:
            cfg["retrieval"]["mode"] = body.mode
        if body.top_k is not None:
            cfg["retrieval"]["top_k"] = body.top_k
        return cfg

    def _retrieve_hits(cfg: dict, body: RetrieveRequest) -> list[dict]:
        try:
            return retrieve(
                cfg,
                body.question,
                scope=body.scope,
                doc_type=body.doc_type,
                company=body.company,
                stock_code=body.stock_code,
                year=body.year,
            )
        except SystemExit as exc:
            raise HTTPException(status_code=400, detail=_exit_message(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        db_path = resolve_path(base_cfg, base_cfg["paths"]["meta_db"])
        return {
            "status": "ok",
            "index_ready": db_path.is_file(),
            "keep_in_memory": keep_model_loaded(base_cfg),
            "model_loaded": model_is_loaded(),
        }

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve_only(body: RetrieveRequest) -> RetrieveResponse:
        cfg = _prepare(body)
        with query_lock:
            hits = _retrieve_hits(cfg, body)
        return RetrieveResponse(question=body.question, mode=cfg["retrieval"]["mode"], hits=hits)

    @app.post("/query", response_model=QueryResponse)
    def query(body: QueryRequest) -> QueryResponse:
        cfg = _prepare(body)
        want_answer = body.answer if body.answer is not None else bool(cfg["llm"].get("enabled"))
        with query_lock:
            hits = _retrieve_hits(cfg, body)

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


def serve(
    config_path: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8100,
    keep_model: bool | None = None,
) -> None:
    uvicorn.run(create_app(config_path, keep_model=keep_model), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="财报 RAG 查询接口")
    parser.add_argument("--config", default=None, help="配置文件路径，默认使用项目里的 config.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只接受本机访问")
    parser.add_argument("--port", type=int, default=8100, help="监听端口，默认 8100")
    parser.add_argument(
        "--keep-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="向量模型是否常驻内存。不传则使用 embedding.keep_in_memory",
    )
    args = parser.parse_args()
    serve(args.config, args.host, args.port, keep_model=args.keep_model)


if __name__ == "__main__":
    main()
