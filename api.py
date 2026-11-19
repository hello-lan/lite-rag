"""财报 RAG 查询接口。knowledge 与 financial_report 分路径，请求字段不混用。"""

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
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_serializer

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config, log, resolve_path
from lite_rag.search import retrieve
from lite_rag.store import get_model, keep_model_loaded, model_is_loaded, release_model

Mode = Literal["hybrid", "bm25", "vector"]
Output = Literal["hits", "answer", "both"]
Collection = Literal["knowledge", "financial_report"]
TheoryType = Literal["investment_theory"]
Figure = Literal["唐朝", "buffett", "munger"]
Occasion = Literal["letter", "meeting", "speech", "article"]
Section = Literal["profile", "segment", "mda", "risk", "material", "governance", "financials", "notes"]
Entity = Literal["consolidated", "parent"]


class UTF8JSONResponse(JSONResponse):
    """中文按 UTF-8 原样返回，不转成 \\uXXXX。"""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


class _RequestBase(BaseModel):
    """检索请求的公共字段。未传的可选项继续使用 config.yaml。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="要查询的问题")
    mode: Mode | None = Field(default=None, description="检索模式。不传则使用 retrieval.mode")
    top_k: StrictInt | None = Field(default=None, ge=1, description="返回的 chunk 数。不传则使用 retrieval.top_k")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("不能为空")
        return text


class KnowledgeRetrieveRequest(_RequestBase):
    """POST /knowledge/retrieve。只查投资通识，可按大类、人物、场合和年份筛选。"""

    doc_type: TheoryType | None = Field(default=None, description="大类。目前只有 investment_theory。不传则不限大类")
    figure: Figure | None = Field(
        default=None,
        description="主要人物：唐朝，buffett 巴菲特，munger 芒格。不传则不限人物",
    )
    occasion: Occasion | None = Field(
        default=None,
        description="场合：letter 致投资者的信，meeting 股东大会，speech 演讲，article 文章。不传则不限场合",
    )
    year: StrictInt | None = Field(default=None, ge=1900, le=2100, description="年份。不传则不限年份")


class FilingRetrieveRequest(_RequestBase):
    """POST /financial_report/retrieve。只查定期报告，可按公司、代码、报告期、章节和口径筛选。"""

    company: str | None = Field(default=None, description="公司简称，精确匹配")
    stock_code: str | None = Field(default=None, description="6 位股票代码")
    period: str | None = Field(default=None, description="报告期，如 2024、2024Q1、2024H1。累计口径")
    section: Section | None = Field(
        default=None,
        description="章节：profile、segment、mda、risk、material、governance、financials、notes",
    )
    entity: Entity | None = Field(
        default=None,
        description="报表口径：consolidated 合并，parent 母公司。只用于财务报表和附注",
    )

    @field_validator("company", "period")
    @classmethod
    def optional_text_not_blank(cls, value: str | None) -> str | None:
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


class KnowledgeQueryRequest(KnowledgeRetrieveRequest):
    """POST /knowledge/query。"""

    output: Output | None = Field(
        default=None,
        description="hits 只返回命中，answer 只返回回答，both 两者都返回。不传则看 llm.enabled",
    )


class FilingQueryRequest(FilingRetrieveRequest):
    """POST /financial_report/query。"""

    output: Output | None = Field(
        default=None,
        description="hits 只返回命中，answer 只返回回答，both 两者都返回。不传则看 llm.enabled",
    )


RetrieveBody = KnowledgeRetrieveRequest | FilingRetrieveRequest
QueryBody = KnowledgeQueryRequest | FilingQueryRequest


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
    author: str
    section: str = ""
    entity: str = ""
    figure: str = ""
    occasion: str = ""


class RetrieveResponse(BaseModel):
    question: str
    collection: Collection
    mode: Mode
    hits: list[Hit]


class QueryResponse(BaseModel):
    question: str
    collection: Collection
    mode: Mode
    output: Output
    hits: list[Hit] | None = None
    answer: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unused(self, handler):
        data = handler(self)
        if self.hits is None:
            data.pop("hits", None)
        if self.answer is None:
            data.pop("answer", None)
        return data


def _exit_message(exc: SystemExit) -> str:
    if isinstance(exc.code, str) and exc.code.strip():
        return exc.code.strip()
    return "查询失败"


def create_app(keep_model: bool | None = None) -> FastAPI:
    configure_stdio()
    base_cfg = load_config()
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

    def _prepare(body: RetrieveBody) -> dict:
        cfg = copy.deepcopy(base_cfg)
        if body.mode is not None:
            cfg["retrieval"]["mode"] = body.mode
        if body.top_k is not None:
            cfg["retrieval"]["top_k"] = body.top_k
        return cfg

    def _retrieve_hits(cfg: dict, body: RetrieveBody) -> list[dict]:
        if isinstance(body, KnowledgeRetrieveRequest):
            filters = {
                "scope": "knowledge",
                "doc_type": body.doc_type,
                "figure": body.figure,
                "occasion": body.occasion,
                "year": body.year,
            }
        else:
            filters = {
                "scope": "company",
                "doc_type": "filing",
                "company": body.company,
                "stock_code": body.stock_code,
                "period": body.period,
                "section": body.section,
                "entity": body.entity,
            }
        try:
            return retrieve(cfg, body.question, **filters)
        except SystemExit as exc:
            raise HTTPException(status_code=400, detail=_exit_message(exc)) from exc

    def _chosen_output(body: QueryBody, cfg: dict) -> Output:
        if body.output is not None:
            return body.output
        return "both" if cfg["llm"].get("enabled") else "hits"

    def _retrieve_response(body: RetrieveBody, collection: Collection) -> RetrieveResponse:
        cfg = _prepare(body)
        with query_lock:
            hits = _retrieve_hits(cfg, body)
        return RetrieveResponse(
            question=body.question,
            collection=collection,
            mode=cfg["retrieval"]["mode"],
            hits=hits,
        )

    def _query_response(body: QueryBody, collection: Collection) -> QueryResponse | JSONResponse:
        cfg = _prepare(body)
        chosen = _chosen_output(body, cfg)
        with query_lock:
            hits = _retrieve_hits(cfg, body)
            answer = None
            if chosen in ("answer", "both"):
                try:
                    answer = generate_answer(cfg, body.question, hits)
                except SystemExit as exc:
                    return UTF8JSONResponse(
                        status_code=502,
                        content={
                            "detail": _exit_message(exc),
                            "collection": collection,
                            "mode": cfg["retrieval"]["mode"],
                            "output": chosen,
                            "hits": hits,
                        },
                    )

        return QueryResponse(
            question=body.question,
            collection=collection,
            mode=cfg["retrieval"]["mode"],
            output=chosen,
            hits=None if chosen == "answer" else hits,
            answer=answer if chosen in ("answer", "both") else None,
        )

    @app.get("/health")
    def health() -> dict:
        db_path = resolve_path(base_cfg, base_cfg["paths"]["meta_db"])
        return {
            "status": "ok",
            "index_ready": db_path.is_file(),
            "keep_in_memory": keep_model_loaded(base_cfg),
            "model_loaded": model_is_loaded(),
        }

    @app.post("/knowledge/retrieve", response_model=RetrieveResponse)
    def retrieve_knowledge(body: KnowledgeRetrieveRequest) -> RetrieveResponse:
        return _retrieve_response(body, "knowledge")

    @app.post("/knowledge/query", response_model=QueryResponse)
    def query_knowledge(body: KnowledgeQueryRequest) -> QueryResponse | JSONResponse:
        return _query_response(body, "knowledge")

    @app.post("/financial_report/retrieve", response_model=RetrieveResponse)
    def retrieve_financial_report(body: FilingRetrieveRequest) -> RetrieveResponse:
        return _retrieve_response(body, "financial_report")

    @app.post("/financial_report/query", response_model=QueryResponse)
    def query_financial_report(body: FilingQueryRequest) -> QueryResponse | JSONResponse:
        return _query_response(body, "financial_report")

    return app


def serve(
    host: str = "127.0.0.1",
    port: int = 8100,
    keep_model: bool | None = None,
) -> None:
    uvicorn.run(create_app(keep_model=keep_model), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="财报 RAG 查询接口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只接受本机访问")
    parser.add_argument("--port", type=int, default=8100, help="监听端口，默认 8100")
    parser.add_argument(
        "--keep-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="向量模型是否常驻内存。不传则使用 embedding.keep_in_memory",
    )
    args = parser.parse_args()
    serve(args.host, args.port, keep_model=args.keep_model)


if __name__ == "__main__":
    main()
