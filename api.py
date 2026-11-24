"""财报 RAG 查询接口。knowledge、financial_report、investor_materials 分路径，请求字段不混用。"""

from __future__ import annotations

import argparse
import copy
import json
import threading
from contextlib import asynccontextmanager
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_serializer

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config, log, resolve_path
from lite_rag.meta import (
    DOC_TYPE_LABELS,
    ENTITY_LABELS,
    FIGURE_LABELS,
    OCCASION_LABELS,
    SECTION_LABELS,
)
from lite_rag.search import retrieve
from lite_rag.store import get_model, keep_model_loaded, model_is_loaded, release_model

Mode = Literal["hybrid", "bm25", "vector"]
Output = Literal["hits", "answer", "both"]
TheoryType = Literal["investment_theory"]
InvestorDocType = Literal["ir_qa", "ir_activities", "research", "news"]
Figure = Literal["唐朝", "buffett", "munger"]
Occasion = Literal["letter", "meeting", "speech", "article"]
Section = Literal["profile", "mda", "material", "governance", "financials", "notes", "other"]
Entity = Literal["consolidated", "parent"]

MODE_LABELS = {
    "hybrid": "混合检索（BM25 + 向量）",
    "bm25": "BM25 关键词检索",
    "vector": "向量语义检索",
}
OUTPUT_LABELS = {
    "hits": "只返回命中片段",
    "answer": "只返回大模型回答",
    "both": "同时返回命中和大模型回答",
}
COLLECTION_LABELS = {
    "knowledge": "投资通识",
    "financial_report": "定期报告",
    "investor_materials": "投资者材料",
}
THEORY_TYPE_LABELS = {
    key: DOC_TYPE_LABELS[key] for key in ("investment_theory",) if key in DOC_TYPE_LABELS
}
INVESTOR_TYPE_LABELS = {
    key: DOC_TYPE_LABELS[key]
    for key in ("ir_qa", "ir_activities", "research", "news")
    if key in DOC_TYPE_LABELS
}


def _enum_choices(labels: Mapping[str, str]) -> str:
    return "；".join(f"{value}={meaning}" for value, meaning in labels.items())


def _enum_field(description: str, labels: Mapping[str, str], **kwargs: Any):
    """枚举字段：description 列出取值与含义，OpenAPI 额外带 x-enum-meanings 供 RAG 配置读取。"""
    choices = _enum_choices(labels)
    text = f"{description}。取值：{choices}" if description else f"取值：{choices}"
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra["x-enum-meanings"] = dict(labels)
    return Field(description=text, json_schema_extra=extra, **kwargs)


class UTF8JSONResponse(JSONResponse):
    """中文按 UTF-8 原样返回，不转成 \\uXXXX。"""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")


class _HitBase(BaseModel):
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


class KnowledgeHit(_HitBase):
    year: int
    figure: str = ""
    occasion: str = ""


class FilingHit(_HitBase):
    stock_name: str
    stock_code: str
    period: str
    published_on: str
    section: str = ""
    entity: str = ""


class InvestorMaterialsHit(_HitBase):
    stock_name: str
    stock_code: str
    year: int


_KNOWLEDGE_HIT_KEYS = tuple(KnowledgeHit.model_fields)
_FILING_HIT_KEYS = tuple(FilingHit.model_fields)
_INVESTOR_HIT_KEYS = tuple(InvestorMaterialsHit.model_fields)
_KNOWLEDGE_HIT_KEY_SET = frozenset(_KNOWLEDGE_HIT_KEYS)
_FILING_HIT_KEY_SET = frozenset(_FILING_HIT_KEYS)
_INVESTOR_HIT_KEY_SET = frozenset(_INVESTOR_HIT_KEYS)
KnowledgeHitField = Literal[*_KNOWLEDGE_HIT_KEYS]
FilingHitField = Literal[*_FILING_HIT_KEYS]
InvestorHitField = Literal[*_INVESTOR_HIT_KEYS]


def _normalize_hit_fields(value: list[str] | None, allowed: frozenset[str]) -> list[str] | None:
    if value is None:
        return None
    if not value:
        raise ValueError("至少指定一个字段")
    unknown = sorted({name for name in value if name not in allowed})
    if unknown:
        raise ValueError("不支持的字段: " + "、".join(unknown))
    ordered: list[str] = []
    seen: set[str] = set()
    for name in value:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _project_hits(hits: list[dict], keys: Sequence[str]) -> list[dict]:
    return [{key: hit[key] for key in keys if key in hit} for hit in hits]


def _hit_keys(requested: list[str] | None, all_keys: Sequence[str]) -> Sequence[str]:
    return requested if requested is not None else all_keys


class _RequestBase(BaseModel):
    """检索请求的公共字段。未传的可选项继续使用 config.yaml。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="要查询的问题")
    mode: Mode | None = _enum_field(
        "检索模式。不传则使用 retrieval.mode",
        MODE_LABELS,
        default=None,
    )
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

    doc_type: TheoryType | None = _enum_field(
        "大类。不传则不限大类",
        THEORY_TYPE_LABELS,
        default=None,
    )
    figure: Figure | None = _enum_field(
        "主要人物。不传则不限人物",
        FIGURE_LABELS,
        default=None,
    )
    occasion: Occasion | None = _enum_field(
        "场合。不传则不限场合",
        OCCASION_LABELS,
        default=None,
    )
    year: StrictInt | None = Field(default=None, ge=1900, le=2100, description="年份。不传则不限年份")
    hit_fields: list[KnowledgeHitField] | None = Field(
        default=None,
        description="要返回的 hit 字段。不传则返回全部；传入则每条 hit 只保留名单中的键",
        min_length=1,
    )

    @field_validator("hit_fields")
    @classmethod
    def knowledge_hit_fields(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_hit_fields(value, _KNOWLEDGE_HIT_KEY_SET)


class FilingRetrieveRequest(_RequestBase):
    """POST /financial_report/retrieve。只查定期报告，可按公司、代码、报告期、章节和口径筛选。"""

    stock_name: str | None = Field(default=None, description="股票简称，精确匹配")
    stock_code: str | None = Field(default=None, description="6 位股票代码")
    period: str | None = Field(default=None, description="报告期，如 2024、2024Q1、2024H1。累计口径")
    section: Section | None = _enum_field(
        "章节。不传则不限章节",
        SECTION_LABELS,
        default=None,
    )
    entity: Entity | None = _enum_field(
        "报表口径。只用于财务报表和附注。不传则不限口径",
        ENTITY_LABELS,
        default=None,
    )
    hit_fields: list[FilingHitField] | None = Field(
        default=None,
        description="要返回的 hit 字段。不传则返回全部；传入则每条 hit 只保留名单中的键",
        min_length=1,
    )

    @field_validator("stock_name", "period")
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

    @field_validator("hit_fields")
    @classmethod
    def filing_hit_fields(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_hit_fields(value, _FILING_HIT_KEY_SET)


class KnowledgeQueryRequest(KnowledgeRetrieveRequest):
    """POST /knowledge/query。"""

    output: Output | None = _enum_field(
        "返回内容。不传则看 llm.enabled：true 相当于 both，false 相当于 hits",
        OUTPUT_LABELS,
        default=None,
    )


class FilingQueryRequest(FilingRetrieveRequest):
    """POST /financial_report/query。"""

    output: Output | None = _enum_field(
        "返回内容。不传则看 llm.enabled：true 相当于 both，false 相当于 hits",
        OUTPUT_LABELS,
        default=None,
    )


class InvestorMaterialsRetrieveRequest(_RequestBase):
    """POST /investor_materials/retrieve。只查投资者材料，可按大类、公司、代码和年份筛选。"""

    doc_type: InvestorDocType | None = _enum_field(
        "大类。不传则不限大类",
        INVESTOR_TYPE_LABELS,
        default=None,
    )
    stock_name: str | None = Field(default=None, description="股票简称，精确匹配")
    stock_code: str | None = Field(default=None, description="6 位股票代码")
    year: StrictInt | None = Field(default=None, ge=1900, le=2100, description="年份。不传则不限年份")
    hit_fields: list[InvestorHitField] | None = Field(
        default=None,
        description="要返回的 hit 字段。不传则返回全部；传入则每条 hit 只保留名单中的键",
        min_length=1,
    )

    @field_validator("stock_name")
    @classmethod
    def investor_stock_name_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("不能为空")
        return text

    @field_validator("stock_code")
    @classmethod
    def investor_code_is_six_digits(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if len(text) != 6 or not text.isdigit():
            raise ValueError("需要是 6 位数字")
        return text

    @field_validator("hit_fields")
    @classmethod
    def investor_hit_fields(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_hit_fields(value, _INVESTOR_HIT_KEY_SET)


class InvestorMaterialsQueryRequest(InvestorMaterialsRetrieveRequest):
    """POST /investor_materials/query。"""

    output: Output | None = _enum_field(
        "返回内容。不传则看 llm.enabled：true 相当于 both，false 相当于 hits",
        OUTPUT_LABELS,
        default=None,
    )


RetrieveBody = KnowledgeRetrieveRequest | FilingRetrieveRequest | InvestorMaterialsRetrieveRequest
QueryBody = KnowledgeQueryRequest | FilingQueryRequest | InvestorMaterialsQueryRequest


_KNOWLEDGE_COLLECTION = {"knowledge": COLLECTION_LABELS["knowledge"]}
_FILING_COLLECTION = {"financial_report": COLLECTION_LABELS["financial_report"]}
_INVESTOR_COLLECTION = {"investor_materials": COLLECTION_LABELS["investor_materials"]}


class KnowledgeRetrieveResponse(BaseModel):
    question: str = Field(description="查询问题")
    collection: Literal["knowledge"] = _enum_field("由请求路径决定的语料范围", _KNOWLEDGE_COLLECTION)
    mode: Mode = _enum_field("实际使用的检索模式", MODE_LABELS)
    hits: list[KnowledgeHit] = Field(description="命中列表。请求传了 hit_fields 时每条只含名单中的键")


class FilingRetrieveResponse(BaseModel):
    question: str = Field(description="查询问题")
    collection: Literal["financial_report"] = _enum_field("由请求路径决定的语料范围", _FILING_COLLECTION)
    mode: Mode = _enum_field("实际使用的检索模式", MODE_LABELS)
    hits: list[FilingHit] = Field(description="命中列表。请求传了 hit_fields 时每条只含名单中的键")


class InvestorMaterialsRetrieveResponse(BaseModel):
    question: str = Field(description="查询问题")
    collection: Literal["investor_materials"] = _enum_field(
        "由请求路径决定的语料范围", _INVESTOR_COLLECTION
    )
    mode: Mode = _enum_field("实际使用的检索模式", MODE_LABELS)
    hits: list[InvestorMaterialsHit] = Field(
        description="命中列表。请求传了 hit_fields 时每条只含名单中的键"
    )


class KnowledgeQueryResponse(BaseModel):
    question: str = Field(description="查询问题")
    collection: Literal["knowledge"] = _enum_field("由请求路径决定的语料范围", _KNOWLEDGE_COLLECTION)
    mode: Mode = _enum_field("实际使用的检索模式", MODE_LABELS)
    output: Output = _enum_field("实际采用的返回内容", OUTPUT_LABELS)
    hits: list[KnowledgeHit] | None = Field(
        default=None,
        description="命中列表。请求传了 hit_fields 时每条只含名单中的键",
    )
    answer: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unused(self, handler):
        data = handler(self)
        if self.hits is None:
            data.pop("hits", None)
        if self.answer is None:
            data.pop("answer", None)
        return data


class FilingQueryResponse(BaseModel):
    question: str = Field(description="查询问题")
    collection: Literal["financial_report"] = _enum_field("由请求路径决定的语料范围", _FILING_COLLECTION)
    mode: Mode = _enum_field("实际使用的检索模式", MODE_LABELS)
    output: Output = _enum_field("实际采用的返回内容", OUTPUT_LABELS)
    hits: list[FilingHit] | None = Field(
        default=None,
        description="命中列表。请求传了 hit_fields 时每条只含名单中的键",
    )
    answer: str | None = None

    @model_serializer(mode="wrap")
    def _omit_unused(self, handler):
        data = handler(self)
        if self.hits is None:
            data.pop("hits", None)
        if self.answer is None:
            data.pop("answer", None)
        return data


class InvestorMaterialsQueryResponse(BaseModel):
    question: str = Field(description="查询问题")
    collection: Literal["investor_materials"] = _enum_field(
        "由请求路径决定的语料范围", _INVESTOR_COLLECTION
    )
    mode: Mode = _enum_field("实际使用的检索模式", MODE_LABELS)
    output: Output = _enum_field("实际采用的返回内容", OUTPUT_LABELS)
    hits: list[InvestorMaterialsHit] | None = Field(
        default=None,
        description="命中列表。请求传了 hit_fields 时每条只含名单中的键",
    )
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
        elif isinstance(body, InvestorMaterialsRetrieveRequest):
            filters = {
                "scope": "investor_materials",
                "doc_type": body.doc_type,
                "stock_name": body.stock_name,
                "stock_code": body.stock_code,
                "year": body.year,
            }
        else:
            filters = {
                "scope": "financial_report",
                "doc_type": "filing",
                "stock_name": body.stock_name,
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

    def _knowledge_retrieve_response(body: KnowledgeRetrieveRequest) -> UTF8JSONResponse:
        cfg = _prepare(body)
        keys = _hit_keys(body.hit_fields, _KNOWLEDGE_HIT_KEYS)
        with query_lock:
            hits = _project_hits(_retrieve_hits(cfg, body), keys)
        return UTF8JSONResponse(
            content={
                "question": body.question,
                "collection": "knowledge",
                "mode": cfg["retrieval"]["mode"],
                "hits": hits,
            }
        )

    def _filing_retrieve_response(body: FilingRetrieveRequest) -> UTF8JSONResponse:
        cfg = _prepare(body)
        keys = _hit_keys(body.hit_fields, _FILING_HIT_KEYS)
        with query_lock:
            hits = _project_hits(_retrieve_hits(cfg, body), keys)
        return UTF8JSONResponse(
            content={
                "question": body.question,
                "collection": "financial_report",
                "mode": cfg["retrieval"]["mode"],
                "hits": hits,
            }
        )

    def _knowledge_query_response(body: KnowledgeQueryRequest) -> UTF8JSONResponse:
        cfg = _prepare(body)
        chosen = _chosen_output(body, cfg)
        keys = _hit_keys(body.hit_fields, _KNOWLEDGE_HIT_KEYS)
        with query_lock:
            raw_hits = _retrieve_hits(cfg, body)
            hits = _project_hits(raw_hits, keys)
            answer = None
            if chosen in ("answer", "both"):
                try:
                    answer = generate_answer(cfg, body.question, raw_hits)
                except SystemExit as exc:
                    return UTF8JSONResponse(
                        status_code=502,
                        content={
                            "detail": _exit_message(exc),
                            "collection": "knowledge",
                            "mode": cfg["retrieval"]["mode"],
                            "output": chosen,
                            "hits": hits,
                        },
                    )

        content: dict[str, Any] = {
            "question": body.question,
            "collection": "knowledge",
            "mode": cfg["retrieval"]["mode"],
            "output": chosen,
        }
        if chosen != "answer":
            content["hits"] = hits
        if chosen in ("answer", "both"):
            content["answer"] = answer
        return UTF8JSONResponse(content=content)

    def _filing_query_response(body: FilingQueryRequest) -> UTF8JSONResponse:
        cfg = _prepare(body)
        chosen = _chosen_output(body, cfg)
        keys = _hit_keys(body.hit_fields, _FILING_HIT_KEYS)
        with query_lock:
            raw_hits = _retrieve_hits(cfg, body)
            hits = _project_hits(raw_hits, keys)
            answer = None
            if chosen in ("answer", "both"):
                try:
                    answer = generate_answer(cfg, body.question, raw_hits)
                except SystemExit as exc:
                    return UTF8JSONResponse(
                        status_code=502,
                        content={
                            "detail": _exit_message(exc),
                            "collection": "financial_report",
                            "mode": cfg["retrieval"]["mode"],
                            "output": chosen,
                            "hits": hits,
                        },
                    )

        content: dict[str, Any] = {
            "question": body.question,
            "collection": "financial_report",
            "mode": cfg["retrieval"]["mode"],
            "output": chosen,
        }
        if chosen != "answer":
            content["hits"] = hits
        if chosen in ("answer", "both"):
            content["answer"] = answer
        return UTF8JSONResponse(content=content)

    def _investor_retrieve_response(body: InvestorMaterialsRetrieveRequest) -> UTF8JSONResponse:
        cfg = _prepare(body)
        keys = _hit_keys(body.hit_fields, _INVESTOR_HIT_KEYS)
        with query_lock:
            hits = _project_hits(_retrieve_hits(cfg, body), keys)
        return UTF8JSONResponse(
            content={
                "question": body.question,
                "collection": "investor_materials",
                "mode": cfg["retrieval"]["mode"],
                "hits": hits,
            }
        )

    def _investor_query_response(body: InvestorMaterialsQueryRequest) -> UTF8JSONResponse:
        cfg = _prepare(body)
        chosen = _chosen_output(body, cfg)
        keys = _hit_keys(body.hit_fields, _INVESTOR_HIT_KEYS)
        with query_lock:
            raw_hits = _retrieve_hits(cfg, body)
            hits = _project_hits(raw_hits, keys)
            answer = None
            if chosen in ("answer", "both"):
                try:
                    answer = generate_answer(cfg, body.question, raw_hits)
                except SystemExit as exc:
                    return UTF8JSONResponse(
                        status_code=502,
                        content={
                            "detail": _exit_message(exc),
                            "collection": "investor_materials",
                            "mode": cfg["retrieval"]["mode"],
                            "output": chosen,
                            "hits": hits,
                        },
                    )

        content: dict[str, Any] = {
            "question": body.question,
            "collection": "investor_materials",
            "mode": cfg["retrieval"]["mode"],
            "output": chosen,
        }
        if chosen != "answer":
            content["hits"] = hits
        if chosen in ("answer", "both"):
            content["answer"] = answer
        return UTF8JSONResponse(content=content)

    @app.get("/health")
    def health() -> dict:
        db_path = resolve_path(base_cfg, base_cfg["paths"]["meta_db"])
        return {
            "status": "ok",
            "index_ready": db_path.is_file(),
            "keep_in_memory": keep_model_loaded(base_cfg),
            "model_loaded": model_is_loaded(),
        }

    @app.post("/knowledge/retrieve", response_model=KnowledgeRetrieveResponse)
    def retrieve_knowledge(body: KnowledgeRetrieveRequest) -> UTF8JSONResponse:
        return _knowledge_retrieve_response(body)

    @app.post("/knowledge/query", response_model=KnowledgeQueryResponse)
    def query_knowledge(body: KnowledgeQueryRequest) -> UTF8JSONResponse:
        return _knowledge_query_response(body)

    @app.post("/financial_report/retrieve", response_model=FilingRetrieveResponse)
    def retrieve_financial_report(body: FilingRetrieveRequest) -> UTF8JSONResponse:
        return _filing_retrieve_response(body)

    @app.post("/financial_report/query", response_model=FilingQueryResponse)
    def query_financial_report(body: FilingQueryRequest) -> UTF8JSONResponse:
        return _filing_query_response(body)

    @app.post("/investor_materials/retrieve", response_model=InvestorMaterialsRetrieveResponse)
    def retrieve_investor_materials(body: InvestorMaterialsRetrieveRequest) -> UTF8JSONResponse:
        return _investor_retrieve_response(body)

    @app.post("/investor_materials/query", response_model=InvestorMaterialsQueryResponse)
    def query_investor_materials(body: InvestorMaterialsQueryRequest) -> UTF8JSONResponse:
        return _investor_query_response(body)

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
