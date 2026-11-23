"""财报 RAG 命令行：预切分 JSON 入库、查询、按当前模型重算向量、启动查询接口。"""

from __future__ import annotations

from pathlib import Path

import click

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config
from lite_rag.ingest import ingest_split_documents, reembed_vectors
from lite_rag.meta import (
    ENTITIES,
    FIGURES,
    OCCASIONS,
    SECTIONS,
    TYPES_BY_SCOPE,
    SearchFilter,
    source_label,
)
from lite_rag.search import retrieve

_KNOWLEDGE_DOC_TYPES = sorted(TYPES_BY_SCOPE["knowledge"])
_FINANCIAL_DOC_TYPES = sorted(TYPES_BY_SCOPE["financial_report"])


@click.group(help="本地财报 RAG：预切分 JSON 入库、查询、按新模型重算向量、启动查询接口。")
def cli() -> None:
    configure_stdio()


def print_hits(mode: str, hits: list[dict], filt: SearchFilter | None = None) -> None:
    line = f"模式: {mode}"
    if filt is not None and filt.active():
        line += f"  筛选: {filt.describe()}"
    print(line)
    if not hits:
        if filt is not None and filt.active():
            print("当前筛选条件下没有命中。")
        else:
            print("没有命中。可以换一种说法，或改用 --mode bm25 / --mode vector 对比。")
        return
    print(f"命中 {len(hits)} 条\n")
    for hit in hits:
        score = hit["score"]
        score_text = f"{score:.4f}" if isinstance(score, float) else "-"
        print(
            f"[{hit['rank']}] id={hit['id']}  {hit['score_name']}={score_text}  "
            f"{source_label(hit)}  chunk #{hit['chunk_idx']}"
        )
        print(hit["chunk_text"])
        print()


def _run_query(cfg: dict, text: str, filt: SearchFilter, output: str | None) -> None:
    hits = retrieve(
        cfg,
        text,
        scope=filt.scope or None,
        doc_type=filt.doc_type or None,
        stock_name=filt.stock_name or None,
        stock_code=filt.stock_code or None,
        year=filt.year or None,
        period=filt.period or None,
        section=filt.section or None,
        entity=filt.entity or None,
        figure=filt.figure or None,
        occasion=filt.occasion or None,
    )
    if output is None:
        output = "both" if cfg["llm"].get("enabled") else "hits"
    if output in ("hits", "both"):
        print_hits(cfg["retrieval"]["mode"], hits, filt)
    if output in ("answer", "both"):
        print("回答:")
        print(generate_answer(cfg, text, hits))


def _prepare_cfg(mode: str | None, top_k: int | None) -> dict:
    cfg = load_config()
    if mode:
        cfg["retrieval"]["mode"] = mode
    if top_k is not None:
        cfg["retrieval"]["top_k"] = top_k
    return cfg


def _question_text(question: tuple[str, ...]) -> str:
    text = " ".join(question).strip()
    if not text:
        raise click.UsageError("问题不能为空")
    return text


_MODE_OPTION = click.option(
    "--mode", type=click.Choice(["hybrid", "bm25", "vector"]), help="覆盖 config.yaml 里的检索模式"
)
_TOP_K_OPTION = click.option("--top-k", type=click.IntRange(min=1), help="最终返回的 chunk 数")
_OUTPUT_OPTION = click.option(
    "--output",
    type=click.Choice(["hits", "answer", "both"]),
    help="返回内容：hits 只打印检索结果，answer 只打印生成回复，both 两者都打印。不传则按 llm.enabled 选择 hits 或 both",
)


@cli.command("ingest-split", help="把已切分的 JSON 嵌入并写入索引。默认读取配置里的 paths.split_dir。")
@click.option("--force", is_flag=True, help="即使文本和元信息没变，也重新嵌入并写入")
@click.option("--check", is_flag=True, help="只校验 JSON，不下载模型，也不写数据库")
@click.option("--bm25-only", is_flag=True, help="不读 JSON，只按 sqlite 里的原文重建 BM25")
@click.option(
    "--dir",
    "split_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="本次读取的 JSON 目录。相对路径相对项目根目录。不传则使用 paths.split_dir",
)
def ingest_split(force: bool, check: bool, bm25_only: bool, split_dir: Path | None) -> None:
    cfg = load_config()
    ingest_split_documents(
        cfg, force=force, check=check, bm25_only=bm25_only, split_dir=split_dir
    )


@cli.command("query-knowledge", help="检索投资通识 chunk，可选调用大模型生成回答。")
@click.argument("question", nargs=-1, required=True)
@_MODE_OPTION
@_TOP_K_OPTION
@_OUTPUT_OPTION
@click.option(
    "--doc-type",
    type=click.Choice(_KNOWLEDGE_DOC_TYPES),
    help="材料类型：investment_theory、beginner",
)
@click.option(
    "--figure",
    type=click.Choice(list(FIGURES)),
    default=None,
    help="投资理论的主要人物：唐朝，buffett 巴菲特，munger 芒格",
)
@click.option(
    "--occasion",
    type=click.Choice(list(OCCASIONS)),
    default=None,
    help="投资理论的场合：letter 致投资者的信，meeting 股东大会，speech 演讲，article 文章",
)
@click.option("--year", type=click.IntRange(1900, 2100), help="年份")
def query_knowledge(
    question: tuple[str, ...],
    mode: str | None,
    top_k: int | None,
    output: str | None,
    doc_type: str | None,
    figure: str | None,
    occasion: str | None,
    year: int | None,
) -> None:
    cfg = _prepare_cfg(mode, top_k)
    text = _question_text(question)
    filt = SearchFilter.parse(
        scope="knowledge",
        doc_type=doc_type,
        figure=figure,
        occasion=occasion,
        year=year,
    )
    _run_query(cfg, text, filt, output)


@cli.command("query-financial-report", help="检索公司材料 chunk，可选调用大模型生成回答。")
@click.argument("question", nargs=-1, required=True)
@_MODE_OPTION
@_TOP_K_OPTION
@_OUTPUT_OPTION
@click.option(
    "--doc-type",
    type=click.Choice(_FINANCIAL_DOC_TYPES),
    help="材料类型：filing、ir_qa、survey、research、news",
)
@click.option("--stock-name", "stock_name", default=None, help="股票简称，精确匹配")
@click.option("--code", "stock_code", default=None, help="6 位股票代码")
@click.option("--period", default=None, help="报告期，如 2024、2024Q1、2024H1。累计口径")
@click.option(
    "--section",
    type=click.Choice(list(SECTIONS)),
    default=None,
    help="章节：profile、mda、material、governance、financials、notes、other",
)
@click.option(
    "--entity",
    type=click.Choice(list(ENTITIES)),
    default=None,
    help="报表口径：consolidated 合并，parent 母公司。只用于财务报表和附注",
)
def query_financial_report(
    question: tuple[str, ...],
    mode: str | None,
    top_k: int | None,
    output: str | None,
    doc_type: str | None,
    stock_name: str | None,
    stock_code: str | None,
    period: str | None,
    section: str | None,
    entity: str | None,
) -> None:
    cfg = _prepare_cfg(mode, top_k)
    text = _question_text(question)
    filt = SearchFilter.parse(
        scope="financial_report",
        doc_type=doc_type,
        stock_name=stock_name,
        stock_code=stock_code,
        period=period,
        section=section,
        entity=entity,
    )
    _run_query(cfg, text, filt, output)


@cli.command(help="启动查询接口，和 python api.py 是同一个服务。")
@click.option("--host", default="127.0.0.1", show_default=True, help="监听地址，默认只接受本机访问")
@click.option("--port", type=click.IntRange(min=1, max=65535), default=8100, show_default=True, help="监听端口")
@click.option(
    "--keep-model/--no-keep-model",
    "keep_model",
    default=None,
    help="向量模型是否常驻内存。不传则使用 embedding.keep_in_memory",
)
def serve(host: str, port: int, keep_model: bool | None) -> None:
    from api import serve as run_api

    run_api(host, port, keep_model=keep_model)


@cli.command(help="按 sqlite 中的 chunk 原文重算向量并重建 Chroma。BM25 和原文都不动。")
@click.option("--force", is_flag=True, help="即使模型名和向量条数都没变，也重新写入")
def reembed(force: bool) -> None:
    cfg = load_config()
    reembed_vectors(cfg, force=force)


if __name__ == "__main__":
    cli()
