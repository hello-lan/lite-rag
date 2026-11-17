"""财报 RAG 命令行：入库、预切分入库、查询、按当前模型重算向量、启动查询接口。"""

from __future__ import annotations

import click

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config
from lite_rag.ingest import ingest_documents, ingest_split_documents, reembed_vectors
from lite_rag.meta import DOC_TYPES, SCOPES, SearchFilter, source_label
from lite_rag.search import retrieve


@click.group(help="本地财报 RAG：入库、预切分入库、查询、按新模型重算向量、启动查询接口。")
def cli() -> None:
    configure_stdio()


def _config_option(func):
    return click.option("--config", "config_path", default=None, help="配置文件路径，默认使用项目里的 config.yaml")(func)


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


@cli.command(help="把 docs/raw 中的 PDF 和 Markdown 切分后写入财报 RAG 索引。")
@_config_option
@click.option("--force", is_flag=True, help="忽略未改动的文档，全部重新切分并写入向量")
@click.option("--check", is_flag=True, help="只解析和切分，不下载模型，也不写数据库")
@click.option("--bm25-only", is_flag=True, help="不重新解析文档，只按 sqlite 里的原文重建 BM25")
def ingest(config_path: str | None, force: bool, check: bool, bm25_only: bool) -> None:
    cfg = load_config(config_path)
    ingest_documents(cfg, force=force, check=check, bm25_only=bm25_only)


@cli.command("ingest-split", help="把 docs/split 中已切分的 JSON 嵌入并写入索引，不再切分。")
@_config_option
@click.option("--force", is_flag=True, help="即使文本和元信息没变，也重新嵌入并写入")
@click.option("--check", is_flag=True, help="只校验 JSON，不下载模型，也不写数据库")
def ingest_split(config_path: str | None, force: bool, check: bool) -> None:
    cfg = load_config(config_path)
    ingest_split_documents(cfg, force=force, check=check)


@cli.command(help="按 BM25、向量或 RRF 混合检索财报 chunk，可选调用大模型生成回答。")
@click.argument("question", nargs=-1, required=True)
@_config_option
@click.option("--mode", type=click.Choice(["hybrid", "bm25", "vector"]), help="覆盖 config.yaml 里的检索模式")
@click.option("--top-k", type=click.IntRange(min=1), help="最终返回的 chunk 数")
@click.option("--answer", is_flag=True, help="用配置里的大模型根据命中材料生成回答")
@click.option("--no-answer", is_flag=True, help="即使配置开启了大模型，也只打印命中原文")
@click.option("--scope", type=click.Choice(list(SCOPES)), help="只查公司材料 company，或投资通识 knowledge")
@click.option("--doc-type", type=click.Choice(list(DOC_TYPES)), help="材料类型：filing、ir_qa、survey、research、news、theory、beginner")
@click.option("--company", default=None, help="公司简称，精确匹配")
@click.option("--code", "stock_code", default=None, help="6 位股票代码")
@click.option("--year", type=click.IntRange(1900, 2100), help="报告期或发布年份")
def query(
    question: tuple[str, ...],
    config_path: str | None,
    mode: str | None,
    top_k: int | None,
    answer: bool,
    no_answer: bool,
    scope: str | None,
    doc_type: str | None,
    company: str | None,
    stock_code: str | None,
    year: int | None,
) -> None:
    if answer and no_answer:
        raise click.UsageError("--answer 和 --no-answer 不能同时使用")
    cfg = load_config(config_path)
    if mode:
        cfg["retrieval"]["mode"] = mode
    if top_k is not None:
        cfg["retrieval"]["top_k"] = top_k

    text = " ".join(question).strip()
    if not text:
        raise click.UsageError("问题不能为空")
    filt = SearchFilter.parse(
        scope=scope,
        doc_type=doc_type,
        company=company,
        stock_code=stock_code,
        year=year,
    )
    hits = retrieve(
        cfg,
        text,
        scope=filt.scope or None,
        doc_type=filt.doc_type or None,
        company=filt.company or None,
        stock_code=filt.stock_code or None,
        year=filt.year or None,
    )
    print_hits(cfg["retrieval"]["mode"], hits, filt)

    want_answer = answer or (cfg["llm"].get("enabled") and not no_answer)
    if no_answer:
        want_answer = False
    if want_answer:
        print("回答:")
        print(generate_answer(cfg, text, hits))


@cli.command(help="启动查询接口，和 python api.py 是同一个服务。")
@_config_option
@click.option("--host", default="127.0.0.1", show_default=True, help="监听地址，默认只接受本机访问")
@click.option("--port", type=click.IntRange(min=1, max=65535), default=8000, show_default=True, help="监听端口")
def serve(config_path: str | None, host: str, port: int) -> None:
    from api import serve as run_api

    run_api(config_path, host, port)


@cli.command(help="按 sqlite 中的 chunk 原文重算向量并重建 Chroma。BM25 和原文都不动。")
@_config_option
@click.option("--force", is_flag=True, help="即使模型名和向量条数都没变，也重新写入")
def reembed(config_path: str | None, force: bool) -> None:
    cfg = load_config(config_path)
    reembed_vectors(cfg, force=force)


if __name__ == "__main__":
    cli()
