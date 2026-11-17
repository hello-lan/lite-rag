"""财报 RAG 命令行：入库、查询、按当前模型重算向量、启动查询接口。"""

from __future__ import annotations

import click

from lite_rag.answer import generate_answer
from lite_rag.config import configure_stdio, load_config
from lite_rag.ingest import ingest_documents, reembed_vectors
from lite_rag.search import retrieve


@click.group(help="本地财报 RAG：入库、查询、按新模型重算向量、启动查询接口。")
def cli() -> None:
    configure_stdio()


def _config_option(func):
    return click.option("--config", "config_path", default=None, help="配置文件路径，默认使用项目里的 config.yaml")(func)


def print_hits(mode: str, hits: list[dict]) -> None:
    print(f"模式: {mode}")
    if not hits:
        print("没有命中。可以换一种说法，或改用 --mode bm25 / --mode vector 对比。")
        return
    print(f"命中 {len(hits)} 条\n")
    for hit in hits:
        score = hit["score"]
        score_text = f"{score:.4f}" if isinstance(score, float) else "-"
        print(
            f"[{hit['rank']}] id={hit['id']}  {hit['score_name']}={score_text}  "
            f"{hit['source_name']}  chunk #{hit['chunk_idx']}"
        )
        print(hit["chunk_text"])
        print()


@cli.command(help="把 docs/ 中的 PDF 和 Markdown 写入财报 RAG 索引。")
@_config_option
@click.option("--force", is_flag=True, help="忽略未改动的文档，全部重新切分并写入向量")
@click.option("--check", is_flag=True, help="只解析和切分，不下载模型，也不写数据库")
@click.option("--bm25-only", is_flag=True, help="不重新解析文档，只按 sqlite 里的原文重建 BM25")
def ingest(config_path: str | None, force: bool, check: bool, bm25_only: bool) -> None:
    cfg = load_config(config_path)
    ingest_documents(cfg, force=force, check=check, bm25_only=bm25_only)


@cli.command(help="按 BM25、向量或 RRF 混合检索财报 chunk，可选调用大模型生成回答。")
@click.argument("question", nargs=-1, required=True)
@_config_option
@click.option("--mode", type=click.Choice(["hybrid", "bm25", "vector"]), help="覆盖 config.yaml 里的检索模式")
@click.option("--top-k", type=click.IntRange(min=1), help="最终返回的 chunk 数")
@click.option("--answer", is_flag=True, help="用配置里的大模型根据命中材料生成回答")
@click.option("--no-answer", is_flag=True, help="即使配置开启了大模型，也只打印命中原文")
def query(
    question: tuple[str, ...],
    config_path: str | None,
    mode: str | None,
    top_k: int | None,
    answer: bool,
    no_answer: bool,
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
    hits = retrieve(cfg, text)
    print_hits(cfg["retrieval"]["mode"], hits)

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
