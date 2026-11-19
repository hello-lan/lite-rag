"""把 corpus/tangchao 的 Markdown 写成预切分 JSON，并交给 ingest-split 写入索引。

不改 lite_rag 的入库流程。文稿记为投资理论（knowledge/investment_theory），人物唐朝，场合文章。
标题和发布日期从文件名解析。图片、image_texts.json 和压缩包不处理。

示例：

    python scripts/ingest_tangchao.py --check
    python scripts/ingest_tangchao.py
    python scripts/ingest_tangchao.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite_rag.config import configure_stdio, load_config, resolve_path
from lite_rag.documents import iter_source_files, md_to_text, split_document
from lite_rag.ingest import ingest_split_documents

log = logging.getLogger("tangchao")

SPLIT_NAME = "tangchao-md-v1"
SOURCE_DIR = ROOT / "corpus" / "tangchao"
_INDEX = re.compile(r"^\d+\s*[-–—.]?\s*")
_DATE = re.compile(r"(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})")
_EDGE = " \t-–—.．"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 corpus/tangchao 的 Markdown 写入向量库")
    parser.add_argument("--check", action="store_true", help="只生成并校验 JSON，不加载模型，也不写向量")
    parser.add_argument("--force", action="store_true", help="即使文本和元信息没变，也重新嵌入并写入")
    args = parser.parse_args()
    if args.check and args.force:
        parser.error("--check 和 --force 不能同时使用")
    return args


def parse_article(stem: str) -> tuple[str, str, int]:
    """从文件名取出标题、发布日期和年份。没有合法日期时日期为空、年份为 0。"""
    title = stem.strip()
    published = ""
    year = 0
    matched = _DATE.search(title)
    if matched:
        year_num, month, day = (int(matched.group(part)) for part in (1, 2, 3))
        try:
            published = date(year_num, month, day).isoformat()
        except ValueError:
            published = ""
        else:
            year = year_num
            title = f"{title[: matched.start()]} {title[matched.end() :]}"
    title = _INDEX.sub("", title.strip())
    title = title.strip(_EDGE)
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        title = stem.strip()
    return title, published, year


def article_meta(stem: str) -> dict:
    title, published, year = parse_article(stem)
    meta = {
        "scope": "knowledge",
        "doc_type": "investment_theory",
        "title": title,
        "figure": "唐朝",
        "occasion": "article",
    }
    if published:
        meta["published_on"] = published
        meta["year"] = year
    return meta


def split_root(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["paths"]["split_dir"]) / "tangchao"


def json_path(out_dir: Path, source: Path) -> Path:
    relative = source.resolve().relative_to(SOURCE_DIR.resolve())
    return out_dir / relative.with_suffix(".json")


def write_article(cfg: dict, out_dir: Path, source: Path) -> tuple[dict, int]:
    raw = source.read_text(encoding="utf-8-sig")
    text = md_to_text(raw)
    chunks = split_document(text, cfg) if text else []
    meta = article_meta(source.stem)
    if not chunks:
        log.warning("跳过（没有抽出文本）: %s", source.name)
        return meta, 0
    payload = {
        "split": SPLIT_NAME,
        "meta": meta,
        "chunks": [{"text": chunk} for chunk in chunks],
    }
    destination = json_path(out_dir, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta, len(chunks)


def remove_stale(out_dir: Path, live: set[Path]) -> int:
    if not out_dir.exists():
        return 0
    removed = 0
    for path in out_dir.rglob("*.json"):
        if path.resolve() in live:
            continue
        path.unlink()
        removed += 1
        log.info("删除已不存在的文稿 JSON: %s", path.name)
    return removed


def export_json(cfg: dict) -> tuple[int, int]:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"文稿目录不存在: {SOURCE_DIR}")
    files = [
        path
        for path in iter_source_files(SOURCE_DIR)
        if path.suffix.lower() in {".md", ".markdown"}
    ]
    if not files:
        raise SystemExit(f"{SOURCE_DIR} 里没有 Markdown。")
    out_dir = split_root(cfg)
    live: set[Path] = set()
    written = 0
    chunk_count = 0
    for source in files:
        meta, count = write_article(cfg, out_dir, source)
        if not count:
            continue
        destination = json_path(out_dir, source)
        live.add(destination.resolve())
        written += 1
        chunk_count += count
        when = meta.get("published_on") or "-"
        log.info("%s  %s  %s  chunk %s", source.name, meta["title"], when, count)
    removed = remove_stale(out_dir, live)
    log.info("JSON 已写到 %s：%s 篇，%s 个 chunk，删除过期 %s", out_dir, written, chunk_count, removed)
    return written, chunk_count


def main() -> None:
    configure_stdio()
    args = parse_args()
    cfg = load_config()
    export_json(cfg)
    ingest_split_documents(cfg, force=args.force, check=args.check)


if __name__ == "__main__":
    main()
