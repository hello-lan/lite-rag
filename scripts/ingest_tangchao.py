"""把 corpus/tangchao 的 Markdown 切成预切分 JSON，再交给 ingest-split 写入索引。

切分在本脚本完成；lite_rag 只导入 JSON。
文稿记为投资理论（knowledge/investment_theory），人物唐朝，场合文章。
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
from lite_rag.ingest import ingest_split_documents

log = logging.getLogger("tangchao")

SPLIT_NAME = "tangchao-md-v1"
SOURCE_DIR = ROOT / "corpus" / "tangchao"
SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}

# 滑窗参数（lite_rag 不再切分，改这里后需重跑本脚本）
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TABLE_AWARE = True
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]
_INDEX = re.compile(r"^\d+\s*[-–—.]?\s*")
_DATE = re.compile(r"(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})")
_EDGE = " \t-–—.．"

_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")
_TABLE_CAPTION = re.compile(r"(单位|注[:：]|备注|说明)")


def _is_md_table_row(line: str) -> bool:
    return bool(_MD_TABLE_ROW.match(line))


def _is_md_table_sep(line: str) -> bool:
    return bool(_MD_TABLE_SEP.match(line))


def _looks_like_table_caption(line: str) -> bool:
    text = (line or "").strip()
    if not text or _is_md_table_row(text):
        return False
    if len(text) > 80:
        return False
    return bool(_TABLE_CAPTION.search(text))


def _find_md_table_end(plain: list[str], start: int) -> int | None:
    """若 plain[start] 起是合法 Markdown 表，返回表结束后的下标；否则 None。"""
    if start >= len(plain) or not _is_md_table_row(plain[start]):
        return None
    index = start
    saw_sep = False
    while index < len(plain) and (
        _is_md_table_row(plain[index]) or _is_md_table_sep(plain[index])
    ):
        if _is_md_table_sep(plain[index]):
            saw_sep = True
        index += 1
    if saw_sep and index - start >= 2:
        return index
    return None


def _peel_trailing_captions(prose: str) -> tuple[str, str]:
    """把 prose 末尾紧邻的单位/注说明剥给表格。返回 (剩余正文, 说明)。"""
    lines = prose.splitlines(keepends=True)
    keep = len(lines)
    while keep > 0 and _looks_like_table_caption(lines[keep - 1].rstrip("\r\n")):
        keep -= 1
    if keep == len(lines):
        return prose, ""
    left = "".join(lines[:keep]).strip("\n")
    caption = "".join(lines[keep:]).strip("\n")
    return left, caption


def partition_prose_and_tables(text: str) -> list[tuple[str, str]]:
    """拆成 (kind, content) 块；kind 为 prose 或 table。表前短单位/注说明吸附到表。"""
    if not (text or "").strip():
        return []

    lines = text.splitlines(keepends=True)
    plain = [line.rstrip("\r\n") for line in lines]
    n = len(plain)
    blocks: list[tuple[str, str]] = []
    index = 0

    def emit(kind: str, start: int, end: int) -> None:
        if start >= end:
            return
        chunk = "".join(lines[start:end]).strip("\n")
        if chunk.strip():
            blocks.append((kind, chunk))

    while index < n:
        table_end = _find_md_table_end(plain, index)
        if table_end is not None:
            table_body = "".join(lines[index:table_end]).strip("\n")
            if blocks and blocks[-1][0] == "prose":
                left, caption = _peel_trailing_captions(blocks[-1][1])
                if caption:
                    if left.strip():
                        blocks[-1] = ("prose", left)
                    else:
                        blocks.pop()
                    table_body = f"{caption}\n{table_body}"
            blocks.append(("table", table_body))
            index = table_end
            continue

        start = index
        index += 1
        while index < n and _find_md_table_end(plain, index) is None:
            if _is_md_table_row(plain[index]):
                # 单行管道文本但构不成表：当正文吃掉，避免卡死。
                index += 1
                continue
            index += 1
        emit("prose", start, index)

    return blocks


def chunk_markdown_table(table_text: str, chunk_size: int) -> list[str]:
    """小表整表保留；大表按行分包，每块重复 header + separator。"""
    text = (table_text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    lines = text.splitlines()
    caption_lines: list[str] = []
    cursor = 0
    while cursor < len(lines) and not _is_md_table_row(lines[cursor]):
        caption_lines.append(lines[cursor])
        cursor += 1
    table_lines = lines[cursor:]
    if len(table_lines) < 2:
        return [text]

    header = table_lines[0]
    separator = table_lines[1] if _is_md_table_sep(table_lines[1]) else None
    body_start = 2 if separator is not None else 1
    if separator is None:
        return [text] if len(text) <= chunk_size else _chunk_lines_fit(lines, chunk_size)

    body_rows = table_lines[body_start:]
    prefix_parts = caption_lines + [header, separator]
    prefix = "\n".join(prefix_parts)
    prefix_len = len(prefix)

    if not body_rows:
        return [text]

    chunks: list[str] = []
    batch: list[str] = []
    batch_len = prefix_len

    def flush() -> None:
        nonlocal batch, batch_len
        if not batch:
            return
        chunks.append("\n".join(prefix_parts + batch))
        batch = []
        batch_len = prefix_len

    for row in body_rows:
        row_cost = 1 + len(row)
        if batch and batch_len + row_cost > chunk_size:
            flush()
        if not batch and prefix_len + 1 + len(row) > chunk_size:
            chunks.append("\n".join(prefix_parts + [row]))
            batch = []
            batch_len = prefix_len
            continue
        batch.append(row)
        batch_len += row_cost
    flush()
    return chunks


def _chunk_lines_fit(lines: list[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    batch: list[str] = []
    size = 0
    for line in lines:
        cost = len(line) + (1 if batch else 0)
        if batch and size + cost > chunk_size:
            chunks.append("\n".join(batch))
            batch = []
            size = 0
            cost = len(line)
        batch.append(line)
        size += cost
    if batch:
        chunks.append("\n".join(batch))
    return chunks


def _split_prose(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    kwargs = dict(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=list(SEPARATORS),
        length_function=len,
    )
    try:
        splitter = RecursiveCharacterTextSplitter(keep_separator=True, **kwargs)
    except TypeError:
        splitter = RecursiveCharacterTextSplitter(**kwargs)
    return [part.strip() for part in splitter.split_text(text) if part and part.strip()]


def split_document(text: str) -> list[str]:
    """切分正文；默认表感知：Markdown 表整表或按行切并重复表头。"""
    body = (text or "").strip()
    if not body:
        return []

    if not TABLE_AWARE:
        return _split_prose(body)

    parts: list[str] = []
    for kind, content in partition_prose_and_tables(body):
        if kind == "table":
            parts.extend(chunk_markdown_table(content, CHUNK_SIZE))
        else:
            parts.extend(_split_prose(content))
    return parts


def iter_source_files(docs_dir: Path, *, suffixes: set[str] | None = None) -> list[Path]:
    wanted = {item.lower() for item in (suffixes or SUPPORTED_SUFFIXES)}
    if not docs_dir.exists():
        return []
    files = [
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in wanted
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


def md_to_text(src: str) -> str:
    import marko
    from marko.ext.gfm import GFM

    document = marko.Markdown(extensions=[GFM]).parse(src)
    return _render_markdown(document).strip()


def _render_markdown(node) -> str:
    if isinstance(node, str):
        return node
    name = type(node).__name__
    children = getattr(node, "children", None)
    if name == "TableRow" and children and not isinstance(children, str):
        cells = [_render_markdown(child).strip() for child in children]
        return " | ".join(cell for cell in cells if cell) + "\n"
    if children is None:
        return ""
    inner = children if isinstance(children, str) else "".join(_render_markdown(child) for child in children)
    if name == "Link":
        dest = getattr(node, "dest", "") or ""
        if dest and dest not in inner:
            return f"{inner} ({dest})"
        return inner
    if name == "ListItem":
        return "- " + inner.strip() + "\n"
    if name == "TableCell":
        return inner.strip()
    if name in {"Paragraph", "Heading", "Quote", "FencedCode", "CodeBlock", "BlankLine", "ThematicBreak"}:
        text = inner.strip()
        return f"{text}\n\n" if text else "\n"
    if name in {"List", "Table", "TableHead", "TableBody"}:
        return inner + ("\n" if inner and not inner.endswith("\n") else "")
    return inner


def pdf_to_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page).strip()


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
    chunks = split_document(text) if text else []
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
