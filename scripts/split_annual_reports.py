"""按 PDF 书签大纲初步切开 A 股年报，写成可 ingest-split 的 JSON。

默认读取 corpus/01_raw/annual_reports/{代码}/*.pdf，
写出到 corpus/03_splited/annual_reports/{代码}/{同名}.json。

切分标记 annual-outline-v4（chunk_size=600, overlap=100）：
- PyMuPDF（fitz）读书签大纲、兜底抽正文
- pdfplumber 抽表格并转成 Markdown，与正文按页交错
- depth<=2 的书签作边界（即最多三级目录）；同页多节按标题在页内切开
- 超长小节表感知滑窗（整表或按行切并重复表头）；无书签时整份报告兜底

加 --dump-segments 时，额外把「书签切段、尚未滑窗」的中间结果写到
corpus/02_processed/annual_reports/{代码}/{年报名}/，每段一个 Markdown 文件。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("split_annual_reports")

SPLIT_ID = "annual-outline-v4"
KNOWN_SECTIONS = ("profile", "mda", "material", "governance", "financials", "notes", "other")

# 超长小节滑窗参数（lite_rag 不再切分，改这里后需重跑本脚本）
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TABLE_AWARE = True
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


class SectionMapError(ValueError):
    """书签路径无法映射到已知 section，整份 PDF 不应写出 JSON。"""

DEFAULT_SOURCE = ROOT / "corpus" / "01_raw" / "annual_reports"
DEFAULT_OUTPUT = ROOT / "corpus" / "03_splited" / "annual_reports"
DEFAULT_SEGMENTS = ROOT / "corpus" / "02_processed" / "annual_reports"
MAX_OUTLINE_DEPTH = 2  # depth 0..2 → 最多三级目录（章 -> 节 -> 小节）
_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_MULTI_SPACE = re.compile(r"\s+")

_YEAR = re.compile(r"(20\d{2})")
_CODE_DIR = re.compile(r"^\d{6}$")
# 年报「公司信息」常见写法：标签后换行或空格接简称。
_STOCK_SHORT = re.compile(r"(?:股票简称|公司的中文简称)\s*[：:]*\s*(\S+)")
_NOISE = re.compile(
    r"[\s□■☑☒√✔✕×]*适用[\s□■☑☒√✔✕×]*不适用|"
    r"[\s□■☑☒√✔✕×]*是[\s□■☑☒√✔✕×]*否|"
    r"\s*无\s*$"
)
_CHECKBOX = re.compile(r"[□■☑☒√✔✕×]+")
_AMOUNT = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^-?\d+\.\d+$|^-?\d{4,}$")
_PIPE = re.compile(r"\|")

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


@dataclass(frozen=True)
class OutlineNode:
    title: str
    depth: int
    page: int  # 0-based
    path: tuple[str, ...]  # 从一级到当前的标题链


@dataclass(frozen=True)
class Segment:
    path: tuple[str, ...]
    start_page: int
    end_page: int
    text: str


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按书签大纲切开年报 PDF（PyMuPDF + pdfplumber）")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="年报 PDF 根目录，默认 corpus/01_raw/annual_reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON 输出根目录，默认 corpus/03_splited/annual_reports",
    )
    parser.add_argument(
        "--dump-segments",
        action="store_true",
        help="写出书签切段中间结果（滑窗前），每段一个 Markdown 文件",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=DEFAULT_SEGMENTS,
        help="中间结果目录，默认 corpus/02_processed/annual_reports",
    )
    parser.add_argument("--code", action="append", default=[], help="只处理该股票代码，可重复")
    parser.add_argument("pdfs", nargs="*", type=Path, help="额外指定的单个 PDF（可选）")
    return parser.parse_args()


def safe_filename(title: str, limit: int = 60) -> str:
    text = _UNSAFE_NAME.sub("_", title)
    text = _MULTI_SPACE.sub(" ", text).strip(" ._")
    text = text.replace("/", "-").replace("\\", "-")
    if not text:
        text = "untitled"
    if len(text) > limit:
        text = text[: limit - 1].rstrip(" ._") + "…"
    return text


def segment_dir_for(pdf: Path, source_root: Path, segments_root: Path) -> Path:
    try:
        rel = pdf.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = Path(pdf.name)
    return segments_root / rel.with_suffix("")


def write_segment_file(
    dest_dir: Path,
    index: int,
    segment: Segment,
    meta: dict,
) -> Path:
    section = map_section(segment.path)
    entity = map_entity(segment.path, section)
    heading = breadcrumb(segment.path) or "未命名章节"
    leaf = segment.path[-1] if segment.path else "未命名章节"
    name = f"{index:03d}_{safe_filename(leaf)}.md"
    path = dest_dir / name

    front = {
        "title": clip_title(segment_title(segment.path, segment.start_page, segment.end_page)),
        "chapter": heading,
        "page_start": segment.start_page + 1,
        "page_end": segment.end_page + 1,
        "section": section,
        "entity": entity,
        "stock_code": meta.get("stock_code", ""),
        "stock_name": meta.get("stock_name", ""),
        "period": meta.get("period", ""),
        "scope": meta.get("scope", "financial_report"),
        "doc_type": meta.get("doc_type", "filing"),
    }
    front = {key: value for key, value in front.items() if value not in ("", None)}
    dumped = yaml.safe_dump(front, allow_unicode=True, sort_keys=False).rstrip()
    body = "\n".join(
        [
            "---",
            dumped,
            "---",
            "",
            f"# {heading}",
            "",
            segment.text.strip(),
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path


def clean_title(title: str) -> str:
    text = (title or "").replace("\u3000", " ").strip()
    text = _NOISE.sub("", text)
    text = _CHECKBOX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" .-–—\t")
    return text


def read_outline(doc) -> list[OutlineNode]:
    """用 PyMuPDF get_toc 读书签，展开为有序节点（含标题路径）。

    toc 项为 [level, title, page]，page 为 1-based。
    """
    try:
        toc = doc.get_toc(simple=True) or []
    except Exception as exc:
        log.warning("读取书签失败: %s", exc)
        return []

    nodes: list[OutlineNode] = []
    stack: list[str] = []
    for entry in toc:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        level, raw_title, page1 = entry[0], entry[1], entry[2]
        try:
            level_num = int(level)
            page1_num = int(page1)
        except (TypeError, ValueError):
            continue
        if level_num < 1 or page1_num < 1:
            continue
        title = clean_title(str(raw_title))
        if not title:
            continue
        depth = level_num - 1
        stack = stack[:depth]
        stack.append(title)
        nodes.append(
            OutlineNode(
                title=title,
                depth=depth,
                page=page1_num - 1,
                path=tuple(stack),
            )
        )
    return nodes


def select_cuts(nodes: list[OutlineNode], *, max_depth: int = MAX_OUTLINE_DEPTH) -> list[OutlineNode]:
    """保留 depth<=max_depth 的节点，按大纲顺序全部作为切点（同页多节也保留）。"""
    return [node for node in nodes if node.depth <= max_depth]


def find_title_pos(text: str, title: str, *, start: int = 0) -> int:
    """在 text[start:] 中定位标题；先精确匹配，再忽略空白。找不到返回 -1。"""
    if not text or not title or start >= len(text):
        return -1
    region = text[start:]
    exact = region.find(title)
    if exact >= 0:
        return start + exact

    compact_title = re.sub(r"\s+", "", title)
    if not compact_title:
        return -1
    compact_chars: list[str] = []
    compact_to_orig: list[int] = []
    for index, char in enumerate(region):
        if not char.isspace():
            compact_chars.append(char)
            compact_to_orig.append(start + index)
    compact = "".join(compact_chars)
    pos = compact.find(compact_title)
    if pos < 0:
        return -1
    return compact_to_orig[pos]


def split_page_by_titles(
    text: str,
    cuts: list[OutlineNode],
) -> list[tuple[OutlineNode | None, str]]:
    """按同页切点标题切开一页。owner=None 表示标题前的前文（属上一节续页）。"""
    if not cuts:
        return [(None, text)] if text else []

    found: list[tuple[OutlineNode, int]] = []
    cursor = 0
    for cut in cuts:
        pos = find_title_pos(text, cut.title, start=cursor)
        if pos < 0:
            pos = find_title_pos(text, cut.title, start=0)
        if pos < 0:
            continue
        if found and pos < found[-1][1]:
            continue
        found.append((cut, pos))
        cursor = pos + max(len(cut.title), 1)

    if not found:
        return [(None, text)] if text else []

    parts: list[tuple[OutlineNode | None, str]] = []
    first_pos = found[0][1]
    if first_pos > 0:
        preamble = text[:first_pos]
        if preamble.strip():
            parts.append((None, preamble))
    for index, (cut, pos) in enumerate(found):
        end = found[index + 1][1] if index + 1 < len(found) else len(text)
        chunk = text[pos:end]
        if chunk.strip():
            parts.append((cut, chunk))
    return parts


def build_outline_segments(
    cuts: list[OutlineNode],
    page_texts: list[str],
) -> list[Segment]:
    """按书签切段；同页多节用标题在页内切开，标题前正文归上一节。"""
    if not cuts:
        return []

    buckets: list[list[str]] = [[] for _ in cuts]
    start_pages: list[int | None] = [None] * len(cuts)
    end_pages: list[int | None] = [None] * len(cuts)
    cut_index = {id(cut): index for index, cut in enumerate(cuts)}
    by_page: dict[int, list[OutlineNode]] = {}
    for cut in cuts:
        by_page.setdefault(cut.page, []).append(cut)

    current: int | None = None
    for page, text in enumerate(page_texts):
        if not (text or "").strip():
            continue
        on_page = by_page.get(page, [])
        if not on_page:
            if current is not None:
                buckets[current].append(text)
                end_pages[current] = page
            continue

        pending_preamble: str | None = None
        for owner, part in split_page_by_titles(text, on_page):
            if not part.strip():
                continue
            if owner is None:
                if current is not None:
                    buckets[current].append(part)
                    end_pages[current] = page
                else:
                    pending_preamble = part
                continue
            index = cut_index[id(owner)]
            if pending_preamble:
                buckets[index].append(pending_preamble)
                pending_preamble = None
            buckets[index].append(part)
            if start_pages[index] is None:
                start_pages[index] = page
            end_pages[index] = page
            current = index

    segments: list[Segment] = []
    for index, cut in enumerate(cuts):
        body = "\n\n".join(part for part in buckets[index] if part.strip()).strip()
        if not body:
            continue
        start = start_pages[index] if start_pages[index] is not None else cut.page
        end = end_pages[index] if end_pages[index] is not None else start
        segments.append(Segment(path=cut.path, start_page=start, end_page=end, text=body))
    return segments


def looks_like_amount(cell: str) -> bool:
    text = (cell or "").strip().replace(" ", "")
    if not text or text in {"-", "—", "–", "／", "/"}:
        return False
    return bool(_AMOUNT.match(text))


def normalize_cell(cell) -> str:
    if cell is None:
        return ""
    text = str(cell).replace("\r", "\n")
    parts = [part.strip() for part in text.split("\n") if part.strip()]
    if not parts:
        return ""
    if all(re.fullmatch(r"[\d,.\-%]+", part) for part in parts):
        return "".join(parts)
    return "".join(parts) if all(len(part) <= 2 or not part.isascii() for part in parts) else " ".join(parts)


def clean_table(rows: list[list]) -> list[list[str]]:
    if not rows:
        return []
    norm = [[normalize_cell(cell) for cell in row] for row in rows]
    width = max((len(row) for row in norm), default=0)
    if width == 0:
        return []
    norm = [row + [""] * (width - len(row)) for row in norm]
    keep = [index for index in range(width) if any(row[index] for row in norm)]
    if not keep:
        return []
    norm = [[row[index] for index in keep] for row in norm]

    merged: list[list[str]] = []
    for row in norm:
        if merged and _is_continuation_row(row, merged[-1]):
            prev = merged[-1]
            for index, cell in enumerate(row):
                if not cell:
                    continue
                prev[index] = f"{prev[index]}{cell}" if prev[index] else cell
            continue
        merged.append(list(row))

    amount_col = _first_amount_col(merged)
    if amount_col > 1:
        collapsed: list[list[str]] = []
        for row in merged:
            label = "".join(row[:amount_col]).strip()
            collapsed.append([label, *row[amount_col:]])
        merged = collapsed

    return [row for row in merged if any(cell.strip() for cell in row)]


def _is_continuation_row(row: list[str], prev: list[str]) -> bool:
    if any(looks_like_amount(cell) for cell in row):
        return False
    if not any(cell.strip() for cell in row):
        return False
    return any(looks_like_amount(cell) for cell in prev)


def _first_amount_col(rows: list[list[str]]) -> int:
    if not rows:
        return 0
    width = len(rows[0])
    body = rows[1:] or rows
    for index in range(width):
        if any(looks_like_amount(row[index]) for row in body):
            return index
    return 1 if width > 1 else 0


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = clean_table(rows)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    lines = [
        "| " + " | ".join(_md_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _md_cell(cell: str) -> str:
    return _PIPE.sub("\\|", (cell or "").replace("\n", " ").strip())


def _crop_text(page, top: float, bottom: float) -> str:
    if bottom - top < 2:
        return ""
    try:
        cropped = page.within_bbox((0, top, page.width, bottom), relative=False)
    except Exception:
        return ""
    return (cropped.extract_text() or "").strip()


def _page_tables(page) -> list:
    """兼容不同 pdfplumber 版本的 find_tables 返回值。"""
    try:
        found = page.find_tables()
    except Exception:
        return []
    if not found:
        return []
    if isinstance(found, list):
        return found
    tables = getattr(found, "tables", None)
    return list(tables) if tables else []


def page_to_markdown(page) -> str:
    """抽一页：表格外保留正文，表格转成 Markdown（pdfplumber）。"""
    tables = _page_tables(page)
    if not tables:
        return (page.extract_text() or "").strip()

    tables = sorted(tables, key=lambda item: (item.bbox[1], item.bbox[0]))
    parts: list[str] = []
    cursor = 0.0
    for table in tables:
        _x0, top, _x1, bottom = table.bbox
        above = _crop_text(page, cursor, top)
        if above:
            parts.append(above)
        try:
            raw = table.extract()
        except Exception:
            raw = None
        md = table_to_markdown(raw or [])
        if md:
            parts.append(md)
        cursor = max(cursor, bottom)
    below = _crop_text(page, cursor, page.height)
    if below:
        parts.append(below)
    return "\n\n".join(parts).strip()


def extract_pages_plumber(plumber_doc, start: int, end: int) -> str:
    """按页抽取正文；能识别的表格写成 Markdown。"""
    parts: list[str] = []
    page_count = len(plumber_doc.pages)
    for index in range(start, end + 1):
        if index >= page_count:
            break
        text = page_to_markdown(plumber_doc.pages[index])
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def extract_pages_fitz(doc, start: int, end: int) -> str:
    """无 pdfplumber 时的兜底：PyMuPDF 纯文本。"""
    parts: list[str] = []
    for index in range(start, end + 1):
        if index >= doc.page_count:
            break
        text = (doc.load_page(index).get_text() or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def extract_pages(doc, plumber_doc, start: int, end: int) -> str:
    if plumber_doc is not None:
        return extract_pages_plumber(plumber_doc, start, end)
    return extract_pages_fitz(doc, start, end)


# 一级标题关键词（含主板 / 北交所 / 旧版常见变体）。
_MDA_L0 = ("管理层讨论", "经营情况讨论", "经营情况分析", "董事会报告", "经营评述")
_GOV_L0 = (
    "公司治理",
    "股份变动",
    "股东情况",
    "优先股",
    "董事、监事、高级管理人员",
    "投资者保护",
)
_MATERIAL_L0 = (
    "重要事项",
    "重大事项",
    "重大事件",
    "环境和社会",
    "环境与社会",
    "环境与社会责任",
    "社会责任",
    "债券",
)
_PROFILE_L0 = (
    "公司简介",
    "公司概况",
    "主要财务指标",
    "会计数据和财务指标",
    "重要提示",
    "目录和释义",
)
# 治理与 ESG 合并章里，这些子标题归 material。
_ESG_SUB = (
    "环境信息披露",
    "环境信息",
    "社会责任情况",
    "社会责任",
    "乡村振兴",
    "脱贫攻坚",
    "巩固拓展脱贫",
)
_STATEMENT_KEYS = (
    "资产负债表",
    "利润表",
    "现金流量表",
    "所有者权益变动表",
    "股东权益变动表",
)
# 去掉书签前的「一、」「1、」「（1）」等编号，再做精确匹配。
_HEADING_NUM = re.compile(
    r"^[（(]?\d+[）)]?[、.\s]*|"
    r"^[一二三四五六七八九十百]+[、.\s]*"
)
# 书签把附注抬成一级时的兜底（不依赖「财务报告」父节点）。
_NOTE_ORPHAN_KEYS = (
    "会计政策",
    "会计估计",
    "编制基础",
    "合并财务报表项目",
    "财务报表项目附注",
    "财务报表项目注释",
    "关联方",
    "或有事项",
    "承诺事项",
    "资产负债表日后",
    "金融工具",
    "公允价值",
    "股份支付",
    "合并范围",
    "在其他主体中的权益",
    "分部信息",
    "分部报告",
    "补充资料",
    "非经常性损益明细",
    "净资产收益率及每股收益",
    "母公司财务报表主要项目",
    "公司财务报表主要项目",
    "政府补助",
    "资本管理",
    "公司基本情况",
    "税项",
    "境内外会计准则",
)


def _has_any(text: str, keys: tuple[str, ...]) -> bool:
    return any(key in text for key in keys)


def _path_has_esg(parts: tuple[str, ...]) -> bool:
    return any(_has_any(part, _ESG_SUB) for part in parts)


def _heading_core(text: str) -> str:
    return _HEADING_NUM.sub("", (text or "").strip()).strip()


def _part_is_audit(part: str) -> bool:
    return _heading_core(part) == "审计报告"


def _part_is_statements_block(part: str) -> bool:
    """「二、财务报表」这一级；不含编制基础 / 项目注释。"""
    return _heading_core(part) == "财务报表"


def _part_is_statement_table(part: str) -> bool:
    core = _heading_core(part)
    for key in _STATEMENT_KEYS:
        if core in {key, f"合并{key}", f"母公司{key}"}:
            return True
    return False


def _path_is_financials(path: tuple[str, ...]) -> bool:
    return any(
        _part_is_audit(part) or _part_is_statements_block(part) or _part_is_statement_table(part)
        for part in path
    )


def map_section(path: tuple[str, ...]) -> str:
    """按书签路径映射到 filing section。

    优先看一级标题；财务报告再拆报表/附注。
    治理与 ESG 合并章按子标题分流；书签层级错乱时用附注/报表关键词兜底。
    """
    if not path:
        return ""
    l0 = path[0]
    joined = " / ".join(path)

    if _has_any(l0, ("财务报告", "财务会计报告")):
        return map_financial_section(path)

    # 例如「第四节 公司治理、环境和社会」：ESG 子节 → material，其余 → governance。
    if "公司治理" in l0 and _has_any(l0, ("环境", "社会")):
        if len(path) > 1 and _path_has_esg(path[1:]):
            return "material"
        return "governance"

    if _has_any(l0, _MDA_L0) or l0.startswith("致股东"):
        return "mda"
    if _has_any(l0, _GOV_L0):
        return "governance"
    if _has_any(l0, _MATERIAL_L0):
        return "material"
    if _has_any(l0, _PROFILE_L0):
        return "profile"

    if _path_is_financials(path):
        return "financials"
    if _has_any(joined, _NOTE_ORPHAN_KEYS):
        return "notes"
    # 美的等把杂项抬成一级「四、其他」；只认去编号后恰为「其他」，避免误伤附注子节。
    if _heading_core(l0) == "其他":
        return "other"
    return ""


def map_financial_section(path: tuple[str, ...]) -> str:
    if _path_is_financials(path):
        return "financials"
    return "notes"


def map_entity(path: tuple[str, ...], section: str) -> str:
    if section not in {"financials", "notes"}:
        return ""
    joined = " / ".join(path)
    if "母公司" in joined:
        return "parent"
    # 美的等用「公司财务报表主要项目附注」指母公司，无「母公司」字样。
    if "公司财务报表主要项目" in joined and "合并" not in joined:
        return "parent"
    if "合并" in joined:
        return "consolidated"
    return ""


def breadcrumb(path: tuple[str, ...]) -> str:
    return " -> ".join(part for part in path if part)


def require_sections(segments: list[Segment], *, pdf: Path) -> None:
    """任一段无法映射 section 则抛错，阻止写出 JSON。"""
    missing = [segment for segment in segments if not map_section(segment.path)]
    if not missing:
        return
    known = "、".join(KNOWN_SECTIONS)
    lines = [
        f"{pdf.name} 有 {len(missing)} 段无法映射 section，已中止（不写 JSON）",
        f"允许的 section: {known}",
        "未映射书签（请在 map_section / 相关关键词表中补充）:",
    ]
    for segment in missing:
        heading = breadcrumb(segment.path) or "(空路径)"
        pages = f"p{segment.start_page + 1}-{segment.end_page + 1}"
        lines.append(f"  - {heading}  ({pages})")
    raise SectionMapError("\n".join(lines))


def segment_title(path: tuple[str, ...], start: int, end: int) -> str:
    label = breadcrumb(path) or "未命名章节"
    return f"{label} (p{start + 1}-{end + 1})"


def clip_title(title: str, limit: int = 120) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) > limit:
        return title[: limit - 1] + "…"
    return title


def build_chunks(segment: Segment) -> tuple[list[dict], int]:
    """返回 chunks 与因超长而滑窗的次数（0 或 1）。

    调用前须已通过 require_sections；此处 section 必非空。
    """
    section = map_section(segment.path)
    if not section:
        raise SectionMapError(
            f"缺少 section: {breadcrumb(segment.path) or '(空路径)'} "
            f"(p{segment.start_page + 1}-{segment.end_page + 1})"
        )
    entity = map_entity(segment.path, section)
    heading = breadcrumb(segment.path) or "未命名章节"
    base_title = clip_title(segment_title(segment.path, segment.start_page, segment.end_page))
    body = segment.text.strip()
    if not body:
        return [], 0

    windowed = 0
    if len(body) <= CHUNK_SIZE:
        parts = [body]
    else:
        parts = split_document(body)
        windowed = 1

    chunks: list[dict] = []
    total = len(parts)
    for index, part in enumerate(parts):
        text = part.strip()
        if not text:
            continue
        if index == 0 and heading and not text.startswith(f"【{heading}】"):
            text = f"【{heading}】\n\n{text}"
        title = base_title if total == 1 else clip_title(f"{base_title} [{index + 1}/{total}]")
        item: dict = {"text": text, "title": title, "section": section}
        if entity:
            item["entity"] = entity
        chunks.append(item)
    return chunks, windowed


def extract_stock_name(doc, *, max_pages: int = 20) -> str:
    """从年报前几页「公司信息」抽取股票简称；找不到则返回空串。"""
    limit = min(max_pages, doc.page_count)
    for index in range(limit):
        text = doc[index].get_text("text") or ""
        matched = _STOCK_SHORT.search(text)
        if matched:
            name = matched.group(1).strip().strip("（）()")
            if name and name not in ("不适用", "无", "—", "-", "－"):
                return name
    return ""


def parse_meta(pdf: Path, source_root: Path, *, stock_name: str = "") -> dict:
    try:
        relative = pdf.resolve().relative_to(source_root.resolve())
        code = relative.parts[0] if relative.parts and _CODE_DIR.match(relative.parts[0]) else ""
    except ValueError:
        code = pdf.parent.name if _CODE_DIR.match(pdf.parent.name) else ""
    if not code:
        matched = re.match(r"^(\d{6})", pdf.stem)
        code = matched.group(1) if matched else ""

    years = _YEAR.findall(pdf.stem)
    year = int(years[-1]) if years else 0
    stock_name = (stock_name or "").strip()
    if stock_name and year:
        title = f"{stock_name}{year}年年度报告"
    elif year:
        title = f"{code}{year}年年度报告" if code else f"{year}年年度报告"
    else:
        title = pdf.stem

    meta: dict = {
        "scope": "financial_report",
        "doc_type": "filing",
        "title": title,
    }
    if stock_name:
        meta["stock_name"] = stock_name
    if code:
        meta["stock_code"] = code
    if year:
        meta["period"] = str(year)
    return meta


def collect_pdfs(source: Path, codes: list[str], extra: list[Path]) -> list[tuple[Path, Path]]:
    """返回 (pdf, source_root) 列表。"""
    found: list[tuple[Path, Path]] = []
    if extra:
        for raw in extra:
            path = raw.expanduser().resolve()
            if not path.is_file() or path.suffix.lower() != ".pdf":
                raise SystemExit(f"不是 PDF: {raw}")
            root = path.parent.parent if _CODE_DIR.match(path.parent.name) else path.parent
            found.append((path, root))
        return found

    source = source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"找不到年报目录: {source}")
    wanted = set(codes) if codes else None
    for pdf in sorted(source.rglob("*.pdf")):
        if wanted is not None:
            try:
                rel = pdf.relative_to(source)
                code = rel.parts[0] if rel.parts else ""
            except ValueError:
                code = ""
            if code not in wanted and pdf.parent.name not in wanted:
                continue
        found.append((pdf, source))
    if not found:
        raise SystemExit("没有找到要处理的 PDF")
    return found


def split_pdf(
    pdf: Path,
    source_root: Path,
    output_root: Path,
    *,
    dump_segments: bool = False,
    segments_root: Path | None = None,
) -> Path | None:
    import pdfplumber
    import pymupdf

    doc = pymupdf.open(str(pdf))
    try:
        page_count = doc.page_count
        nodes = read_outline(doc)
        meta = parse_meta(pdf, source_root, stock_name=extract_stock_name(doc))
        chunks: list[dict] = []
        segments = 0
        windowed_segments = 0
        dumped = 0
        fallback = False

        segment_dir: Path | None = None
        with pdfplumber.open(str(pdf)) as plumber_doc:
            page_texts = [
                extract_pages(doc, plumber_doc, index, index) for index in range(page_count)
            ]
            if nodes:
                outline_segments = build_outline_segments(select_cuts(nodes), page_texts)
            else:
                # 无书签：整份报告兜底，再滑窗（对齐参考脚本）。
                fallback = True
                body = "\n\n".join(text for text in page_texts if text.strip()).strip()
                outline_segments = (
                    [Segment(path=("全文",), start_page=0, end_page=page_count - 1, text=body)]
                    if body
                    else []
                )
                log.warning("无书签大纲，改用全文滑窗兜底: %s", pdf.name)

            # 先校验 section，避免半截 dump / 半截 JSON。
            require_sections(outline_segments, pdf=pdf)

            if dump_segments:
                if segments_root is None:
                    raise ValueError("dump_segments 需要 segments_root")
                segment_dir = segment_dir_for(pdf, source_root, segments_root)
                if segment_dir.exists():
                    shutil.rmtree(segment_dir)
                segment_dir.mkdir(parents=True, exist_ok=True)

            for segment in outline_segments:
                if segment_dir is not None:
                    write_segment_file(segment_dir, segments + 1, segment, meta)
                    dumped += 1
                parts, windowed = build_chunks(segment)
                chunks.extend(parts)
                segments += 1
                windowed_segments += windowed
    finally:
        doc.close()

    if not chunks:
        log.warning("跳过（未抽出文字）: %s", pdf)
        return None

    try:
        rel = pdf.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = Path(pdf.name)
    dest = output_root / rel.with_suffix(".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": SPLIT_ID, "meta": meta, "chunks": chunks}
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    note = ""
    if segment_dir is not None:
        note = f"  中间段 {dumped} → {segment_dir}"
    if fallback:
        note += "  [fallback]"
    log.info(
        "%s  书签 %s  切段 %s  滑窗段 %s  chunk %s  → %s%s",
        pdf.name,
        len(nodes),
        segments,
        windowed_segments,
        len(chunks),
        dest.relative_to(output_root) if dest.is_relative_to(output_root) else dest,
        note,
    )
    return dest


def main() -> None:
    configure_stdio()
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    segments_root = args.segments_dir.resolve() if args.dump_segments else None
    jobs = collect_pdfs(source, args.code, args.pdfs)
    ok = 0
    skipped = 0
    failed = 0
    for pdf, root in jobs:
        try:
            result = split_pdf(
                pdf,
                root,
                output,
                dump_segments=args.dump_segments,
                segments_root=segments_root,
            )
        except SectionMapError as exc:
            log.error("%s", exc)
            failed += 1
            continue
        if result is None:
            skipped += 1
        else:
            ok += 1
    log.info("完成 %s，跳过 %s，失败 %s", ok, skipped, failed)
    if failed or ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
