"""按问答切开投资者关系活动记录 PDF，写成可 ingest-split 的 JSON。

默认读取 corpus/01_raw/ir_activities/{代码}/*.pdf，
写出到 corpus/03_splited/ir_activities/{代码}/{同名}.json。

切分标记 ir-qa-v1（chunk_size=800, overlap=100）：
- pdfplumber 还原跨页 2 列表单；时间/地点后再跳一行，左列含「内容」起拼到附件清单前
- 破表时把「不像表头的左列散文」并入上一单元格，避免最长碎片误选
- 无 2 列表时回退纯文本定位；封面过短再用附件页
- 切分：按章节 / 问句标记 / 编号问句定位切点，整段切片（不拆问与答）
- 定位不到正文时：整份纯文本直接滚动切割
- 切出的超长段仍会再滑窗
- 加 --llm 时：PDF 全文交给 LLM，由其定位正文并切段（跳过规则定位/切分）

加 --dump-segments 时，额外把「问答切段、尚未滑窗」的中间结果写到
corpus/02_processed/ir_activities/{代码}/{文件名}/，每段一个 Markdown 文件。

示例：

    python scripts/split_ir_activities.py
    python scripts/split_ir_activities.py --code 000333
    python scripts/split_ir_activities.py --dump-segments path/to/one.pdf
    python scripts/split_ir_activities.py --llm --code 300760
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

from scripts.agent_util import AgentError, SimpleAgent

log = logging.getLogger("split_ir_activities")

SPLIT_ID = "ir-qa-v1"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

DEFAULT_SOURCE = ROOT / "corpus" / "01_raw" / "ir_activities"
DEFAULT_OUTPUT = ROOT / "corpus" / "03_splited" / "ir_activities"
DEFAULT_SEGMENTS = ROOT / "corpus" / "02_processed" / "ir_activities"

_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_MULTI_SPACE = re.compile(r"\s+")
_CODE_DIR = re.compile(r"^\d{6}$")
_DATE_SUFFIX = re.compile(r"_(20\d{2})-(\d{2})-(\d{2})$")
_YEAR = re.compile(r"(20\d{2})")

_STOCK_CODE = re.compile(r"证券代码\s*[：:]\s*(\d{6})")
_STOCK_SHORT = re.compile(r"证券简称\s*[：:]\s*(\S+)")
_ACTIVITY_TIME = re.compile(
    r"时间\s*\n?\s*(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
)

# 勾选符须出现在类型名之前（表单为「☑类型」；去空白后「□A█B」会误伤后缀匹配）
_CHECKED_CHAR = r"[■█☑√✔●◆◼▣✔]"
_ACTIVITY_TYPES = (
    "特定对象调研",
    "分析师会议",
    "媒体采访",
    "业绩说明会",
    "新闻发布会",
    "路演活动",
    "现场参观",
    "其他",
)

# 表单行标签（去空白后匹配）
_LABEL_TIME = re.compile(r"^时间$")
_LABEL_PLACE = re.compile(r"^地点$")
_LABEL_STAFF = re.compile(r"接待|人员姓名")
_LABEL_FOOTER = re.compile(r"附件清单|^日期$")
_LABEL_CATEGORY = re.compile(r"活动类别|投资者关系活动类别")
_LABEL_PARTICIPANT = re.compile(r"参与单位")
_KNOWN_FORM_LABEL = re.compile(
    r"活动类别|参与单位|时间|地点|接待|人员姓名|主要内容|附件清单|^日期$|证券代码|证券简称|编号"
)
# 表内正文过短则改走附件页（封面只写目录时常见）
_TABLE_BODY_MIN = 200

# 问句行首切点（含 Q1： / Q1. / 问：）；答留在同一段，不单独切开
_QA_Q = re.compile(r"^\s*Q\s*(?:\d{1,2})?\s*[：:.\.．]\s*(.*)$", re.I)
_QA_ASK = re.compile(r"^\s*问\s*(?:\d{1,2})?\s*[：:.\.．]\s*(.*)$")
# 裁表尾时判断后面是否还有问答（含答行标记）
_HAS_QA_MARKER = re.compile(r"^\s*(?:Q|A|问|答)\s*(?:\d{1,2})?\s*[：:.\.．]", re.I)

# 编号问答：1. / 1． / 1、（点号后空格可省略：1.问题；排除 16.2% 小数）
_NUM_DOT = re.compile(r"^\s*(\d{1,2})[\.．](?!\d)\s*(\S.*)$")
_NUM_DOT_ONLY = re.compile(r"^\s*(\d{1,2})[\.．]\s*$")
_NUM_DUN = re.compile(r"^\s*(\d{1,2})、\s*(\S.*)$")

# 章节标题：一、二、 / （一） / 1、经营要点汇报
_SECTION_CN = re.compile(r"^\s*([一二三四五六七八九十百]+)、\s*(\S.*)$")
_SECTION_CN_PAREN = re.compile(r"^\s*[（(]([一二三四五六七八九十]+)[）)]\s*(\S.*)$")
_SECTION_NUM_DUN = re.compile(r"^\s*(\d{1,2})、\s*(\S.*)$")

_PAGE_NOISE = re.compile(r"^\s*\d+\s*$")
_ATTACH_TITLE = re.compile(
    r"附件\s*\d*\s*(?:[\n\r]+\s*)?(?:调研活动记录|投资者关系活动记录|活动记录)",
    re.I,
)
_TEXT_BODY_MARK = re.compile(
    r"(?:投资者关系活动)?主\s*要\s*内\s*容\s*介\s*绍"
)

LLM_SPLIT_SYSTEM_PROMPT = """\
你是投资者关系活动记录切分助手。用户会给你一份 PDF 抽出的全文（含表头、名单、页眉页脚等噪声）。

请完成两步，然后只输出 JSON：

1) 定位正文：只保留「主要内容介绍」里的实质内容。忽略并丢弃：
   - 证券代码/简称、编号、活动类别勾选、时间、地点、接待人员、参与单位名单
   - 附件清单、页码、页眉页脚里重复的公司名/表格标题
   - 其它与问答或经营要点无关的表单噪声
2) 切分：对定位后的正文按语义切成若干片段（一条问答一对、一节经营要点等均可自成一段）。
不要编造原文没有的内容；尽量保留原文措辞与数字。

只输出一个 JSON 字符串数组，不要 Markdown 解释。格式示例：
["第一段完整原文…", "第二段完整原文…"]
每项是一段完整文本（可含问与答），不要再拆成 question/answer 对象。
若全文几乎没有可保留内容，输出 []。
"""


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str
    index: int  # 1-based
    kind: str = "qa"  # qa | section | window


@dataclass(frozen=True)
class ActivityInfo:
    stock_code: str
    stock_name: str
    published_on: str
    year: int
    activity_type: str
    title: str


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按问答切开投资者关系活动记录 PDF（PyMuPDF）"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="投关 PDF 根目录，默认 corpus/01_raw/ir_activities",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON 输出根目录，默认 corpus/03_splited/ir_activities",
    )
    parser.add_argument(
        "--dump-segments",
        action="store_true",
        help="写出问答切段中间结果（滑窗前），每段一个 Markdown 文件",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=DEFAULT_SEGMENTS,
        help="中间结果目录，默认 corpus/02_processed/ir_activities",
    )
    parser.add_argument("--code", action="append", default=[], help="只处理该股票代码，可重复")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="全文交给 LLM：定位正文并切分（默认用规则定位+切分）",
    )
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


def clip_title(title: str, limit: int = 120) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) > limit:
        return title[: limit - 1] + "…"
    return title


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\u3000", " ").replace("\r\n", "\n").replace("\r", "\n")
    # 合并被 PDF 拆开的换行空白，但保留空行分段
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def extract_pdf_text(pdf: Path) -> str:
    """全文纯文本（meta / 附件回退用）。"""
    import pymupdf

    doc = pymupdf.open(str(pdf))
    try:
        parts: list[str] = []
        for index in range(doc.page_count):
            page_text = doc[index].get_text("text") or ""
            parts.append(page_text)
        return normalize_whitespace("\n".join(parts))
    finally:
        doc.close()


def _cell_text(cell: object) -> str:
    if cell is None:
        return ""
    return normalize_whitespace(str(cell)).strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _is_known_form_label(label: str) -> bool:
    compact = _compact(label)
    if not compact:
        return False
    if len(compact) > 40:
        return False
    return bool(_KNOWN_FORM_LABEL.search(compact))


def extract_form_rows(pdf: Path) -> list[tuple[str, str]]:
    """按页抽取 2 列表单行；跨页续行与破表散文行并入上一单元格。"""
    import pdfplumber

    raw_rows: list[tuple[str, str]] = []
    with pdfplumber.open(str(pdf)) as doc:
        for page in doc.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                width = max((len(row) for row in table), default=0)
                if width != 2:
                    continue
                for row in table:
                    if len(row) < 2:
                        continue
                    label = _cell_text(row[0])
                    value = _cell_text(row[1])
                    if not label and not value:
                        continue
                    raw_rows.append((label, value))

    if not raw_rows:
        return []

    stitched: list[tuple[str, str]] = []
    for label, value in raw_rows:
        if not label and stitched:
            prev_label, prev_value = stitched[-1]
            joined = f"{prev_value}\n{value}".strip() if value else prev_value
            stitched[-1] = (prev_label, joined)
            continue

        if (
            label
            and not value
            and stitched
            and _LABEL_FOOTER.search(_compact(stitched[-1][0]))
            and len(_compact(label)) <= 8
        ):
            prev_label, prev_value = stitched[-1]
            stitched[-1] = (f"{prev_label}{label}", prev_value)
            continue

        # 破表：左列其实是正文续写（不像表头字段）→ 并入上一行
        if stitched and label and not _is_known_form_label(label):
            prev_label, prev_value = stitched[-1]
            piece = label if not value else f"{label}\n{value}"
            # 同一段被拆表重复抽出时跳过，避免正文翻倍
            if _compact(piece) and _compact(piece) in _compact(prev_value):
                continue
            stitched[-1] = (prev_label, f"{prev_value}\n{piece}".strip())
            continue

        stitched.append((label, value))
    return stitched


def _row_is_time_or_place(label: str) -> bool:
    compact = _compact(label)
    return bool(_LABEL_TIME.match(compact) or _LABEL_PLACE.match(compact))


def extract_table_body(rows: list[tuple[str, str]]) -> str:
    """正文行：时间/地点之后再跳过一行，且左列含「内容」；拼到附件清单/日期前。"""
    if not rows:
        return ""

    anchor = -1
    for index, (label, _value) in enumerate(rows):
        if _row_is_time_or_place(label):
            anchor = index
    if anchor < 0:
        return ""

    # 时间/地点后再跳过一行（常见为接待人员），再找左列带「内容」的行
    after_skip = anchor + 2
    start = -1
    search_from = max(after_skip, 0)
    for index in range(search_from, len(rows)):
        if "内容" in _compact(rows[index][0]):
            start = index
            break
    if start < 0:
        # 容错：跳过行数不准时，仍在时间/地点之后找「内容」
        for index in range(anchor + 1, len(rows)):
            if "内容" in _compact(rows[index][0]):
                start = index
                break
    if start < 0:
        return ""

    parts: list[str] = []
    for index in range(start, len(rows)):
        label, value = rows[index]
        compact = _compact(label)
        if _LABEL_FOOTER.search(compact):
            break
        if _looks_like_roster_row(label, value):
            break
        if index == start:
            # 正文首行：只要右列值（左列是「…内容介绍」标签）
            if value:
                parts.append(value)
            continue
        if _is_known_form_label(label) and "内容" not in compact:
            if _LABEL_STAFF.search(compact) or _LABEL_CATEGORY.search(compact) or _LABEL_PARTICIPANT.search(compact):
                continue
        if value:
            parts.append(value)
        elif label and not _is_known_form_label(label):
            parts.append(label)
    return "\n".join(parts).strip()


def _looks_like_roster_row(label: str, value: str) -> bool:
    """参会机构名单行：短标签+短值，且多像机构名。"""
    lab = (label or "").strip()
    val = (value or "").strip()
    if not lab or not val:
        return False
    if len(lab) > 60 or len(val) > 60:
        return False
    if _is_known_form_label(lab):
        return False
    # 英文机构名 / 含基金证券公司等
    if re.search(r"(Fund|Capital|Asset|Limited|证券|基金|资管|银行)", lab) and re.search(
        r"(Fund|Capital|Asset|Limited|证券|基金|资管|银行)", val
    ):
        return True
    # 纯英文短词对
    if re.fullmatch(r"[A-Za-z0-9&.,'\- ]{3,60}", lab) and re.fullmatch(
        r"[A-Za-z0-9&.,'\- \u4e00-\u9fff]{2,60}", val
    ):
        return True
    return False


def extract_attachment_text(pdf: Path) -> str:
    """封面表正文过短时：收集无 2 列表页的纯文本（附件问答）。"""
    import pdfplumber
    import pymupdf

    attach_pages: list[int] = []
    with pdfplumber.open(str(pdf)) as doc:
        saw_form = False
        for index, page in enumerate(doc.pages):
            tables = page.extract_tables() or []
            two_col = [t for t in tables if t and max((len(r) for r in t), default=0) == 2]
            wide = [t for t in tables if t and max((len(r) for r in t), default=0) >= 3]
            if two_col:
                saw_form = True
                continue
            if wide and saw_form:
                break
            if not two_col:
                attach_pages.append(index)

    if not attach_pages:
        return ""

    doc = pymupdf.open(str(pdf))
    try:
        parts = [(doc[i].get_text("text") or "") for i in attach_pages if i < doc.page_count]
    finally:
        doc.close()
    text = normalize_whitespace("\n".join(parts)).strip()
    matched = _ATTACH_TITLE.search(text)
    if matched:
        return text[matched.end() :].strip()
    return text


def trim_trailing_roster(text: str) -> str:
    """去掉正文末尾误吸入的参会机构名单。"""
    lines = (text or "").splitlines()
    if not lines:
        return ""

    cut = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        compact = _compact(stripped)
        if compact in ("公司名称", "机构名称") or stripped in ("公司名称", "机构名称"):
            cut = index
            break
        if stripped.startswith("附件清单"):
            cut = index
            break
        # 连续英文机构名：从该行起若后面也多为名单则截断
        if re.fullmatch(r"[A-Za-z0-9&.,'\-() ]{8,80}", stripped) and re.search(
            r"(Fund|Capital|Asset|Limited|Management|Securities|Investment)",
            stripped,
            re.I,
        ):
            rest = lines[index : index + 5]
            score = 0
            for item in rest:
                s = item.strip()
                if re.search(r"(Fund|Capital|Asset|Limited|证券|基金|资管)", s, re.I):
                    score += 1
            if score >= 3:
                cut = index
                break
    return "\n".join(lines[:cut]).strip()


def extract_text_body(plain: str) -> str:
    """无可用 2 列表时：从纯文本里裁出正文区。"""
    text = (plain or "").strip()
    if not text:
        return ""

    main = _TEXT_BODY_MARK.search(text)
    if main:
        body = text[main.end() :].strip()
        # 去掉末尾附件清单 / 日期表尾（若后面没有问答则裁）
        cut = len(body)
        lines = body.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("附件清单") or stripped in ("日期",):
                rest = "\n".join(lines[index + 1 :])
                if not (_HAS_QA_MARKER.search(rest) or _ATTACH_TITLE.search(rest)):
                    cut = sum(len(x) + 1 for x in lines[:index])
                    break
        body = body[:cut].strip()
        if body:
            return body

    attach = _ATTACH_TITLE.search(text)
    if attach:
        return text[attach.end() :].strip()

    first_q = re.search(r"(?m)^\s*(?:问|Q)\s*(?:\d{1,2})?\s*[：:.\.．]", text)
    if first_q and first_q.start() > 80:
        return text[first_q.start() :].strip()

    # 定位不到明确正文起点时返回空，由上层改走全文滚切
    return ""


def locate_body_from_pdf(pdf: Path) -> tuple[str, str]:
    """返回 (body, mode_hint)。

    能定位：table / attach / text / table-short。
    定位不到：unlocated（整份纯文本，交给 window 滚切）。
    """
    rows = extract_form_rows(pdf)
    table_body = extract_table_body(rows) if rows else ""
    table_body = trim_trailing_roster(table_body)
    plain = extract_pdf_text(pdf)

    if len(table_body) >= _TABLE_BODY_MIN:
        # 表内只有经营要点、没有互动问答时，尝试用纯文本补互动段
        if ("经营要点" in table_body or "情况汇报" in table_body) and not re.search(
            r"(?m)^\s*(?:问|Q)\s*[：:.\.．]", table_body
        ):
            text_body = trim_trailing_roster(extract_text_body(plain) or "")
            if len(text_body) > len(table_body) + 200:
                return text_body, "text"
        return table_body, "table"

    attach = extract_attachment_text(pdf)
    if attach:
        trimmed = trim_trailing_roster(extract_text_body(attach) or attach)
        if len(trimmed) > len(table_body):
            return trimmed, "attach"

    text_body = trim_trailing_roster(extract_text_body(plain) or "")
    if text_body and len(text_body) > len(table_body):
        return text_body, "text"

    if table_body:
        return table_body, "table-short"

    full = trim_trailing_roster((plain or "").strip())
    return full, "unlocated"


def date_from_filename(stem: str) -> tuple[str, int]:
    matched = _DATE_SUFFIX.search(stem)
    if matched:
        year, month, day = matched.groups()
        return f"{year}-{month}-{day}", int(year)
    years = _YEAR.findall(stem)
    if years:
        year = int(years[-1])
        return "", year
    return "", 0


def date_from_body(text: str) -> tuple[str, int]:
    matched = _ACTIVITY_TIME.search(text)
    if not matched:
        return "", 0
    year, month, day = matched.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}", int(year)


def detect_activity_type(text: str) -> str:
    """从勾选行推断活动类别；多选用顿号连接。"""
    found: list[str] = []
    compact = re.sub(r"\s+", "", text[:2500])
    for name in _ACTIVITY_TYPES:
        if re.search(_CHECKED_CHAR + re.escape(name), compact):
            found.append(name)
            continue
        pattern = re.compile(_CHECKED_CHAR + r"\s*" + re.escape(name))
        for line in text.splitlines()[:80]:
            if pattern.search(line):
                found.append(name)
                break
    return "、".join(found)


def extract_header_fields(text: str) -> tuple[str, str]:
    code = ""
    name = ""
    matched = _STOCK_CODE.search(text)
    if matched:
        code = matched.group(1)
    matched = _STOCK_SHORT.search(text)
    if matched:
        name = matched.group(1).strip().strip("（）()；;，,")
        name = re.split(r"[\s\|]", name)[0]
    return code, name


def _is_questionish(text: str) -> bool:
    """编号项是否像「问题」而非财务条目/章节小点。"""
    text = (text or "").strip()
    if not text:
        return False
    first = text.splitlines()[0].strip()
    if "？" in first or "?" in first:
        return True
    # 财务/数据条目：营业收入：xxx
    if re.search(r"(营业收入|归母|净利润|现金流|毛利率|费用).{0,8}[：:]", first):
        return False
    if "：" in first or ":" in first:
        return False
    # 无问号时仅接受较短问句标题
    return 4 <= len(first) <= 60


def _segment_title(text: str, index: int) -> str:
    first = ""
    for line in (text or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    return clip_title(first) if first else f"段落{index}"


def _blocks_from_starts(lines: list[str], starts: list[int]) -> list[QAPair]:
    """按行号切点切片：每段保留原文整块，不拆问/答。"""
    if not starts:
        return []
    pairs: list[QAPair] = []
    for order, start in enumerate(starts):
        end = starts[order + 1] if order + 1 < len(starts) else len(lines)
        block = [
            line.rstrip()
            for line in lines[start:end]
            if not _PAGE_NOISE.match(line)
        ]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(block)).strip()
        if not text:
            continue
        index = len(pairs) + 1
        pairs.append(
            QAPair(
                question=_segment_title(text, index),
                answer=text,
                index=index,
                kind="window",
            )
        )
    return pairs


def split_qa_pairs(body: str) -> list[QAPair]:
    """按 Q：/Q1：/问： 等问句行定位切点，整段切开（含答，不拆字段）。"""
    lines = body.splitlines()
    starts: list[int] = []
    for index, line in enumerate(lines):
        if _QA_Q.match(line) or _QA_ASK.match(line):
            starts.append(index)
    if len(starts) < 1:
        return []
    return _blocks_from_starts(lines, starts)


def _collect_numbered_starts(lines: list[str], pattern: re.Pattern[str]) -> list[tuple[int, int, str]]:
    """返回 (line_index, number, title)。支持「1.」独占一行、正文在下一行。"""
    starts: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        matched = pattern.match(line)
        if matched:
            number = int(matched.group(1))
            title = matched.group(2).strip()
            starts.append((index, number, title))
            continue
        if pattern is _NUM_DOT:
            alone = _NUM_DOT_ONLY.match(line)
            if alone and index + 1 < len(lines):
                nxt = lines[index + 1].strip()
                if nxt and not _NUM_DOT.match(lines[index + 1]) and not _NUM_DOT_ONLY.match(lines[index + 1]):
                    starts.append((index, int(alone.group(1)), nxt))
    return starts


def _filter_sequential(starts: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """保留从 1 起大体递增的编号序列，去掉答案里的子编号噪声。"""
    if not starts:
        return []
    kept: list[tuple[int, int, str]] = []
    expect = 1
    for item in starts:
        _line, number, title = item
        if number == expect or (kept and number == kept[-1][1] + 1):
            kept.append(item)
            expect = number + 1
            continue
        # 允许偶尔跳号（如缺 3）但必须接近
        if kept and kept[-1][1] < number <= kept[-1][1] + 2:
            kept.append(item)
            expect = number + 1
            continue
        # 尚未开始：必须是 1，或问句式的 1/2
        if not kept and number == 1:
            kept.append(item)
            expect = 2
            continue
        if not kept and number <= 2 and _is_questionish(title):
            kept.append(item)
            expect = number + 1
    return kept


def split_numbered_pairs(body: str) -> list[QAPair]:
    """仅切开像问句的编号项；财务条目/无问号小节不走这条。整段保留，不拆问/答。"""
    lines = body.splitlines()
    starts = _filter_sequential(_collect_numbered_starts(lines, _NUM_DOT))
    starts = [item for item in starts if _is_questionish(item[2])]
    if len(starts) < 2:
        raw = _collect_numbered_starts(lines, _NUM_DUN)
        raw = [item for item in raw if _is_questionish(item[2])]
        starts = _filter_sequential(raw)
    # 至少 2 条问句，且过半带问号，才认定是编号问答
    if len(starts) < 2:
        return []
    with_mark = sum(1 for _i, _n, title in starts if ("？" in title or "?" in title))
    if with_mark < max(2, (len(starts) + 1) // 2):
        return []
    return _blocks_from_starts(lines, [item[0] for item in starts])


def split_section_pairs(body: str) -> list[QAPair]:
    """按中文章节 / 经营要点等非问答结构切开；每段含标题+正文原文。"""
    lines = body.splitlines()
    starts: list[tuple[int, str]] = []

    cn_hits = []
    for index, line in enumerate(lines):
        matched = _SECTION_CN.match(line)
        if matched:
            cn_hits.append((index, matched.group(0).strip()))
    if len(cn_hits) >= 2:
        starts = cn_hits
    else:
        paren_hits = []
        for index, line in enumerate(lines):
            matched = _SECTION_CN_PAREN.match(line)
            if matched:
                title = matched.group(0).strip()
                if len(title) <= 40:
                    paren_hits.append((index, title))
        if len(paren_hits) >= 2:
            starts = paren_hits
        else:
            dun_hits = []
            for index, line in enumerate(lines):
                matched = _SECTION_NUM_DUN.match(line)
                if not matched:
                    continue
                title = matched.group(2).strip()
                # 章节感：短标题、无问号；排除「1、相比…」这类答内子点
                if "？" in title or "?" in title:
                    continue
                if len(title) > 40:
                    continue
                if re.match(r"^(相比|在|通过|其中|另外|此外|同时)", title):
                    continue
                dun_hits.append((index, matched.group(0).strip()))
            # 需要从 1、开始且至少 2 段
            filtered: list[tuple[int, str]] = []
            expect = 1
            for index, title in dun_hits:
                num = int(_SECTION_NUM_DUN.match(lines[index]).group(1))
                if num == expect:
                    filtered.append((index, title))
                    expect = num + 1
            if len(filtered) >= 2:
                starts = filtered

    if len(starts) < 2:
        return []
    return _blocks_from_starts(lines, [item[0] for item in starts])


def _refine_sections_with_qa(sections: list[QAPair]) -> tuple[list[QAPair], str]:
    """经营要点等保持整段；互动/问答段再按问句切点切开。"""
    refined: list[QAPair] = []
    used_qa = False
    for section in sections:
        title = section.question
        if any(key in title for key in ("互动", "问答", "提问")):
            # 去掉章节标题行后再找问句切点，避免标题粘在第一段上
            lines = section.answer.splitlines()
            rest = "\n".join(lines[1:]).strip() if len(lines) > 1 else section.answer
            sub = split_qa_pairs(rest)
            if len(sub) < 1:
                sub = split_numbered_pairs(rest)
            if len(sub) >= 1:
                used_qa = True
                for item in sub:
                    refined.append(
                        QAPair(
                            question=item.question,
                            answer=item.answer,
                            index=len(refined) + 1,
                            kind="window",
                        )
                    )
                continue
        refined.append(
            QAPair(
                question=section.question,
                answer=section.answer,
                index=len(refined) + 1,
                kind="window",
            )
        )
    strategy = "section+qa" if used_qa else "section"
    return refined, strategy


def split_pairs(body: str) -> tuple[list[QAPair], str]:
    """返回 (pairs, strategy)。

    规则只负责找切点并整段切片（不拆问/答）；
    匹配不上则 strategy=window，由 build_chunks 对整段正文滚动切割。
    """
    cleaned = (body or "").strip()
    if not cleaned:
        return [], "empty"

    sections = split_section_pairs(cleaned)
    if len(sections) >= 2:
        return _refine_sections_with_qa(sections)

    qa = split_qa_pairs(cleaned)
    if len(qa) >= 1:
        return qa, "qa"

    numbered = split_numbered_pairs(cleaned)
    if len(numbered) >= 1:
        return numbered, "numbered"

    # 样式匹配不上：整段正文交给滑窗
    return [QAPair(question="正文", answer=cleaned, index=1, kind="window")], "window"


def _pairs_from_llm_payload(data: object) -> list[QAPair]:
    """LLM 返回字符串列表 → QAPair（kind=window，正文在 answer）。"""
    if not isinstance(data, list):
        raise AgentError(f"LLM 返回不是 JSON 数组: {type(data).__name__}")
    pairs: list[QAPair] = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            text = item.strip()
        else:
            raise AgentError(
                f"LLM 数组元素须为字符串，收到 {type(item).__name__}"
            )
        if not text:
            continue
        pairs.append(
            QAPair(
                question=_segment_title(text, index),
                answer=text,
                index=len(pairs) + 1,
                kind="window",
            )
        )
    return pairs


def split_pairs_with_llm(full_text: str, agent: SimpleAgent) -> tuple[list[QAPair], str]:
    """把 PDF 全文交给 LLM：定位正文并切段，返回 (pairs, \"llm\")。"""
    cleaned = (full_text or "").strip()
    if not cleaned:
        return [], "empty"
    data = agent.run_json(cleaned)
    pairs = _pairs_from_llm_payload(data)
    if not pairs:
        raise AgentError("LLM 未切出任何段落")
    return pairs, "llm"


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


def context_prefix(info: ActivityInfo) -> str:
    bits = [bit for bit in (info.stock_name, info.published_on, info.activity_type) if bit]
    if not bits:
        return ""
    return "【" + " · ".join(bits) + "】"


def format_pair_text(pair: QAPair, info: ActivityInfo) -> str:
    """片段正文在 answer；question 仅作标题，不拼回正文。"""
    prefix = context_prefix(info)
    body = (pair.answer or pair.question or "").strip()
    if prefix:
        return f"{prefix}\n\n{body}"
    return body


def pair_title(pair: QAPair) -> str:
    if pair.kind == "window" and (not pair.question or pair.question == "正文"):
        return "正文"
    return clip_title(pair.question)


def build_chunks(pairs: list[QAPair], info: ActivityInfo) -> tuple[list[dict], int]:
    """返回 chunks 与被滑窗的段数。

    每条 chunk.title 统一用整份文件的标题（来自文件名），不再用片段首行。
    window 策略：对整段正文做滚动切割（chunk_size/overlap），不按问答/章节再切。
    """
    chunks: list[dict] = []
    windowed = 0
    doc_title = clip_title(info.title) if info.title else "正文"

    # 样式未匹配的整篇正文（唯一一段且标题为「正文」）：一次滚完
    if (
        len(pairs) == 1
        and pairs[0].kind == "window"
        and (not pairs[0].question or pairs[0].question == "正文")
    ):
        base = format_pair_text(pairs[0], info).strip()
        if not base:
            return [], 0
        parts = _split_prose(base) if len(base) > CHUNK_SIZE else [base]
        windowed = 1 if len(parts) > 1 else 0
        for part in parts:
            text = part.strip()
            if not text:
                continue
            chunks.append({"text": text, "title": doc_title})
        return chunks, windowed

    for pair in pairs:
        base = format_pair_text(pair, info).strip()
        if not base:
            continue
        if len(base) <= CHUNK_SIZE:
            parts = [base]
        else:
            parts = _split_prose(base)
            windowed += 1
        for part in parts:
            text = part.strip()
            if not text:
                continue
            chunks.append({"text": text, "title": doc_title})
    return chunks, windowed


def parse_activity_info(pdf: Path, source_root: Path, text: str) -> ActivityInfo:
    try:
        relative = pdf.resolve().relative_to(source_root.resolve())
        code_from_path = relative.parts[0] if relative.parts and _CODE_DIR.match(relative.parts[0]) else ""
    except ValueError:
        code_from_path = pdf.parent.name if _CODE_DIR.match(pdf.parent.name) else ""

    header_code, stock_name = extract_header_fields(text)
    stock_code = header_code or code_from_path
    if not stock_code:
        matched = re.match(r"^(\d{6})", pdf.stem)
        stock_code = matched.group(1) if matched else ""

    published_on, year = date_from_filename(pdf.stem)
    body_date, body_year = date_from_body(text)
    if not published_on and body_date:
        published_on = body_date
    if not year and body_year:
        year = body_year
    if published_on and not year:
        year = int(published_on[:4])

    activity_type = detect_activity_type(text)

    # 标题：去掉代码前缀（可重复）与日期后缀
    stem = pdf.stem
    if stock_code:
        while stem.startswith(f"{stock_code}_") or stem.startswith(stock_code):
            if stem.startswith(f"{stock_code}_"):
                stem = stem[len(stock_code) + 1 :]
            else:
                stem = stem[len(stock_code) :]
            stem = stem.lstrip("_")
    stem = _DATE_SUFFIX.sub("", stem)
    title = stem.strip("_ ") or pdf.stem
    if stock_name and stock_name not in title:
        title = f"{stock_name}{title}" if title else stock_name

    return ActivityInfo(
        stock_code=stock_code,
        stock_name=stock_name,
        published_on=published_on,
        year=year,
        activity_type=activity_type,
        title=title,
    )


def meta_from_info(info: ActivityInfo) -> dict:
    meta: dict = {
        "scope": "investor_materials",
        "doc_type": "ir_activities",
        "title": info.title,
    }
    if info.stock_name:
        meta["stock_name"] = info.stock_name
    if info.stock_code:
        meta["stock_code"] = info.stock_code
    if info.year:
        meta["year"] = info.year
    if info.published_on:
        meta["published_on"] = info.published_on
    return meta


def collect_pdfs(source: Path, codes: list[str], extra: list[Path]) -> list[tuple[Path, Path]]:
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
        raise SystemExit(f"找不到投关目录: {source}")
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


def output_rel_for(pdf: Path, source_root: Path, stock_code: str = "") -> Path:
    """相对输出路径：优先 {证券代码}/{文件名}.json，保证有代码这一层目录。"""
    name = pdf.with_suffix(".json").name
    code = (stock_code or "").strip()
    if code and _CODE_DIR.match(code):
        return Path(code) / name
    try:
        rel = pdf.resolve().relative_to(source_root.resolve())
        if len(rel.parts) >= 2 and _CODE_DIR.match(rel.parts[0]):
            return rel.with_suffix(".json")
        parent_code = pdf.parent.name
        if _CODE_DIR.match(parent_code):
            return Path(parent_code) / name
        return rel.with_suffix(".json")
    except ValueError:
        parent_code = pdf.parent.name
        if _CODE_DIR.match(parent_code):
            return Path(parent_code) / name
        return Path(name)


def segment_dir_for(
    pdf: Path,
    source_root: Path,
    segments_root: Path,
    *,
    stock_code: str = "",
) -> Path:
    rel = output_rel_for(pdf, source_root, stock_code)
    return segments_root / rel.with_suffix("")


def write_segment_file(
    dest_dir: Path,
    pair: QAPair,
    info: ActivityInfo,
    meta: dict,
) -> Path:
    name = f"{pair.index:03d}_{safe_filename(pair.question)}.md"
    path = dest_dir / name
    front = {
        "title": info.title or pair_title(pair),
        "kind": pair.kind,
        "index": pair.index,
        "stock_code": meta.get("stock_code", ""),
        "stock_name": meta.get("stock_name", ""),
        "year": meta.get("year", 0) or "",
        "published_on": meta.get("published_on", ""),
        "activity_type": info.activity_type,
        "scope": meta.get("scope", "investor_materials"),
        "doc_type": meta.get("doc_type", "ir_activities"),
    }
    front = {key: value for key, value in front.items() if value not in ("", None, 0)}
    dumped = yaml.safe_dump(front, allow_unicode=True, sort_keys=False).rstrip()
    body = "\n".join(
        [
            "---",
            dumped,
            "---",
            "",
            f"# {pair.question}",
            "",
            (pair.answer or "").strip(),
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return path


def split_pdf(
    pdf: Path,
    source_root: Path,
    output_root: Path,
    *,
    dump_segments: bool = False,
    segments_root: Path | None = None,
    use_llm: bool = False,
    agent: SimpleAgent | None = None,
) -> Path | None:
    text = extract_pdf_text(pdf)
    if not text.strip():
        log.warning("跳过（未抽出文字）: %s", pdf)
        return None

    info = parse_activity_info(pdf, source_root, text)
    meta = meta_from_info(info)

    if use_llm:
        if agent is None:
            raise ValueError("use_llm 需要传入 SimpleAgent")
        pairs, strategy = split_pairs_with_llm(text, agent)
        body_hint = "llm"
    else:
        body, body_hint = locate_body_from_pdf(pdf)
        if not (body or text).strip():
            log.warning("跳过（未抽出文字）: %s", pdf)
            return None
        # 定位不到正文位置：整份文本直接滚动切割，不再套问答/章节规则
        if body_hint == "unlocated":
            source = (body or text).strip()
            pairs = [QAPair(question="正文", answer=source, index=1, kind="window")]
            strategy = "window"
        else:
            pairs, strategy = split_pairs(body)

    if not pairs:
        log.warning("跳过（未切出内容）: %s", pdf)
        return None

    segment_dir: Path | None = None
    dumped = 0
    if dump_segments:
        if segments_root is None:
            raise ValueError("dump_segments 需要 segments_root")
        segment_dir = segment_dir_for(
            pdf, source_root, segments_root, stock_code=info.stock_code
        )
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            write_segment_file(segment_dir, pair, info, meta)
            dumped += 1

    chunks, windowed = build_chunks(pairs, info)
    if not chunks:
        log.warning("跳过（chunk 为空）: %s", pdf)
        return None

    dest = output_root / output_rel_for(pdf, source_root, info.stock_code)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": SPLIT_ID, "meta": meta, "chunks": chunks}
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    note = ""
    if segment_dir is not None:
        note = f"  中间段 {dumped} → {segment_dir}"
    log.info(
        "%s  body=%s  strategy=%s  qa=%s  windowed=%s  chunk=%s  → %s%s",
        pdf.name,
        body_hint,
        strategy,
        len(pairs),
        windowed,
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
    agent: SimpleAgent | None = None
    if args.llm:
        agent = SimpleAgent(LLM_SPLIT_SYSTEM_PROMPT, temperature=0.1, retries=1)
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
                use_llm=args.llm,
                agent=agent,
            )
        except Exception as exc:
            log.error("%s: %s", pdf.name, exc)
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
