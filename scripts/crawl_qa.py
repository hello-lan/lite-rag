"""爬取深交所互动易 / 上证 e 互动上的上市公司问答，写成可入库的 Markdown。

深市走互动易，沪市走上证 e 互动；默认按股票代码自动分流。

互动易：

- 代码换组织编号：POST https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo
- 问答列表：POST https://irm.cninfo.com.cn/newircs/company/question

上证 e 互动：

- 代码换用户编号：GET https://sns.sseinfo.com/company.do?stockcode=600519
- 问答列表：GET https://sns.sseinfo.com/ajax/userfeeds.do
  type=11 最新答复，type=10 最新提问（补未回复）

示例：

    python scripts/crawl_qa.py --code 000001
    python scripts/crawl_qa.py --code 600519
    python scripts/crawl_qa.py --code 000001,600519 --answered-only
    python scripts/crawl_qa.py --codes-file codes.txt --start 2026-01-01
    python scripts/crawl_qa.py --code 600519 --source sse --output corpus/02_processed/financial_report/ir_qa
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("crawl_qa")

ROOT = Path(__file__).resolve().parent.parent
SHANGHAI = timezone(timedelta(hours=8))
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_TAG = re.compile(r"<[^>]+>")
_BLANK_LINES = re.compile(r"\n{3,}")
_HEADING = re.compile(r"^#{1,6}(?=\s|$)")
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_CODE = re.compile(r"^\d{6}$")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")

# 互动易
IRM_KEYBOARD_URL = "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo"
IRM_QUESTION_URL = "https://irm.cninfo.com.cn/newircs/company/question"
IRM_COMPANY_URL = "https://irm.cninfo.com.cn/ircs/company/companyDetail"
IRM_CLIENTS = {"2": "APP", "4": "网站", "5": "公众号"}
IRM_DEFAULT_PAGE_SIZE = 100

# 上证 e 互动
SSE_COMPANY_URL = "https://sns.sseinfo.com/company.do"
SSE_FEED_URL = "https://sns.sseinfo.com/ajax/userfeeds.do"
SSE_DEFAULT_PAGE_SIZE = 10
_SSE_ITEM = re.compile(r'<div[^>]*\bid="item-(\d+)"[^>]*>', re.I)
_SSE_FEED_TEXT = re.compile(r'class="m_feed_txt"[^>]*>(.*?)</div>', re.S)
_SSE_FEED_TIME = re.compile(r'class="m_feed_from"[^>]*>\s*<span>([^<]*)</span>', re.S)
_SSE_FEED_CLIENT = re.compile(r'class="m_feed_from"[^>]*>.*?href="javascript:;">([^<]+)</a>', re.S)
_SSE_COMPANY_PREFIX = re.compile(r"^[:：][^()（）]{1,40}\(\d{6}\)")
_SSE_ABSOLUTE_TIME = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}:\d{2})")


# ---------------------------------------------------------------------------
# 共用：参数、清洗、写出
# ---------------------------------------------------------------------------


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="爬取互动易 / 上证 e 互动上市公司问答")
    parser.add_argument("--code", action="append", default=[], help="6 位股票代码，可重复，也可用逗号分隔")
    parser.add_argument("--codes-file", help="代码列表，一行一个，空行和 # 开头的行会跳过")
    parser.add_argument(
        "--source",
        choices=("auto", "irm", "sse"),
        default="auto",
        help="数据源：auto 按代码分流（0/3→互动易，6→上证e互动），默认 auto",
    )
    parser.add_argument("--start", default="", help="提问起始日期，YYYY-MM-DD")
    parser.add_argument("--end", default="", help="提问结束日期，YYYY-MM-DD")
    parser.add_argument("--keyword", default="", help="关键词过滤（互动易传给接口；上证e互动本地过滤）")
    parser.add_argument(
        "--page-size",
        type=int,
        default=None,
        help=f"每页条数，默认互动易 {IRM_DEFAULT_PAGE_SIZE}、上证e互动 {SSE_DEFAULT_PAGE_SIZE}",
    )
    parser.add_argument("--max-pages", type=int, default=0, help="最多翻多少页，0 表示翻完")
    parser.add_argument("--sleep", type=float, default=0.5, help="两次请求之间的间隔秒数")
    parser.add_argument("--timeout", type=float, default=30, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败后的重试次数")
    parser.add_argument("--answered-only", action="store_true", help="只保留已经有公司回复的问答")
    parser.add_argument(
        "--output",
        default=None,
        help="输出目录；默认按来源写到 corpus/raw/irm 或 corpus/raw/sse",
    )
    parser.add_argument("--format", choices=("md", "jsonl", "both"), default="md", help="输出格式，默认 Markdown")
    args = parser.parse_args()
    if args.page_size is not None and args.page_size < 1:
        parser.error("--page-size 必须大于 0")
    if args.max_pages < 0:
        parser.error("--max-pages 不能为负")
    if args.sleep < 0:
        parser.error("--sleep 不能为负")
    for label, value in (("start", args.start), ("end", args.end)):
        if value and not _valid_date(value):
            parser.error(f"--{label} 需要 YYYY-MM-DD，收到 {value}")
    return args


def load_codes(args: argparse.Namespace) -> list[str]:
    found: list[str] = []
    for raw in args.code:
        found.extend(part.strip() for part in raw.split(","))
    if args.codes_file:
        path = Path(args.codes_file)
        if not path.exists():
            raise SystemExit(f"找不到代码列表: {path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            text = line.split("#", 1)[0].strip()
            if text:
                found.append(text)
    codes: list[str] = []
    seen: set[str] = set()
    for raw in found:
        code = raw.strip().upper()
        for suffix in (".SZ", ".SH", ".BJ"):
            if code.endswith(suffix):
                code = code[: -len(suffix)]
        if not _CODE.fullmatch(code):
            raise SystemExit(f"股票代码需要是 6 位数字: {raw}")
        if code not in seen:
            seen.add(code)
            codes.append(code)
    if not codes:
        raise SystemExit("请用 --code 或 --codes-file 指定至少一只股票")
    return codes


def detect_source(code: str, preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if code.startswith(("0", "3")):
        return "irm"
    if code.startswith("6"):
        return "sse"
    raise RuntimeError(f"{code} 无法自动判断来源，请用 --source irm|sse")


def default_output(source: str) -> Path:
    return ROOT / "corpus" / "raw" / source


def page_size_for(args: argparse.Namespace, source: str) -> int:
    if args.page_size is not None:
        return args.page_size
    return IRM_DEFAULT_PAGE_SIZE if source == "irm" else SSE_DEFAULT_PAGE_SIZE


def clean_text(value, *, breaks: bool = False) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\xa0", " ")
    if breaks:
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TAG.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = _HEADING.sub("＃", line)
        if line:
            lines.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def now_text() -> str:
    return datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M")


def safe_filename(code: str, stock_name: str) -> str:
    name = _INVALID_FILENAME.sub("_", stock_name).strip().strip(".")
    return f"{code}_{name or '公司'}"


def write_outputs(
    records: list[dict],
    output_dir: Path,
    output_format: str,
    render_markdown: Callable[[list[dict]], str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(records[0]["stock_code"], records[0]["stock_name"])
    written: list[Path] = []
    if output_format in {"md", "both"}:
        path = output_dir / f"{stem}.md"
        path.write_text(render_markdown(records), encoding="utf-8")
        written.append(path)
    if output_format in {"jsonl", "both"}:
        path = output_dir / f"{stem}.jsonl"
        body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def run_codes(codes: list[str], crawl_one: Callable[[str], None], pause: float) -> None:
    failed: list[str] = []
    for index, code in enumerate(codes):
        try:
            crawl_one(code)
        except Exception as exc:
            failed.append(code)
            log.error("%s 失败: %s", code, exc)
        if index + 1 < len(codes):
            time.sleep(pause)
    if failed:
        raise SystemExit(f"未完成: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# 深交所互动易
# ---------------------------------------------------------------------------


def irm_post_json(
    url: str, *, params: dict | None, data: dict | None, timeout: float, retries: int, pause: float
) -> dict:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    body = None
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://irm.cninfo.com.cn/",
        "Origin": "https://irm.cninfo.com.cn",
    }
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("接口返回的不是 JSON 对象")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait = pause * attempt
            log.info("请求失败，%.1f 秒后重试（%s/%s）：%s", wait, attempt, retries, exc)
            time.sleep(wait)
    raise RuntimeError(f"请求失败: {url.split('?', 1)[0]} ({last_error})")


def irm_resolve_company(code: str, timeout: float, retries: int, pause: float) -> dict:
    payload = irm_post_json(
        IRM_KEYBOARD_URL,
        params=None,
        data={"keyWord": code},
        timeout=timeout,
        retries=retries,
        pause=pause,
    )
    matches = payload.get("data") or []
    for item in matches:
        if str(item.get("stockCode") or "") == code and item.get("secid"):
            return item
    message = payload.get("message") or "互动易没有这只股票"
    raise RuntimeError(f"{code} {message}")


def irm_fetch_questions(company: dict, args: argparse.Namespace, page_size: int) -> list[dict]:
    code = str(company["stockCode"])
    org_id = str(company["secid"])
    name = str(company.get("shortName") or code)
    page = 1
    total_pages = 1
    rows: list[dict] = []
    seen: set[str] = set()
    while page <= total_pages:
        payload = irm_post_json(
            IRM_QUESTION_URL,
            params={
                "stockcode": code,
                "orgId": org_id,
                "pageSize": str(page_size),
                "pageNum": str(page),
                "keyWord": args.keyword,
                "startDay": args.start,
                "endDay": args.end,
            },
            data=None,
            timeout=args.timeout,
            retries=args.retries,
            pause=args.sleep,
        )
        if "rows" not in payload:
            message = payload.get("message") or "问答接口没有返回 rows"
            raise RuntimeError(f"{code} {message}")
        batch = payload.get("rows") or []
        total_pages = max(int(payload.get("totalPage") or 1), 1)
        fresh = 0
        for row in batch:
            index_id = str(row.get("indexId") or "")
            if index_id and index_id in seen:
                continue
            if index_id:
                seen.add(index_id)
            rows.append(row)
            fresh += 1
        log.info("%s %s 第 %s/%s 页，新增 %s 条", code, name, page, total_pages, fresh)
        if not batch or fresh == 0:
            break
        if args.max_pages and page >= args.max_pages:
            break
        page += 1
        if page <= total_pages:
            time.sleep(args.sleep)
    return rows


def irm_format_time(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    seconds = value / 1000 if value > 10_000_000_000 else value
    moment = datetime.fromtimestamp(seconds, SHANGHAI)
    return moment.strftime("%Y-%m-%d %H:%M")


def irm_to_record(company: dict, row: dict) -> dict:
    code = str(row.get("stockCode") or company.get("stockCode") or "")
    name = str(row.get("companyShortName") or company.get("shortName") or "")
    org_id = str(company.get("secid") or "")
    trade = row.get("trade") or []
    if isinstance(trade, list):
        trade_name = "、".join(str(item) for item in trade if item)
    else:
        trade_name = str(trade or "")
    question = clean_text(row.get("mainContent"))
    answer = clean_text(row.get("attachedContent"))
    answer_time = ""
    if answer:
        answer_time = irm_format_time(row.get("attachedPubDate")) or irm_format_time(row.get("updateDate"))
    return {
        "index_id": str(row.get("indexId") or ""),
        "stock_code": code,
        "stock_name": name,
        "org_id": org_id,
        "trade": trade_name,
        "question": question,
        "answer": answer,
        "questioner": clean_text(row.get("authorName")),
        "answerer": clean_text(row.get("attachedAuthor")),
        "question_time": irm_format_time(row.get("pubDate")),
        "answer_time": answer_time,
        "client": IRM_CLIENTS.get(str(row.get("pubClient") or ""), ""),
        "url": f"{IRM_COMPANY_URL}?stockcode={code}&orgId={urllib.parse.quote(org_id)}",
    }


def irm_keep_record(record: dict, answered_only: bool) -> bool:
    if not record["question"]:
        return False
    if answered_only and not record["answer"]:
        return False
    return True


def irm_render_markdown(records: list[dict], fetched_at: str) -> str:
    first = records[0]
    lines = [
        f"# {first['stock_name']}（{first['stock_code']}）投资者互动问答",
        "",
        f"来源：深交所互动易 {first['url']}",
        f"抓取时间：{fetched_at}",
        f"条数：{len(records)}",
        "",
    ]
    for record in records:
        when = record["question_time"] or "时间未知"
        lines.append(f"{record['stock_name']}（{record['stock_code']}）{when}")
        lines.append("")
        lines.append(f"问：{record['question']}")
        lines.append("")
        if record["answer"]:
            who = f"，{record['answerer']}" if record["answerer"] else ""
            replied = record["answer_time"] or "时间未知"
            lines.append(f"答：（{replied}{who}）{record['answer']}")
        else:
            lines.append("答：尚未回复。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def crawl_irm(code: str, args: argparse.Namespace, fetched_at: str, output_dir: Path) -> None:
    company = irm_resolve_company(code, args.timeout, args.retries, args.sleep)
    time.sleep(args.sleep)
    page_size = page_size_for(args, "irm")
    rows = irm_fetch_questions(company, args, page_size)
    records = [irm_to_record(company, row) for row in rows]
    records = [record for record in records if irm_keep_record(record, args.answered_only)]
    records.sort(key=lambda record: record["question_time"], reverse=True)
    name = str(company.get("shortName") or code)
    if not records:
        log.info("%s %s 没有符合条件的问答", code, name)
        return
    paths = write_outputs(
        records,
        output_dir,
        args.format,
        lambda rows: irm_render_markdown(rows, fetched_at),
    )
    for path in paths:
        log.info("写入 %s（%s 条）", path, len(records))


# ---------------------------------------------------------------------------
# 上证 e 互动
# ---------------------------------------------------------------------------


def sse_fetch_text(url: str, *, referer: str, timeout: float, retries: int, pause: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,*/*",
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, UnicodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait = pause * attempt
            log.info("请求失败，%.1f 秒后重试（%s/%s）：%s", wait, attempt, retries, exc)
            time.sleep(wait)
    raise RuntimeError(f"请求失败: {url.split('?', 1)[0]} ({last_error})")


def sse_resolve_company(code: str, timeout: float, retries: int, pause: float) -> dict:
    page = sse_fetch_text(
        f"{SSE_COMPANY_URL}?stockcode={code}",
        referer="https://sns.sseinfo.com/",
        timeout=timeout,
        retries=retries,
        pause=pause,
    )
    uid_match = re.search(r"uid=(\d+)&pageSize=\d+&typeCode=company", page)
    name_match = re.search(r'<input\b[^>]*\bid="company_txt"[^>]*>', page)
    value = ""
    if name_match:
        value_match = re.search(r'value="([^"]*)"', name_match.group(0))
        value = html.unescape(value_match.group(1)) if value_match else ""
    short_name = value
    named = re.match(r"(.+)\(\d{6}\)$", value.strip())
    if named:
        short_name = named.group(1).strip()
    if not uid_match or not short_name:
        raise RuntimeError(f"{code} 上证e互动没有这只股票")
    return {"stock_code": code, "stock_name": short_name, "uid": uid_match.group(1)}


def sse_normalize_time(value: str) -> str:
    text = clean_text(value)
    match = _SSE_ABSOLUTE_TIME.search(text)
    if not match:
        return text
    year, month, day, clock = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d} {clock}"


def sse_strip_company_prefix(text: str) -> str:
    return _SSE_COMPANY_PREFIX.sub("", text).strip()


def sse_person_name(fragment: str, marker: str) -> str:
    match = re.search(rf"{re.escape(marker)}.*?</a>\s*<p>([^<]*)</p>", fragment, re.S)
    return clean_text(match.group(1)) if match else ""


def sse_fragment_field(fragment: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(fragment)
    return match.group(1).strip() if match else ""


def sse_parse_feed(page: str, company: dict) -> list[dict]:
    records = []
    matches = list(_SSE_ITEM.finditer(page))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page)
        chunk = page[match.end() : end]
        answer_at = chunk.find('class="m_feed_detail m_qa"')
        if answer_at < 0:
            question_html, answer_html = chunk, ""
        else:
            question_html, answer_html = chunk[:answer_at], chunk[answer_at:]
        question = sse_strip_company_prefix(clean_text(sse_fragment_field(question_html, _SSE_FEED_TEXT), breaks=True))
        answer = clean_text(sse_fragment_field(answer_html, _SSE_FEED_TEXT), breaks=True) if answer_html else ""
        records.append(
            {
                "index_id": match.group(1),
                "stock_code": company["stock_code"],
                "stock_name": company["stock_name"],
                "uid": company["uid"],
                "question": question,
                "answer": answer,
                "questioner": sse_person_name(question_html, 'rel="face"'),
                "answerer": sse_person_name(answer_html, 'class="ansface"'),
                "question_time": sse_normalize_time(sse_fragment_field(question_html, _SSE_FEED_TIME)),
                "answer_time": sse_normalize_time(sse_fragment_field(answer_html, _SSE_FEED_TIME)) if answer else "",
                "client": clean_text(sse_fragment_field(question_html, _SSE_FEED_CLIENT)),
                "url": f"{SSE_COMPANY_URL}?stockcode={company['stock_code']}",
            }
        )
    return records


def sse_fetch_feed(
    company: dict, feed_type: str, label: str, args: argparse.Namespace, page_size: int
) -> list[dict]:
    referer = f"{SSE_COMPANY_URL}?stockcode={company['stock_code']}"
    page_no = 1
    rows: list[dict] = []
    seen: set[str] = set()
    while True:
        query = urllib.parse.urlencode(
            {
                "typeCode": "company",
                "type": feed_type,
                "pageSize": str(page_size),
                "uid": company["uid"],
                "page": str(page_no),
            }
        )
        page = sse_fetch_text(
            f"{SSE_FEED_URL}?{query}",
            referer=referer,
            timeout=args.timeout,
            retries=args.retries,
            pause=args.sleep,
        )
        fresh = []
        for record in sse_parse_feed(page, company):
            if record["index_id"] in seen:
                continue
            seen.add(record["index_id"])
            fresh.append(record)
        rows.extend(fresh)
        log.info(
            "%s %s %s 第 %s 页，新增 %s 条",
            company["stock_code"],
            company["stock_name"],
            label,
            page_no,
            len(fresh),
        )
        if not fresh:
            break
        if args.max_pages and page_no >= args.max_pages:
            break
        page_no += 1
        time.sleep(args.sleep)
    return rows


def sse_in_range(question_time: str, start: str, end: str) -> bool:
    if not start and not end:
        return True
    day = question_time[:10]
    if not _DAY.fullmatch(day):
        return False
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def sse_keep_record(record: dict, args: argparse.Namespace) -> bool:
    if not record["question"]:
        return False
    if args.answered_only and not record["answer"]:
        return False
    if args.keyword and args.keyword not in record["question"] and args.keyword not in record["answer"]:
        return False
    return sse_in_range(record["question_time"], args.start, args.end)


def sse_merge_records(answered: list[dict], questions: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for record in answered + questions:
        current = merged.get(record["index_id"])
        if current is None:
            merged[record["index_id"]] = record
            continue
        for key, value in record.items():
            if value and not current.get(key):
                current[key] = value
    return list(merged.values())


def sse_display_time(value: str) -> str:
    if not value:
        return "时间未知"
    if _DAY.match(value):
        return value
    return f"截至抓取时 {value}"


def sse_render_markdown(records: list[dict], fetched_at: str) -> str:
    first = records[0]
    lines = [
        f"# {first['stock_name']}（{first['stock_code']}）投资者互动问答",
        "",
        f"来源：上证e互动 {first['url']}",
        "说明：公司页目前只展示近一个月的提问和答复。",
        f"抓取时间：{fetched_at}",
        f"条数：{len(records)}",
        "",
    ]
    for record in records:
        lines.append(f"{record['stock_name']}（{record['stock_code']}）{sse_display_time(record['question_time'])}")
        lines.append("")
        lines.append(f"问：{record['question']}")
        lines.append("")
        if record["answer"]:
            who = f"，{record['answerer']}" if record["answerer"] else ""
            lines.append(f"答：（{sse_display_time(record['answer_time'])}{who}）{record['answer']}")
        else:
            lines.append("答：尚未回复。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def crawl_sse(code: str, args: argparse.Namespace, fetched_at: str, output_dir: Path) -> None:
    company = sse_resolve_company(code, args.timeout, args.retries, args.sleep)
    time.sleep(args.sleep)
    page_size = page_size_for(args, "sse")
    answered = sse_fetch_feed(company, "11", "最新答复", args, page_size)
    time.sleep(args.sleep)
    questions = sse_fetch_feed(company, "10", "最新提问", args, page_size)
    records = [record for record in sse_merge_records(answered, questions) if sse_keep_record(record, args)]
    records.sort(key=lambda record: record["question_time"], reverse=True)
    if not records:
        log.info("%s %s 没有符合条件的问答", code, company["stock_name"])
        return
    paths = write_outputs(
        records,
        output_dir,
        args.format,
        lambda rows: sse_render_markdown(rows, fetched_at),
    )
    for path in paths:
        log.info("写入 %s（%s 条）", path, len(records))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def crawl_one(code: str, args: argparse.Namespace, fetched_at: str) -> None:
    source = detect_source(code, args.source)
    output_dir = Path(args.output) if args.output else default_output(source)
    log.info("%s → %s", code, "互动易" if source == "irm" else "上证e互动")
    if source == "irm":
        crawl_irm(code, args, fetched_at, output_dir)
    else:
        crawl_sse(code, args, fetched_at, output_dir)


def main() -> None:
    configure_stdio()
    args = parse_args()
    fetched_at = now_text()
    run_codes(load_codes(args), lambda code: crawl_one(code, args, fetched_at), args.sleep)


if __name__ == "__main__":
    main()
