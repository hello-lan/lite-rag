"""爬取上证 e 互动上的上市公司问答，写成可入库的 Markdown。

公司页和问答列表是两步：

- 代码换用户编号：GET https://sns.sseinfo.com/company.do?stockcode=600519
- 问答列表：GET https://sns.sseinfo.com/ajax/userfeeds.do

type=11 是「最新答复」，问题和公司回复在同一条里。type=10 是「最新提问」，
用来补上还没有回复的问题。公司页目前只展示近一个月，翻完接口返回的页就是站点上能看到的全部。

深市公司不在这个站点上。深交所互动易用 scripts/crawl_irm.py。

示例：

    python scripts/crawl_sse.py --code 600519
    python scripts/crawl_sse.py --code 600519,601899 --answered-only
    python scripts/crawl_sse.py --codes-file codes.txt --start 2026-11-01
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("crawl_sse")

ROOT = Path(__file__).resolve().parent.parent
COMPANY_URL = "https://sns.sseinfo.com/company.do"
FEED_URL = "https://sns.sseinfo.com/ajax/userfeeds.do"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
SHANGHAI = timezone(timedelta(hours=8))
FEEDS = (("11", "最新答复"), ("10", "最新提问"))
_TAG = re.compile(r"<[^>]+>")
_BLANK_LINES = re.compile(r"\n{3,}")
_HEADING = re.compile(r"^#{1,6}(?=\s|$)")
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_CODE = re.compile(r"^\d{6}$")
_ITEM = re.compile(r'<div[^>]*\bid="item-(\d+)"[^>]*>', re.I)
_FEED_TEXT = re.compile(r'class="m_feed_txt"[^>]*>(.*?)</div>', re.S)
_FEED_TIME = re.compile(r'class="m_feed_from"[^>]*>\s*<span>([^<]*)</span>', re.S)
_FEED_CLIENT = re.compile(r'class="m_feed_from"[^>]*>.*?href="javascript:;">([^<]+)</a>', re.S)
_COMPANY_PREFIX = re.compile(r"^[:：][^()（）]{1,40}\(\d{6}\)")
_ABSOLUTE_TIME = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}:\d{2})")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="爬取上证 e 互动上的上市公司问答")
    parser.add_argument("--code", action="append", default=[], help="6 位股票代码，可重复，也可用逗号分隔")
    parser.add_argument("--codes-file", help="代码列表，一行一个，空行和 # 开头的行会跳过")
    parser.add_argument("--start", default="", help="只保留这个日期之后的提问，YYYY-MM-DD")
    parser.add_argument("--end", default="", help="只保留这个日期之前的提问，YYYY-MM-DD")
    parser.add_argument("--keyword", default="", help="只保留问题或答复里含这个词的记录")
    parser.add_argument("--page-size", type=int, default=10, help="每页条数，站点默认 10")
    parser.add_argument("--max-pages", type=int, default=0, help="每种列表最多翻多少页，0 表示翻完")
    parser.add_argument("--sleep", type=float, default=0.5, help="两次请求之间的间隔秒数")
    parser.add_argument("--timeout", type=float, default=30, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败后的重试次数")
    parser.add_argument("--answered-only", action="store_true", help="只保留已经有公司回复的问答")
    parser.add_argument("--output", default=str(ROOT / "docs" / "sse"), help="输出目录，默认 docs/sse")
    parser.add_argument("--format", choices=("md", "jsonl", "both"), default="md", help="输出格式，默认 Markdown")
    args = parser.parse_args()
    if args.page_size < 1:
        parser.error("--page-size 必须大于 0")
    if args.max_pages < 0:
        parser.error("--max-pages 不能为负")
    if args.sleep < 0:
        parser.error("--sleep 不能为负")
    for label, value in (("start", args.start), ("end", args.end)):
        if value and not _valid_date(value):
            parser.error(f"--{label} 需要 YYYY-MM-DD，收到 {value}")
    return args


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


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


def fetch_text(url: str, *, referer: str, timeout: float, retries: int, pause: float) -> str:
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


def resolve_company(code: str, timeout: float, retries: int, pause: float) -> dict:
    page = fetch_text(
        f"{COMPANY_URL}?stockcode={code}",
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
    return {"stock_code": code, "company": short_name, "uid": uid_match.group(1)}


def clean_text(value: str) -> str:
    text = html.unescape(value).replace("\xa0", " ")
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


def normalize_time(value: str) -> str:
    text = clean_text(value)
    match = _ABSOLUTE_TIME.search(text)
    if not match:
        return text
    year, month, day, clock = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d} {clock}"


def strip_company_prefix(text: str) -> str:
    return _COMPANY_PREFIX.sub("", text).strip()


def person_name(fragment: str, marker: str) -> str:
    match = re.search(rf"{re.escape(marker)}.*?</a>\s*<p>([^<]*)</p>", fragment, re.S)
    return clean_text(match.group(1)) if match else ""


def fragment_field(fragment: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(fragment)
    return match.group(1).strip() if match else ""


def parse_feed(page: str, company: dict) -> list[dict]:
    records = []
    matches = list(_ITEM.finditer(page))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page)
        chunk = page[match.end():end]
        answer_at = chunk.find('class="m_feed_detail m_qa"')
        if answer_at < 0:
            question_html, answer_html = chunk, ""
        else:
            question_html, answer_html = chunk[:answer_at], chunk[answer_at:]
        question = strip_company_prefix(clean_text(fragment_field(question_html, _FEED_TEXT)))
        answer = clean_text(fragment_field(answer_html, _FEED_TEXT)) if answer_html else ""
        records.append(
            {
                "index_id": match.group(1),
                "stock_code": company["stock_code"],
                "company": company["company"],
                "uid": company["uid"],
                "question": question,
                "answer": answer,
                "questioner": person_name(question_html, 'rel="face"'),
                "answerer": person_name(answer_html, 'class="ansface"'),
                "question_time": normalize_time(fragment_field(question_html, _FEED_TIME)),
                "answer_time": normalize_time(fragment_field(answer_html, _FEED_TIME)) if answer else "",
                "client": clean_text(fragment_field(question_html, _FEED_CLIENT)),
                "url": f"{COMPANY_URL}?stockcode={company['stock_code']}",
            }
        )
    return records


def fetch_feed(company: dict, feed_type: str, label: str, args: argparse.Namespace) -> list[dict]:
    referer = f"{COMPANY_URL}?stockcode={company['stock_code']}"
    page_no = 1
    rows: list[dict] = []
    seen: set[str] = set()
    while True:
        query = urllib.parse.urlencode(
            {
                "typeCode": "company",
                "type": feed_type,
                "pageSize": str(args.page_size),
                "uid": company["uid"],
                "page": str(page_no),
            }
        )
        page = fetch_text(
            f"{FEED_URL}?{query}",
            referer=referer,
            timeout=args.timeout,
            retries=args.retries,
            pause=args.sleep,
        )
        fresh = []
        for record in parse_feed(page, company):
            if record["index_id"] in seen:
                continue
            seen.add(record["index_id"])
            fresh.append(record)
        rows.extend(fresh)
        log.info(
            "%s %s %s 第 %s 页，新增 %s 条",
            company["stock_code"],
            company["company"],
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


def in_range(question_time: str, start: str, end: str) -> bool:
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


def keep_record(record: dict, args: argparse.Namespace) -> bool:
    if not record["question"]:
        return False
    if args.answered_only and not record["answer"]:
        return False
    if args.keyword and args.keyword not in record["question"] and args.keyword not in record["answer"]:
        return False
    return in_range(record["question_time"], args.start, args.end)


def merge_records(answered: list[dict], questions: list[dict]) -> list[dict]:
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


def display_time(value: str) -> str:
    if not value:
        return "时间未知"
    if _DAY.match(value):
        return value
    return f"截至抓取时 {value}"


def render_markdown(records: list[dict], fetched_at: str) -> str:
    first = records[0]
    lines = [
        f"# {first['company']}（{first['stock_code']}）投资者互动问答",
        "",
        f"来源：上证e互动 {first['url']}",
        "说明：公司页目前只展示近一个月的提问和答复。",
        f"抓取时间：{fetched_at}",
        f"条数：{len(records)}",
        "",
    ]
    for record in records:
        lines.append(f"{record['company']}（{record['stock_code']}）{display_time(record['question_time'])}")
        lines.append("")
        lines.append(f"问：{record['question']}")
        lines.append("")
        if record["answer"]:
            who = f"，{record['answerer']}" if record["answerer"] else ""
            lines.append(f"答：（{display_time(record['answer_time'])}{who}）{record['answer']}")
        else:
            lines.append("答：尚未回复。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def safe_filename(code: str, company: str) -> str:
    name = _INVALID_FILENAME.sub("_", company).strip().strip(".")
    return f"{code}_{name or '公司'}"


def write_outputs(records: list[dict], output_dir: Path, output_format: str, fetched_at: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(records[0]["stock_code"], records[0]["company"])
    written: list[Path] = []
    if output_format in {"md", "both"}:
        path = output_dir / f"{stem}.md"
        path.write_text(render_markdown(records, fetched_at), encoding="utf-8")
        written.append(path)
    if output_format in {"jsonl", "both"}:
        path = output_dir / f"{stem}.jsonl"
        body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        path.write_text(body, encoding="utf-8")
        written.append(path)
    return written


def crawl_one(code: str, args: argparse.Namespace, fetched_at: str) -> None:
    company = resolve_company(code, args.timeout, args.retries, args.sleep)
    time.sleep(args.sleep)
    answered = fetch_feed(company, "11", "最新答复", args)
    time.sleep(args.sleep)
    questions = fetch_feed(company, "10", "最新提问", args)
    records = [record for record in merge_records(answered, questions) if keep_record(record, args)]
    records.sort(key=lambda record: record["question_time"], reverse=True)
    if not records:
        log.info("%s %s 没有符合条件的问答", code, company["company"])
        return
    paths = write_outputs(records, Path(args.output), args.format, fetched_at)
    for path in paths:
        log.info("写入 %s（%s 条）", path, len(records))


def main() -> None:
    configure_stdio()
    args = parse_args()
    codes = load_codes(args)
    fetched_at = datetime.now(SHANGHAI).strftime("%Y-%m-%d %H:%M")
    failed: list[str] = []
    for index, code in enumerate(codes):
        try:
            crawl_one(code, args, fetched_at)
        except Exception as exc:
            failed.append(code)
            log.error("%s 失败: %s", code, exc)
        if index + 1 < len(codes):
            time.sleep(args.sleep)
    if failed:
        raise SystemExit(f"未完成: {', '.join(failed)}")


if __name__ == "__main__":
    main()
