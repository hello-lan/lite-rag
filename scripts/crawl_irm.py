"""爬取深交所互动易上的上市公司问答，写成可入库的 Markdown。

互动易页面自己用的接口是：

- 代码换组织编号：POST https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo
- 问答列表：POST https://irm.cninfo.com.cn/newircs/company/question

问答列表的参数放在查询字符串里。`rows` 里的 `mainContent` 是提问，
`attachedContent` 是公司回复。接口返回的是互动易页面上能翻到的问答，
站点不一定保留该公司自开通以来的全部历史。

沪市公司可以解析出组织编号，但互动易上通常没有问答。上证 e 互动用
scripts/crawl_sse.py。

示例：

    python scripts/crawl_irm.py --code 000001
    python scripts/crawl_irm.py --code 000001,300750 --start 2026-01-01
    python scripts/crawl_irm.py --codes-file codes.txt --answered-only
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

log = logging.getLogger("crawl_irm")

ROOT = Path(__file__).resolve().parent.parent
KEYBOARD_URL = "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo"
QUESTION_URL = "https://irm.cninfo.com.cn/newircs/company/question"
COMPANY_URL = "https://irm.cninfo.com.cn/ircs/company/companyDetail"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
SHANGHAI = timezone(timedelta(hours=8))
CLIENTS = {"2": "APP", "4": "网站", "5": "公众号"}
_TAG = re.compile(r"<[^>]+>")
_BLANK_LINES = re.compile(r"\n{3,}")
_HEADING = re.compile(r"^#{1,6}(?=\s|$)")
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_CODE = re.compile(r"^\d{6}$")


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="爬取深交所互动易上的上市公司问答")
    parser.add_argument("--code", action="append", default=[], help="6 位股票代码，可重复，也可用逗号分隔")
    parser.add_argument("--codes-file", help="代码列表，一行一个，空行和 # 开头的行会跳过")
    parser.add_argument("--start", default="", help="提问起始日期，YYYY-MM-DD")
    parser.add_argument("--end", default="", help="提问结束日期，YYYY-MM-DD")
    parser.add_argument("--keyword", default="", help="传给互动易的关键词，只拉取命中的问答")
    parser.add_argument("--page-size", type=int, default=100, help="每页条数，默认 100")
    parser.add_argument("--max-pages", type=int, default=0, help="每家公司最多翻多少页，0 表示翻完")
    parser.add_argument("--sleep", type=float, default=0.5, help="两次请求之间的间隔秒数")
    parser.add_argument("--timeout", type=float, default=30, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败后的重试次数")
    parser.add_argument("--answered-only", action="store_true", help="只保留已经有公司回复的问答")
    parser.add_argument("--output", default=str(ROOT / "docs" / "irm"), help="输出目录，默认 docs/irm")
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


def post_json(url: str, *, params: dict | None, data: dict | None, timeout: float, retries: int, pause: float) -> dict:
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


def resolve_company(code: str, timeout: float, retries: int, pause: float) -> dict:
    payload = post_json(
        KEYBOARD_URL,
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


def fetch_questions(company: dict, args: argparse.Namespace) -> list[dict]:
    code = str(company["stockCode"])
    org_id = str(company["secid"])
    name = str(company.get("shortName") or code)
    page = 1
    total_pages = 1
    rows: list[dict] = []
    seen: set[str] = set()
    while page <= total_pages:
        payload = post_json(
            QUESTION_URL,
            params={
                "stockcode": code,
                "orgId": org_id,
                "pageSize": str(args.page_size),
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


def format_time(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    seconds = value / 1000 if value > 10_000_000_000 else value
    moment = datetime.fromtimestamp(seconds, SHANGHAI)
    return moment.strftime("%Y-%m-%d %H:%M")


def clean_text(value) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TAG.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = _HEADING.sub("＃", line)
        if line:
            lines.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def to_record(company: dict, row: dict) -> dict:
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
        answer_time = format_time(row.get("attachedPubDate")) or format_time(row.get("updateDate"))
    return {
        "index_id": str(row.get("indexId") or ""),
        "stock_code": code,
        "company": name,
        "org_id": org_id,
        "trade": trade_name,
        "question": question,
        "answer": answer,
        "questioner": clean_text(row.get("authorName")),
        "answerer": clean_text(row.get("attachedAuthor")),
        "question_time": format_time(row.get("pubDate")),
        "answer_time": answer_time,
        "client": CLIENTS.get(str(row.get("pubClient") or ""), ""),
        "url": f"{COMPANY_URL}?stockcode={code}&orgId={urllib.parse.quote(org_id)}",
    }


def keep_record(record: dict, answered_only: bool) -> bool:
    if not record["question"]:
        return False
    if answered_only and not record["answer"]:
        return False
    return True


def render_markdown(records: list[dict], fetched_at: str) -> str:
    first = records[0]
    lines = [
        f"# {first['company']}（{first['stock_code']}）投资者互动问答",
        "",
        f"来源：深交所互动易 {first['url']}",
        f"抓取时间：{fetched_at}",
        f"条数：{len(records)}",
        "",
    ]
    for record in records:
        when = record["question_time"] or "时间未知"
        lines.append(f"{record['company']}（{record['stock_code']}）{when}")
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


def crawl_one(code: str, args: argparse.Namespace, fetched_at: str) -> bool:
    company = resolve_company(code, args.timeout, args.retries, args.sleep)
    time.sleep(args.sleep)
    rows = fetch_questions(company, args)
    records = [to_record(company, row) for row in rows]
    records = [record for record in records if keep_record(record, args.answered_only)]
    records.sort(key=lambda record: record["question_time"], reverse=True)
    name = str(company.get("shortName") or code)
    if not records:
        log.info("%s %s 没有符合条件的问答", code, name)
        return True
    paths = write_outputs(records, Path(args.output), args.format, fetched_at)
    for path in paths:
        log.info("写入 %s（%s 条）", path, len(records))
    return True


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
