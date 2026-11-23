"""投资者问答抓取的共用部分：股票代码、文本清洗、Markdown / JSONL 写出。"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def add_crawl_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path,
    output_help: str,
    page_size: int,
    page_size_help: str,
    max_pages_help: str,
    keyword_help: str,
    start_help: str,
    end_help: str,
) -> None:
    parser.add_argument("--code", action="append", default=[], help="6 位股票代码，可重复，也可用逗号分隔")
    parser.add_argument("--codes-file", help="代码列表，一行一个，空行和 # 开头的行会跳过")
    parser.add_argument("--start", default="", help=start_help)
    parser.add_argument("--end", default="", help=end_help)
    parser.add_argument("--keyword", default="", help=keyword_help)
    parser.add_argument("--page-size", type=int, default=page_size, help=page_size_help)
    parser.add_argument("--max-pages", type=int, default=0, help=max_pages_help)
    parser.add_argument("--sleep", type=float, default=0.5, help="两次请求之间的间隔秒数")
    parser.add_argument("--timeout", type=float, default=30, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败后的重试次数")
    parser.add_argument("--answered-only", action="store_true", help="只保留已经有公司回复的问答")
    parser.add_argument("--output", default=str(default_output), help=output_help)
    parser.add_argument("--format", choices=("md", "jsonl", "both"), default="md", help="输出格式，默认 Markdown")


def check_crawl_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
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


def run_codes(codes: list[str], crawl_one: Callable[[str], None], pause: float, logger: logging.Logger) -> None:
    failed: list[str] = []
    for index, code in enumerate(codes):
        try:
            crawl_one(code)
        except Exception as exc:
            failed.append(code)
            logger.error("%s 失败: %s", code, exc)
        if index + 1 < len(codes):
            time.sleep(pause)
    if failed:
        raise SystemExit(f"未完成: {', '.join(failed)}")
