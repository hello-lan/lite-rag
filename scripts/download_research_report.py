"""A股机构研报下载工具（基于 akshare / 东方财富）。

数据来源：东方财富网-数据中心-研究报告-个股研报
接口：ak.stock_research_report_em

示例：

    python scripts/download_research_report.py --stock 002027
    python scripts/download_research_report.py --stock 002027 --start 2024 --end 2025
    python scripts/download_research_report.py --stock 002027,000333 --org 国信证券 --max-results 20
    python scripts/download_research_report.py --stock 002027 --list-only
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RAW_ROOT = _PROJECT_ROOT / "corpus" / "01_raw"
_DEFAULT_SUBDIR = "research_reports"

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://data.eastmoney.com/report/stock.jshtml",
}

MAX_RETRIES = 3
DOWNLOAD_DELAY = 1.0


def default_save_dir() -> str:
    return str(_RAW_ROOT / _DEFAULT_SUBDIR)


def resolve_save_dir(save_dir: str | None) -> str:
    if not save_dir:
        return default_save_dir()
    path = Path(save_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


def parse_stock_codes(raw: str) -> List[str]:
    codes: List[str] = []
    for part in raw.split(","):
        code = part.strip()
        if not code:
            continue
        if not (code.isdigit() and len(code) == 6):
            raise ValueError(f"无效股票代码: {code!r}，请传入 6 位数字代码")
        codes.append(code)
    if not codes:
        raise ValueError("未提供有效股票代码")
    return codes


def _safe_filename(title: str, max_len: int = 60) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    cleaned = cleaned.replace(" ", "_").strip("._")
    return cleaned[:max_len] or "untitled"


def _normalize_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    m = re.search(r"(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return text[:10] if len(text) >= 10 else text


def _year_of(date_str: str) -> Optional[int]:
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


def fetch_research_reports(stock_code: str) -> List[Dict[str, Any]]:
    """通过 akshare 拉取个股机构研报列表。"""
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError(
            "缺少依赖 akshare，请先安装: uv pip install akshare 或 pip install akshare"
        ) from exc

    print(f"  [akshare] 拉取 {stock_code} 研报列表...")
    df = ak.stock_research_report_em(symbol=stock_code)
    if df is None or df.empty:
        return []

    items: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        publish_date = _normalize_date(row.get("日期"))
        items.append(
            {
                "stock_code": str(row.get("股票代码") or stock_code),
                "stock_name": str(row.get("股票简称") or ""),
                "title": str(row.get("报告名称") or "").strip(),
                "org": str(row.get("机构") or "").strip(),
                "rating": str(row.get("东财评级") or "").strip(),
                "industry": str(row.get("行业") or "").strip(),
                "publish_date": publish_date,
                "year": _year_of(publish_date),
                "pdf_url": str(row.get("报告PDF链接") or "").strip(),
            }
        )
    print(f"  [akshare] 共获取 {len(items)} 条研报")
    return items


def filter_reports(
    items: List[Dict[str, Any]],
    start_year: int | None = None,
    end_year: int | None = None,
    org: str | None = None,
    rating: str | None = None,
    keyword: str | None = None,
    max_results: int | None = None,
) -> List[Dict[str, Any]]:
    filtered = items

    if start_year is not None or end_year is not None:
        lo = start_year or 1900
        hi = end_year or 9999
        filtered = [
            x for x in filtered if x.get("year") and lo <= int(x["year"]) <= hi
        ]

    if org:
        org_key = org.strip().lower()
        filtered = [x for x in filtered if org_key in (x.get("org") or "").lower()]

    if rating:
        rating_key = rating.strip()
        filtered = [x for x in filtered if rating_key in (x.get("rating") or "")]

    if keyword:
        kw = keyword.strip().lower()
        filtered = [x for x in filtered if kw in (x.get("title") or "").lower()]

    # 按日期倒序（接口通常已是新→旧，这里再保证一次）
    filtered = sorted(
        filtered,
        key=lambda x: x.get("publish_date") or "",
        reverse=True,
    )

    if max_results is not None:
        filtered = filtered[:max_results]
    return filtered


def download_pdf(
    session: requests.Session,
    url: str,
    save_path: Path,
    delay: float,
    overwrite: bool = False,
) -> bool:
    if save_path.exists() and not overwrite:
        print(f"  [skip] 已存在: {save_path.name}")
        return True

    if not url:
        print(f"  [skip] 无 PDF 链接: {save_path.name}")
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [download] [{attempt}/{MAX_RETRIES}] {save_path.name}")
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            content = resp.content
            if b"%PDF" not in content[:100]:
                print(f"  [warn] 非有效 PDF（{resp.headers.get('Content-Type')}）")
                if attempt < MAX_RETRIES:
                    time.sleep(delay * attempt)
                    continue
                return False
            save_path.write_bytes(content)
            print(f"  [ok] {save_path} ({len(content) / 1024:.1f} KB)")
            return True
        except Exception as e:
            print(f"  [err] {str(e)[:100]}")
            if attempt < MAX_RETRIES:
                time.sleep(delay * attempt)

    print(f"  [fail] 最终失败: {save_path.name}")
    return False


def build_save_name(item: Dict[str, Any]) -> str:
    code = item.get("stock_code") or "unknown"
    org = _safe_filename(item.get("org") or "未知机构", max_len=20)
    title = _safe_filename(item.get("title") or "untitled")
    date_str = item.get("publish_date") or ""
    parts = [code, org, title]
    if date_str:
        parts.append(date_str)
    return "_".join(parts)


def download_stock_reports(
    stock_code: str,
    save_dir: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    org: str | None = None,
    rating: str | None = None,
    keyword: str | None = None,
    max_results: int | None = None,
    delay: float = DOWNLOAD_DELAY,
    list_only: bool = False,
    overwrite: bool = False,
    save_meta: bool = True,
) -> List[Dict[str, Any]]:
    root = Path(resolve_save_dir(save_dir))
    stock_dir = root / stock_code
    stock_dir.mkdir(parents=True, exist_ok=True)

    items = fetch_research_reports(stock_code)
    items = filter_reports(
        items,
        start_year=start_year,
        end_year=end_year,
        org=org,
        rating=rating,
        keyword=keyword,
        max_results=max_results,
    )
    print(f"  [filter] 筛选后剩余 {len(items)} 条")

    if list_only:
        for i, item in enumerate(items, 1):
            print(
                f"  {i:3d}. [{item.get('publish_date')}] "
                f"{item.get('org')} | {item.get('rating')} | {item.get('title')}"
            )
        return items

    session = requests.Session()
    session.headers.update(COMMON_HEADERS)

    downloaded: List[Dict[str, Any]] = []
    for item in items:
        save_name = build_save_name(item)
        save_path = stock_dir / f"{save_name}.pdf"
        ok = download_pdf(
            session=session,
            url=item.get("pdf_url") or "",
            save_path=save_path,
            delay=delay,
            overwrite=overwrite,
        )
        if ok:
            record = {
                **item,
                "file_path": str(save_path),
                "source": "eastmoney/akshare",
            }
            downloaded.append(record)
        time.sleep(delay)

    if save_meta and downloaded:
        meta_path = stock_dir / "index.json"
        import json

        existing: List[Dict[str, Any]] = []
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []

        by_url = {
            (r.get("pdf_url") or r.get("file_path")): r
            for r in existing
            if isinstance(r, dict)
        }
        for r in downloaded:
            by_url[r.get("pdf_url") or r.get("file_path")] = r
        merged = list(by_url.values())
        merged.sort(key=lambda x: x.get("publish_date") or "", reverse=True)
        meta_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [meta] 已更新索引: {meta_path} ({len(merged)} 条)")

    return downloaded


def batch_download(
    stock_codes: List[str],
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    delay = float(kwargs.get("delay") or DOWNLOAD_DELAY)

    for i, code in enumerate(stock_codes):
        print(f"\n{'=' * 50}")
        print(f"[{i + 1}/{len(stock_codes)}] 处理股票: {code}")
        print(f"{'=' * 50}")
        try:
            results[code] = download_stock_reports(stock_code=code, **kwargs)
        except Exception as e:
            print(f"  [FAIL] {code}: {e}")
            results[code] = []
        if i < len(stock_codes) - 1:
            time.sleep(delay)

    return results


def _ensure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main() -> None:
    import argparse

    _ensure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description="A股机构研报下载工具（akshare / 东方财富个股研报）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_research_report.py --stock 002027
  python scripts/download_research_report.py --stock 002027 --start 2024 --end 2025
  python scripts/download_research_report.py --stock 002027 --org 国信证券 --max-results 10
  python scripts/download_research_report.py --stock 002027 --list-only

默认目录:
  corpus/01_raw/research_reports/{代码}/
        """,
    )
    parser.add_argument(
        "--stock",
        required=True,
        help="股票代码，多个用逗号分隔，如 002027 或 002027,000333",
    )
    parser.add_argument("--start", type=int, help="起始年份（按发布日期）")
    parser.add_argument("--end", type=int, help="结束年份（按发布日期）")
    parser.add_argument("--org", help="按机构名称过滤（模糊匹配），如 国信证券")
    parser.add_argument("--rating", help="按东财评级过滤，如 买入 / 增持")
    parser.add_argument("--keyword", help="按报告标题关键词过滤")
    parser.add_argument("--max-results", type=int, default=None, help="每只股票最多下载数量")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="PDF 保存目录（默认 corpus/01_raw/research_reports）",
    )
    parser.add_argument("--delay", type=float, default=DOWNLOAD_DELAY, help="下载间隔秒数")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="仅列出研报，不下载 PDF",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 PDF",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="不写入 index.json 元数据",
    )

    args = parser.parse_args()

    try:
        codes = parse_stock_codes(args.stock)
    except ValueError as e:
        parser.error(str(e))

    current_year = datetime.now().year
    start_year = args.start if args.start is not None else current_year - 2
    end_year = args.end if args.end is not None else current_year
    save_dir = resolve_save_dir(args.save_dir)

    print("准备下载机构研报")
    print(f"   股票代码: {', '.join(codes)}")
    print(f"   年份范围: {start_year} ~ {end_year}")
    if args.org:
        print(f"   机构过滤: {args.org}")
    if args.rating:
        print(f"   评级过滤: {args.rating}")
    if args.keyword:
        print(f"   标题关键词: {args.keyword}")
    print(f"   保存目录: {save_dir}")
    print(f"   模式: {'仅列表' if args.list_only else '下载 PDF'}")
    print("   数据源: akshare.stock_research_report_em（东方财富）")

    results = batch_download(
        stock_codes=codes,
        save_dir=save_dir,
        start_year=start_year,
        end_year=end_year,
        org=args.org,
        rating=args.rating,
        keyword=args.keyword,
        max_results=args.max_results,
        delay=args.delay,
        list_only=args.list_only,
        overwrite=args.overwrite,
        save_meta=not args.no_meta,
    )

    total = sum(len(v) for v in results.values())
    action = "列出" if args.list_only else "下载"
    print(f"\n完成: {len(codes)} 只股票, 共 {action} {total} 条研报")

    for code, reports in results.items():
        if not reports:
            print(f"   {code}: 0 条")
            continue
        orgs = sorted({r.get("org") for r in reports if r.get("org")})
        years = sorted({r.get("year") for r in reports if r.get("year")})
        print(f"   {code}: {len(reports)} 条, 年份: {years}, 机构数: {len(orgs)}")
        if not args.list_only:
            for report in reports[:5]:
                fp = report.get("file_path")
                if fp:
                    print(f"      {fp}")
            if len(reports) > 5:
                print(f"      ... 另有 {len(reports) - 5} 个文件")

    if total == 0:
        print("\n[WARN] 未获取到任何研报，可能原因：")
        print("   1. 年份/机构/评级过滤过严")
        print("   2. 该股票暂无机构研报")
        print("   3. 网络问题或东财接口限流")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
