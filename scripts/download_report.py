"""A股上市公司公告下载工具（多数据源，自动切换）。

按优先级尝试：巨潮资讯网 → 东方财富 → 新浪财经。
支持年报等定期报告，以及投资者关系活动记录。

示例：

    python scripts/download_report.py --stock 002027 --start 2020 --end 2024
    python scripts/download_report.py --stock 002027 --type ir --start 2023 --end 2025
"""

from __future__ import annotations

import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RAW_ROOT = _PROJECT_ROOT / "corpus" / "01_raw"

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

REPORT_PROFILES: Dict[str, Dict[str, Any]] = {
    "annual": {
        "categories": {
            "年报": {
                "cninfo": "category_ndbg_szsh",
                "keywords": ["年度报告", "年报"],
            },
            "半年报": {
                "cninfo": "category_sjdbg_szsh",
                "keywords": ["半年度报告", "半年报", "中期报告"],
            },
            "季报": {
                "cninfo": "category_yjdbg_szsh",
                "keywords": ["季度报告", "季报", "第一季度报告", "第三季度报告"],
            },
        },
        "default_category": "年报",
        "dedupe": "by_year",
        "date_mode": "report_year",
        "end_date_pad": True,
        "default_subdir": "annual_reports",
    },
    "ir": {
        "categories": {
            "投资者关系": {
                # 巨潮投关走 relation Tab，不是 fulltext
                "tabName": "relation",
                "cninfo": "category_tzzgx_szsh",
                "searchkey": "",
                "keywords": [
                    "投资者关系活动记录",
                    "投资者关系活动",
                    "投资者关系管理信息",
                ],
            },
        },
        "default_category": "投资者关系",
        "dedupe": "by_id",
        "date_mode": "announce_date",
        "end_date_pad": False,
        "default_subdir": "ir_activities",
    },
}

PAGE_SIZE = 30
MAX_RETRIES = 3
DOWNLOAD_DELAY = 1.0


def _random_sleep(base: float = DOWNLOAD_DELAY) -> None:
    """Sleep a random interval around `base` to reduce rate-limit risk.

    Actual wait is uniform in about [0.8x, 2.0x] of base (at least +0.5s span).
    """
    if base <= 0:
        return
    lo = max(0.3, base * 0.8)
    hi = max(lo + 0.5, base * 2.0)
    time.sleep(random.uniform(lo, hi))


def default_save_dir(report_type: str) -> str:
    subdir = REPORT_PROFILES[report_type]["default_subdir"]
    return str(_RAW_ROOT / subdir)


def resolve_save_dir(save_dir: str | None, report_type: str = "annual") -> str:
    if not save_dir:
        return default_save_dir(report_type)
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


def _extract_year(text: str) -> Optional[int]:
    """从文本提取四位年份；跳过股票代码前缀（如 002027… 中的 2027）。"""
    if not text:
        return None
    # 标题常以 6 位代码开头，先剥掉以免 002027 → 2027
    cleaned = re.sub(r"^\d{6}", "", text)
    m = re.search(r"(20\d{2}|19\d{2})", cleaned)
    return int(m.group(1)) if m else None


def _normalize_date(value: Any) -> str:
    """统一成 YYYY-MM-DD；支持毫秒时间戳 / 常见字符串。"""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        ts = value / 1000 if value > 1e12 else value
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return ""
    text = str(value).strip()
    if text.isdigit():
        return _normalize_date(int(text))
    m = re.search(r"(20\d{2}|19\d{2})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return text[:10] if len(text) >= 10 else text


def _date_window(
    start_year: int | None,
    end_year: int | None,
    end_date_pad: bool,
) -> Tuple[str, str]:
    start_date = f"{start_year}-01-01" if start_year else "1900-01-01"
    today = datetime.now().strftime("%Y-%m-%d")
    if end_year is None:
        return start_date, today
    if end_date_pad:
        end_date = min(f"{end_year + 1}-06-30", today)
    else:
        end_date = min(f"{end_year}-12-31", today)
    return start_date, end_date


def _safe_filename(title: str) -> str:
    return title.replace("/", "_").replace("\\", "_").replace(" ", "_")[:50]


# ==================== 公共下载管线 ====================


class BaseSpider:
    source_name = "base"
    headers: Dict[str, str] = COMMON_HEADERS
    request_delay = 0.5
    download_base: str | None = None

    def __init__(self, save_dir: str | None = None, delay: float | None = None):
        self.save_dir = Path(save_dir or default_save_dir("annual"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay if delay is not None else self.request_delay
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _log(self, message: str) -> None:
        print(f"  [{self.source_name}] {message}")

    def search_page(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        category_cfg: Dict[str, Any],
        page: int,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _resolve_url(self, url: str) -> str:
        if not url:
            return ""
        if url.startswith("http"):
            return url
        if self.download_base:
            return urljoin(self.download_base.rstrip("/") + "/", url.lstrip("/"))
        return url

    def _fetch_pdf_bytes(self, url: str) -> Optional[bytes]:
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get("Content-Type", "")

        if b"%PDF" not in content[:100] and "text/html" in content_type:
            pdf_match = re.search(
                rb'href=["\']([^"\']*\.pdf[^"\']*)["\']',
                content,
                re.IGNORECASE,
            )
            if pdf_match:
                pdf_url = pdf_match.group(1).decode("utf-8", errors="ignore")
                if not pdf_url.startswith("http"):
                    pdf_url = urljoin(url, pdf_url)
                resp = self.session.get(pdf_url, timeout=30)
                resp.raise_for_status()
                content = resp.content

        return content if b"%PDF" in content[:100] else None

    def download_pdf(
        self,
        stock_code: str,
        save_name: str,
        url: str,
        overwrite: bool = False,
    ) -> bool:
        if not url:
            self._log(f"跳过 {save_name}: 无下载链接")
            return False

        save_path = self.save_dir / stock_code / f"{save_name}.pdf"
        if save_path.exists() and not overwrite:
            self._log(f"跳过 {save_name}: 文件已存在")
            return True

        save_path.parent.mkdir(parents=True, exist_ok=True)
        download_url = self._resolve_url(url)

        for attempt in range(MAX_RETRIES):
            try:
                self._log(f"下载 [{attempt + 1}/{MAX_RETRIES}]: {save_name}")
                content = self._fetch_pdf_bytes(download_url)
                if content is None:
                    if attempt + 1 < MAX_RETRIES:
                        _random_sleep(self.delay * (attempt + 1))
                        continue
                    self._log(f"下载失败 {save_name}: 文件不是有效PDF")
                    return False
                save_path.write_bytes(content)
                self._log(f"下载完成: {save_path} ({len(content) / 1024:.2f} KB)")
                return True
            except Exception as e:
                self._log(f"下载重试中... {str(e)}")
                _random_sleep(self.delay * (attempt + 1))

        self._log(f"最终下载失败: {save_name}")
        return False

    def _filter_items(
        self,
        items: List[Dict[str, Any]],
        profile: Dict[str, Any],
        category_cfg: Dict[str, Any],
        start_year: int | None,
        end_year: int | None,
    ) -> List[Dict[str, Any]]:
        keywords = category_cfg.get("keywords") or []
        if keywords:
            items = [
                a
                for a in items
                if any(kw in (a.get("title") or "") for kw in keywords)
            ]

        end_y = end_year or 9999
        if start_year and profile.get("date_mode") == "report_year":
            items = [
                a
                for a in items
                if a.get("report_year") and start_year <= a["report_year"] <= end_y
            ]
        elif start_year and profile.get("date_mode") == "announce_date":
            filtered = []
            for a in items:
                d = a.get("announce_date") or ""
                y = _extract_year(d) or a.get("report_year")
                if y and start_year <= y <= end_y:
                    filtered.append(a)
            items = filtered

        if profile.get("dedupe") == "by_year":

            def _title_priority(title: str) -> int:
                if "摘要" in title:
                    return 2
                if "英文版" in title or "英文" in title or "English" in title:
                    return 1
                return 0

            items = sorted(items, key=lambda x: _title_priority(x.get("title") or ""))
            seen_years: set = set()
            deduped = []
            for a in items:
                y = a.get("report_year")
                if y and y not in seen_years:
                    seen_years.add(y)
                    deduped.append(a)
            return deduped

        seen: set = set()
        deduped = []
        for a in items:
            key = a.get("announcement_id") or (
                a.get("title"),
                a.get("announce_date"),
                a.get("url"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        return deduped

    def get_reports(
        self,
        stock_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
        report_type: str = "annual",
        category: str | None = None,
        max_results: int | None = None,
    ) -> List[Dict[str, Any]]:
        profile = REPORT_PROFILES[report_type]
        cat_name = category or profile["default_category"]
        category_cfg = profile["categories"][cat_name]
        start_date, end_date = _date_window(
            start_year, end_year, bool(profile.get("end_date_pad"))
        )

        all_items: List[Dict[str, Any]] = []
        page = 1
        while True:
            items = self.search_page(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                category_cfg=category_cfg,
                page=page,
            )
            if not items:
                break
            all_items.extend(items)
            if len(items) < PAGE_SIZE:
                break
            page += 1
            _random_sleep(self.delay)

        all_items = self._filter_items(
            all_items, profile, category_cfg, start_year, end_year
        )
        if max_results is not None:
            all_items = all_items[:max_results]

        downloaded: List[Dict[str, Any]] = []
        for item in all_items:
            title = _safe_filename(item.get("title") or "untitled")
            suffix = item.get("announce_date") or item.get("report_year") or ""
            year_info = f"_{suffix}" if suffix else ""
            save_name = f"{stock_code}_{title}{year_info}"

            ok = self.download_pdf(
                stock_code=stock_code,
                save_name=save_name,
                url=item.get("url") or "",
            )
            if ok:
                year = _extract_year(item.get("announce_date") or "") or item.get(
                    "report_year"
                )
                downloaded.append(
                    {
                        "stock_code": stock_code,
                        "title": item.get("title"),
                        "file_path": str(self.save_dir / stock_code / f"{save_name}.pdf"),
                        "year": year,
                        "announce_date": item.get("announce_date"),
                        "source": self.source_name,
                    }
                )
            # 每个文件下载后随机间隔，避免请求过密被封
            _random_sleep(max(self.delay, DOWNLOAD_DELAY))
        return downloaded


# ==================== 巨潮资讯网 ====================


class CninfoSpider(BaseSpider):
    source_name = "cninfo"
    request_delay = 0.5
    download_base = "http://static.cninfo.com.cn"
    headers = {
        **COMMON_HEADERS,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "http://www.cninfo.com.cn/new/index",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }
    SEARCH_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"

    def _get_stock_orgid(self, stock_code: str) -> Optional[Tuple[str, str]]:
        url = "http://www.cninfo.com.cn/new/data/szse_stock.json"
        try:
            resp = self.session.get(url, timeout=10)
            data = resp.json()
            for stock in data.get("stockList", []):
                if stock["code"] == stock_code:
                    self._log(f"找到股票: {stock_code} -> {stock['orgId']}")
                    return stock["orgId"], stock.get("fullshortname", stock_code)
        except Exception as e:
            self._log(f"获取orgId失败 {stock_code}: {str(e)}")
        return None

    def search_page(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        category_cfg: Dict[str, Any],
        page: int,
    ) -> List[Dict[str, Any]]:
        stock_info = self._get_stock_orgid(stock_code)
        if not stock_info:
            return []
        org_id, _ = stock_info

        if stock_code.startswith(("6", "5")):
            plate, column = "sh", "sse"
        elif stock_code.startswith(("0", "3", "2")):
            plate, column = "sz", "szse"
        else:
            plate, column = "bj", "bj"

        data = {
            "stock": f"{stock_code},{org_id}",
            "tabName": category_cfg.get("tabName", "fulltext"),
            "pageSize": str(PAGE_SIZE),
            "pageNum": str(page),
            "column": column,
            "category": category_cfg.get("cninfo", ""),
            "plate": plate,
            "seDate": f"{start_date}~{end_date}",
            "searchkey": category_cfg.get("searchkey", ""),
            "secid": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        try:
            self._log(f"搜索 {stock_code} 页码: {page}")
            resp = self.session.post(self.SEARCH_URL, data=data, timeout=15)
            resp.raise_for_status()
            announcements = resp.json().get("announcements", []) or []
            items = []
            for item in announcements:
                title = item.get("announcementTitle", "")
                items.append(
                    {
                        "title": title,
                        "url": item.get("adjunctUrl", ""),
                        "announce_date": _normalize_date(item.get("announcementTime")),
                        "report_year": _extract_year(title),
                        "announcement_id": item.get("announcementId", ""),
                    }
                )
            return items
        except Exception as e:
            self._log(f"搜索失败 {stock_code}: {str(e)}")
            return []


# ==================== 东方财富 ====================


class EastmoneySpider(BaseSpider):
    source_name = "eastmoney"
    request_delay = 1.0
    download_base = "https://pdf.dfcfw.com"
    headers = {
        **COMMON_HEADERS,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://data.eastmoney.com/",
    }
    SEARCH_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"

    @staticmethod
    def _stock_market_code(code: str) -> str:
        if code.startswith(("6", "5")):
            return f"SH{code}"
        if code.startswith(("0", "3", "2")):
            return f"SZ{code}"
        if code.startswith(("8", "4")):
            return f"BJ{code}"
        return f"SZ{code}"

    def search_page(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        category_cfg: Dict[str, Any],
        page: int,
    ) -> List[Dict[str, Any]]:
        del category_cfg  # 东财靠标题关键词在基类过滤
        params = {
            "page_size": PAGE_SIZE,
            "page_index": page,
            "stock_list": self._stock_market_code(stock_code),
            "begin_time": start_date.replace("-", ""),
            "end_time": end_date.replace("-", ""),
            "f_node": "0",
            "s_node": "0",
        }
        try:
            self._log(f"搜索 {stock_code} 页码: {page}")
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                self._log(f"API返回异常 code={data.get('code')}")
                return []
            raw = data.get("data", {}).get("list", []) or []
            items = []
            for item in raw:
                title = item.get("art_title", "")
                items.append(
                    {
                        "title": title,
                        "url": item.get("adjunct_url", ""),
                        "announce_date": _normalize_date(item.get("noticed_date")),
                        "report_year": _extract_year(title),
                        "announcement_id": item.get("art_code", ""),
                    }
                )
            if items:
                self._log(f"获取到 {len(items)} 条公告")
            return items
        except Exception as e:
            self._log(f"搜索失败 {stock_code}: {str(e)}")
            return []


# ==================== 新浪财经 ====================


class SinaSpider(BaseSpider):
    source_name = "sina"
    request_delay = 1.5
    headers = {
        **COMMON_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://vip.stock.finance.sina.com.cn/",
    }

    def search_page(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        category_cfg: Dict[str, Any],
        page: int,
    ) -> List[Dict[str, Any]]:
        # 新浪一次拉全列表；仅第一页有数据
        if page > 1:
            return []
        del category_cfg
        url = (
            "https://vip.stock.finance.sina.com.cn/corp/go.php/"
            f"vCB_AllBulletin/stockid/{stock_code}.phtml"
        )
        try:
            self._log(f"搜索 {stock_code}")
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "gb2312"
            items = self._parse_bulletin_page(resp.text)
            items = [
                a
                for a in items
                if (not start_date or (a.get("announce_date") or "") >= start_date)
                and (a.get("announce_date") or "") <= end_date
            ]
            self._log(f"获取到 {len(items)} 条公告")
            return items
        except Exception as e:
            self._log(f"搜索失败 {stock_code}: {str(e)}")
            return []

    def _build_pdf_url(self, href: str) -> str:
        if not href:
            return ""
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"https://vip.stock.finance.sina.com.cn{href}"
        return f"https://vip.stock.finance.sina.com.cn/{href}"

    def _parse_bulletin_page(self, html: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        row_pattern = re.compile(
            r"<tr[^>]*>.*?"
            r"<td[^>]*>.*?(\d{4}-\d{2}-\d{2}).*?</td>.*?"
            r'<td[^>]*>.*?<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>.*?</td>',
            re.DOTALL | re.IGNORECASE,
        )
        for date_str, href, title in row_pattern.findall(html):
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            items.append(
                {
                    "title": title_clean,
                    "url": self._build_pdf_url(href),
                    "announce_date": date_str,
                    "report_year": _extract_year(title_clean),
                    "announcement_id": href,
                }
            )

        if items:
            return items

        link_pattern = re.compile(
            r'<a[^>]*href=["\']([^"\']*(?:announcement|bulletin|PDF)[^"\']*)["\'][^>]*>'
            r"(.*?20\d{2}.*?)"
            r"</a>",
            re.DOTALL | re.IGNORECASE,
        )
        dates = re.findall(r"(\d{4}-\d{2}-\d{2})", html)
        for i, (href, title) in enumerate(link_pattern.findall(html)):
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            items.append(
                {
                    "title": title_clean,
                    "url": self._build_pdf_url(href),
                    "announce_date": dates[i] if i < len(dates) else "",
                    "report_year": _extract_year(title_clean),
                    "announcement_id": href,
                }
            )
        return items


SOURCE_PRIORITY = [
    ("cninfo", CninfoSpider),
    ("eastmoney", EastmoneySpider),
    ("sina", SinaSpider),
]


def _create_spider(source_name: str, save_dir: str | None, delay: float | None):
    spider_map = {
        "cninfo": CninfoSpider,
        "eastmoney": EastmoneySpider,
        "sina": SinaSpider,
    }
    spider_cls = spider_map.get(source_name)
    if spider_cls is None:
        return None
    return spider_cls(save_dir=save_dir, delay=delay)


def download_single_stock(
    stock_code: str,
    save_dir: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    report_type: str = "annual",
    category: str | None = None,
    delay: float | None = None,
    preferred_source: str | None = None,
    max_results: int | None = None,
) -> tuple:
    sources_to_try = list(SOURCE_PRIORITY)
    if preferred_source:
        sources_to_try = [
            (name, cls) for name, cls in sources_to_try if name == preferred_source
        ] + [(name, cls) for name, cls in sources_to_try if name != preferred_source]

    for source_name, _ in sources_to_try:
        spider = _create_spider(source_name, save_dir, delay)
        if spider is None:
            continue
        try:
            print(f"\n  [*] 尝试数据源 [{source_name}]: {stock_code}")
            reports = spider.get_reports(
                stock_code=stock_code,
                start_year=start_year,
                end_year=end_year,
                report_type=report_type,
                category=category,
                max_results=max_results,
            )
            if reports:
                print(f"  [OK] [{source_name}] 成功下载 {len(reports)} 个文件")
                return reports, source_name
            print(f"  [WARN] [{source_name}] 返回 0 条结果，切换下一数据源")
        except Exception as e:
            print(f"  [ERR] [{source_name}] 异常: {str(e)[:80]}，切换下一数据源")
            continue

    print(f"  [FAIL] 所有数据源均失败: {stock_code}")
    return [], None


def batch_download(
    stock_codes: List[str],
    save_dir: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    report_type: str = "annual",
    category: str | None = None,
    delay: float | None = None,
    preferred_source: str | None = None,
    max_results: int | None = None,
) -> Dict[str, List[Dict]]:
    results: Dict[str, List[Dict]] = {}
    source_stats: Dict[str, int] = {}

    for i, code in enumerate(stock_codes):
        print(f"\n{'=' * 50}")
        print(f"[{i + 1}/{len(stock_codes)}] 处理股票: {code}")
        print(f"{'=' * 50}")

        downloaded, source_used = download_single_stock(
            stock_code=code,
            save_dir=save_dir,
            start_year=start_year,
            end_year=end_year,
            report_type=report_type,
            category=category,
            delay=delay,
            preferred_source=preferred_source,
            max_results=max_results,
        )
        results[code] = downloaded
        key = source_used or "失败"
        source_stats[key] = source_stats.get(key, 0) + 1

        if i < len(stock_codes) - 1:
            _random_sleep(delay or DOWNLOAD_DELAY)

    print(f"\n{'=' * 50}")
    print("数据源使用统计:")
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"   {src}: {count} 只股票")
    print(f"{'=' * 50}")
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
        description="A股上市公司公告下载工具（多数据源自动切换）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_report.py --stock 600519 --start 2020 --end 2024
  python scripts/download_report.py --stock 002027 --type annual --category 年报
  python scripts/download_report.py --stock 002027 --type ir --start 2023 --end 2025

数据源优先级: 巨潮资讯网 → 东方财富 → 新浪财经
默认目录:
  annual -> corpus/01_raw/annual_reports/{代码}/
  ir     -> corpus/01_raw/ir_activities/{代码}/
        """,
    )
    parser.add_argument(
        "--stock",
        required=True,
        help="股票代码，多个用逗号分隔，如 600519 或 600519,000858",
    )
    parser.add_argument(
        "--type",
        dest="report_type",
        choices=["annual", "ir"],
        default="annual",
        help="下载类型: annual=定期报告, ir=投资者关系活动记录",
    )
    parser.add_argument("--start", type=int, help="起始年份，默认当前年份-5")
    parser.add_argument("--end", type=int, help="结束年份，默认当前年份")
    parser.add_argument(
        "--category",
        choices=["年报", "半年报", "季报"],
        default="年报",
        help="仅 --type annual 生效",
    )
    parser.add_argument("--max-results", type=int, default=None, help="每只股票最多下载数量")
    parser.add_argument(
        "--save-dir",
        default=None,
        help="PDF保存目录（默认按 --type 选择 annual_reports / ir_activities）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="请求间隔基准秒数（实际会在约 0.8x~2.0x 内随机抖动）",
    )
    parser.add_argument(
        "--source",
        choices=["cninfo", "eastmoney", "sina"],
        default=None,
        help="指定数据源（可选，默认自动切换）",
    )

    args = parser.parse_args()

    try:
        codes = parse_stock_codes(args.stock)
    except ValueError as e:
        parser.error(str(e))

    current_year = datetime.now().year
    start_year = args.start if args.start else current_year - 5
    end_year = args.end if args.end else current_year
    report_type = args.report_type
    category = None if report_type == "ir" else args.category

    # 显式传入 --save-dir 才覆盖；否则按 type 选默认目录
    save_dir_arg = args.save_dir
    if save_dir_arg is None:
        save_dir = default_save_dir(report_type)
    else:
        save_dir = resolve_save_dir(save_dir_arg, report_type)

    type_label = "投资者关系活动记录" if report_type == "ir" else f"定期报告/{category}"
    print("准备下载公告")
    print(f"   股票代码: {', '.join(codes)}")
    print(f"   下载类型: {type_label}")
    print(f"   年份范围: {start_year} ~ {end_year}")
    print(f"   保存目录: {save_dir}")
    print(f"   项目根目录: {_PROJECT_ROOT}")
    if args.source:
        print(f"   指定数据源: {args.source}")
    else:
        print("   数据源策略: 自动切换（cninfo → eastmoney → sina）")

    results = batch_download(
        stock_codes=codes,
        save_dir=save_dir,
        start_year=start_year,
        end_year=end_year,
        report_type=report_type,
        category=category,
        delay=args.delay,
        preferred_source=args.source,
        max_results=args.max_results,
    )

    total = sum(len(v) for v in results.values())
    print(f"\n下载完成: {len(codes)} 只股票, 共 {total} 个文件")

    for code, reports in results.items():
        if reports:
            years = sorted({r.get("year") for r in reports if r.get("year")})
            sources = {r.get("source", "?") for r in reports}
            print(f"   {code}: {len(reports)} 个文件, 年份: {years}, 来源: {', '.join(sources)}")
            for report in reports:
                file_path = report.get("file_path")
                if file_path:
                    print(f"   文件: {file_path}")

    if total == 0:
        print("\n[WARN] 未下载到任何文件，可能原因：")
        print("   1. 年份范围内无对应报告")
        print("   2. 所有数据源均无法访问（网络问题或反爬限制）")
        print("   3. 建议稍后重试，或通过 --source 指定特定数据源")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
