"""A股上市公司财报下载工具（多数据源，自动切换）。

按优先级尝试：巨潮资讯网 → 东方财富 → 新浪财经。
默认保存到 corpus/01_raw/annual_reports/{代码}/，可直接交给
scripts/split_annual_reports.py 切分。

示例：

    python scripts/download_report.py --stock 002027 --start 2020 --end 2024
    python scripts/download_report.py --stock 600519,000858 --start 2020 --end 2024
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# scripts/ -> 项目根
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_DIR = str(_PROJECT_ROOT / "corpus" / "01_raw" / "annual_reports")


# ==================== 巨潮资讯网 ====================


class CninfoConfig:
    BASE_URL = "http://www.cninfo.com.cn/new/index"
    SEARCH_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    DOWNLOAD_BASE = "http://static.cninfo.com.cn"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "http://www.cninfo.com.cn/new/index",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }
    REQUEST_DELAY = 0.5
    DOWNLOAD_DELAY = 1.0
    MAX_RETRIES = 3

    CATEGORY_KEYWORDS = {
        "年报": ["年度报告", "年报"],
        "半年报": ["半年度报告", "半年报"],
        "季报": ["季度报告", "季报", "第一季度报告", "第三季度报告"],
    }


class CninfoSpider:
    """巨潮资讯网爬虫"""

    source_name = "cninfo"

    def __init__(self, save_dir: str | None = None, delay: float | None = None):
        if save_dir is None:
            save_dir = DEFAULT_SAVE_DIR
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay if delay is not None else CninfoConfig.REQUEST_DELAY
        self.session = requests.Session()
        self.session.headers.update(CninfoConfig.HEADERS)

    def _log(self, message: str):
        print(f"  [cninfo] {message}")

    def _get_stock_orgid(self, stock_code: str) -> Optional[Tuple[str, str]]:
        """通过股票代码获取orgId"""
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

    def search_announcements(
        self,
        stock_code: str,
        start_date: str = "1900-01-01",
        end_date: str | None = None,
        category: str = "年报",
        page_num: int = 1,
        page_size: int = 30,
    ) -> List[Dict]:
        """搜索公告列表"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        stock_info = self._get_stock_orgid(stock_code)
        if not stock_info:
            return []

        org_id, _ = stock_info
        stock_param = f"{stock_code},{org_id}"

        if stock_code.startswith(("6", "5")):
            plate, column = "sh", "sse"
        elif stock_code.startswith(("0", "3", "2")):
            plate, column = "sz", "szse"
        else:
            plate, column = "bj", "bj"

        category_value = {
            "年报": "category_ndbg_szsh",
            "半年报": "category_sjdbg_szsh",
            "季报": "category_yjdbg_szsh",
        }.get(category, "category_ndbg_szsh")

        data = {
            "stock": stock_param,
            "tabName": "fulltext",
            "pageSize": str(page_size),
            "pageNum": str(page_num),
            "column": column,
            "category": category_value,
            "plate": plate,
            "seDate": f"{start_date}~{end_date}",
            "searchkey": "",
            "secid": "",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        try:
            self._log(f"搜索 {stock_code} 页码: {page_num}")
            resp = self.session.post(CninfoConfig.SEARCH_URL, data=data, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            announcements = result.get("announcements", []) or []
            for item in announcements:
                title = item.get("announcementTitle", "")
                year_match = re.search(r"(20\d{2}|19\d{2})", title)
                item["report_year"] = int(year_match.group(1)) if year_match else None
            return announcements
        except Exception as e:
            self._log(f"搜索失败 {stock_code}: {str(e)}")
            return []

    def download_pdf(
        self,
        stock_code: str,
        save_name: str,
        adjunct_url: str,
        overwrite: bool = False,
    ) -> bool:
        """下载PDF文件"""
        if not adjunct_url:
            self._log(f"跳过 {save_name}: 无下载链接")
            return False

        save_path = self.save_dir / stock_code / f"{save_name}.pdf"
        if save_path.exists() and not overwrite:
            self._log(f"跳过 {save_name}: 文件已存在")
            return True

        save_path.parent.mkdir(parents=True, exist_ok=True)
        download_url = f"{CninfoConfig.DOWNLOAD_BASE}/{adjunct_url.lstrip('/')}"

        for attempt in range(CninfoConfig.MAX_RETRIES):
            try:
                self._log(f"下载 [{attempt+1}/{CninfoConfig.MAX_RETRIES}]: {save_name}")
                resp = self.session.get(download_url, timeout=30)
                resp.raise_for_status()

                if b"%PDF" not in resp.content[:100]:
                    if attempt + 1 < CninfoConfig.MAX_RETRIES:
                        time.sleep(self.delay * (attempt + 1))
                        continue
                    self._log(f"下载失败 {save_name}: 文件可能无效")
                    return False

                with open(save_path, "wb") as f:
                    f.write(resp.content)
                self._log(f"下载完成: {save_path} ({len(resp.content) / 1024:.2f} KB)")
                return True
            except Exception as e:
                self._log(f"下载重试中... {str(e)}")
                time.sleep(self.delay * (attempt + 1))

        self._log(f"最终下载失败: {save_name}")
        return False

    def get_reports(
        self,
        stock_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
        categories: List[str] | None = None,
    ) -> List[Dict]:
        """获取并下载指定股票的财报"""
        if categories is None:
            categories = ["年报"]

        start_date = f"{start_year}-01-01" if start_year else "1900-01-01"
        if end_year:
            end_date = min(f"{end_year + 1}-06-30", datetime.now().strftime("%Y-%m-%d"))
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")

        all_announcements: List[Dict] = []
        page = 1
        category = categories[0] if categories else "年报"

        while True:
            items = self.search_announcements(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                category=category,
                page_num=page,
            )
            if not items:
                break
            all_announcements.extend(items)
            if len(items) < 30:
                break
            page += 1
            time.sleep(self.delay)

        if start_year:
            all_announcements = [
                a
                for a in all_announcements
                if a.get("report_year") and start_year <= a["report_year"] <= (end_year or 9999)
            ]

        keywords = CninfoConfig.CATEGORY_KEYWORDS.get(category, [category])
        all_announcements = [
            a
            for a in all_announcements
            if any(kw in a.get("announcementTitle", "") for kw in keywords)
        ]

        def _title_priority(title: str) -> int:
            if "摘要" in title:
                return 2
            if "英文版" in title or "英文" in title or "English" in title:
                return 1
            return 0

        all_announcements.sort(key=lambda x: _title_priority(x.get("announcementTitle", "")))
        seen_years: set = set()
        deduped = []
        for a in all_announcements:
            y = a.get("report_year")
            if y and y not in seen_years:
                seen_years.add(y)
                deduped.append(a)
        all_announcements = deduped

        downloaded = []
        for item in all_announcements:
            title = item.get("announcementTitle", "").replace("/", "_").replace(" ", "_")
            year_info = f"_{item.get('report_year')}" if item.get("report_year") else ""
            save_name = f"{stock_code}_{title[:50]}{year_info}"

            success = self.download_pdf(
                stock_code=stock_code,
                save_name=save_name,
                adjunct_url=item.get("adjunctUrl", ""),
            )
            if success:
                downloaded.append(
                    {
                        "stock_code": stock_code,
                        "title": title,
                        "file_path": str(self.save_dir / stock_code / f"{save_name}.pdf"),
                        "year": item.get("report_year"),
                        "source": self.source_name,
                    }
                )
            time.sleep(CninfoConfig.DOWNLOAD_DELAY)

        return downloaded


# ==================== 东方财富 ====================


class EastmoneySpider:
    """东方财富财报爬虫"""

    source_name = "eastmoney"

    SEARCH_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://data.eastmoney.com/",
    }
    REQUEST_DELAY = 1.0
    DOWNLOAD_DELAY = 1.0
    MAX_RETRIES = 3

    CATEGORY_KEYWORDS = {
        "年报": ["年度报告", "年报"],
        "半年报": ["半年度报告", "半年报", "中期报告"],
        "季报": ["季度报告", "季报", "第一季度报告", "第三季度报告"],
    }

    def __init__(self, save_dir: str | None = None, delay: float | None = None):
        if save_dir is None:
            save_dir = DEFAULT_SAVE_DIR
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay if delay is not None else self.REQUEST_DELAY
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _log(self, message: str):
        print(f"  [eastmoney] {message}")

    def _stock_market_code(self, code: str) -> str:
        """获取东方财富格式的股票代码（带交易所前缀）"""
        if code.startswith(("6", "5")):
            return f"SH{code}"
        if code.startswith(("0", "3", "2")):
            return f"SZ{code}"
        if code.startswith(("8", "4")):
            return f"BJ{code}"
        return f"SZ{code}"

    def search_announcements(
        self,
        stock_code: str,
        start_date: str = "1900-01-01",
        end_date: str | None = None,
        category: str = "年报",
        page: int = 1,
        page_size: int = 30,
    ) -> List[Dict]:
        """搜索公告列表"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        stock_list = self._stock_market_code(stock_code)

        params = {
            "page_size": page_size,
            "page_index": page,
            "stock_list": stock_list,
            "begin_time": start_date.replace("-", ""),
            "end_time": end_date.replace("-", ""),
            "f_node": "0",
            "s_node": "0",
        }

        try:
            self._log(f"搜索 {stock_code} ({stock_list}) 页码: {page}")
            resp = self.session.get(self.SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != 0:
                self._log(f"API返回异常 code={data.get('code')}: {data.get('message', '')}")
                return []

            items = data.get("data", {}).get("list", [])
            if not items:
                return []

            announcements = []
            for item in items:
                title = item.get("art_title", "")
                adjunct_url = item.get("adjunct_url", "")

                year_match = re.search(r"(20\d{2}|19\d{2})", title)
                year = int(year_match.group(1)) if year_match else None

                announcements.append(
                    {
                        "announcementTitle": title,
                        "adjunctUrl": adjunct_url,
                        "report_year": year,
                        "announcementId": item.get("art_code", ""),
                        "announcementTime": item.get("noticed_date", ""),
                    }
                )

            if announcements:
                self._log(f"获取到 {len(announcements)} 条公告")
            return announcements
        except requests.RequestException as e:
            self._log(f"网络请求失败: {str(e)}")
            return []
        except Exception as e:
            self._log(f"搜索失败 {stock_code}: {str(e)}")
            return []

    def download_pdf(
        self,
        stock_code: str,
        save_name: str,
        adjunct_url: str,
        overwrite: bool = False,
    ) -> bool:
        """下载PDF文件"""
        if not adjunct_url:
            self._log(f"跳过 {save_name}: 无下载链接")
            return False

        save_path = self.save_dir / stock_code / f"{save_name}.pdf"
        if save_path.exists() and not overwrite:
            self._log(f"跳过 {save_name}: 文件已存在")
            return True

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if adjunct_url.startswith("http"):
            download_url = adjunct_url
        elif adjunct_url.startswith("/"):
            download_url = f"https://np-anotice-stock.eastmoney.com{adjunct_url}"
        else:
            download_url = f"https://np-anotice-stock.eastmoney.com/{adjunct_url}"

        for attempt in range(self.MAX_RETRIES):
            try:
                self._log(f"下载 [{attempt+1}/{self.MAX_RETRIES}]: {save_name}")
                resp = self.session.get(download_url, timeout=30)
                resp.raise_for_status()

                if b"%PDF" not in resp.content[:100]:
                    if attempt + 1 < self.MAX_RETRIES:
                        time.sleep(self.delay * (attempt + 1))
                        continue
                    self._log(f"下载失败 {save_name}: 文件不是有效PDF")
                    return False

                with open(save_path, "wb") as f:
                    f.write(resp.content)
                self._log(f"下载完成: {save_path} ({len(resp.content) / 1024:.2f} KB)")
                return True
            except Exception as e:
                self._log(f"下载重试中... {str(e)}")
                time.sleep(self.delay * (attempt + 1))

        self._log(f"最终下载失败: {save_name}")
        return False

    def get_reports(
        self,
        stock_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
        categories: List[str] | None = None,
    ) -> List[Dict]:
        """获取并下载指定股票的财报"""
        if categories is None:
            categories = ["年报"]

        start_date = f"{start_year}-01-01" if start_year else "1900-01-01"
        if end_year:
            end_date = f"{end_year + 1}-06-30"
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")

        category = categories[0] if categories else "年报"

        all_announcements: List[Dict] = []
        page = 1

        while True:
            items = self.search_announcements(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                category=category,
                page=page,
            )
            if not items:
                break
            all_announcements.extend(items)
            if len(items) < 30:
                break
            page += 1
            time.sleep(self.delay)

        if start_year:
            all_announcements = [
                a
                for a in all_announcements
                if a.get("report_year") and start_year <= a["report_year"] <= (end_year or 9999)
            ]

        keywords = self.CATEGORY_KEYWORDS.get(category, [category])
        all_announcements = [
            a
            for a in all_announcements
            if any(kw in a.get("announcementTitle", "") for kw in keywords)
        ]

        seen_years: set = set()
        deduped = []
        for a in all_announcements:
            y = a.get("report_year")
            if y and y not in seen_years:
                seen_years.add(y)
                deduped.append(a)
        all_announcements = deduped

        downloaded = []
        for item in all_announcements:
            title = item.get("announcementTitle", "").replace("/", "_").replace(" ", "_")
            year_info = f"_{item.get('report_year')}" if item.get("report_year") else ""
            save_name = f"{stock_code}_{title[:50]}{year_info}"

            success = self.download_pdf(
                stock_code=stock_code,
                save_name=save_name,
                adjunct_url=item.get("adjunctUrl", ""),
            )
            if success:
                downloaded.append(
                    {
                        "stock_code": stock_code,
                        "title": title,
                        "file_path": str(self.save_dir / stock_code / f"{save_name}.pdf"),
                        "year": item.get("report_year"),
                        "source": self.source_name,
                    }
                )
            time.sleep(self.DOWNLOAD_DELAY)

        return downloaded


# ==================== 新浪财经 ====================


class SinaSpider:
    """新浪财经财报爬虫"""

    source_name = "sina"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://vip.stock.finance.sina.com.cn/",
    }
    REQUEST_DELAY = 1.5
    DOWNLOAD_DELAY = 1.0
    MAX_RETRIES = 3

    CATEGORY_KEYWORDS = {
        "年报": ["年度报告", "年报"],
        "半年报": ["半年度报告", "半年报", "中期报告"],
        "季报": ["季度报告", "季报", "第一季度报告", "第三季度报告"],
    }

    def __init__(self, save_dir: str | None = None, delay: float | None = None):
        if save_dir is None:
            save_dir = DEFAULT_SAVE_DIR
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay if delay is not None else self.REQUEST_DELAY
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def _log(self, message: str):
        print(f"  [sina] {message}")

    def search_announcements(
        self,
        stock_code: str,
        start_date: str = "1900-01-01",
        end_date: str | None = None,
        category: str = "年报",
        page: int = 1,
        page_size: int = 30,
    ) -> List[Dict]:
        """搜索公告列表（通过新浪财经公告页面）"""
        del page, page_size  # 新浪页一次拉全列表
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        url = (
            "https://vip.stock.finance.sina.com.cn/corp/go.php/"
            f"vCB_AllBulletin/stockid/{stock_code}.phtml"
        )

        try:
            self._log(f"搜索 {stock_code}")
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = "gb2312"
            html = resp.text

            announcements = self._parse_bulletin_page(html, stock_code)

            keywords = self.CATEGORY_KEYWORDS.get(category, [category])
            announcements = [
                a
                for a in announcements
                if any(kw in a.get("announcementTitle", "") for kw in keywords)
            ]

            if start_date and start_date != "1900-01-01":
                announcements = [
                    a for a in announcements if a.get("announcementTime", "") >= start_date
                ]
            announcements = [
                a for a in announcements if a.get("announcementTime", "") <= end_date
            ]

            self._log(f"获取到 {len(announcements)} 条公告")
            return announcements
        except requests.RequestException as e:
            self._log(f"网络请求失败: {str(e)}")
            return []
        except Exception as e:
            self._log(f"搜索失败 {stock_code}: {str(e)}")
            return []

    def _parse_bulletin_page(self, html: str, stock_code: str) -> List[Dict]:
        """解析新浪财经公告页面HTML"""
        del stock_code
        announcements = []

        row_pattern = re.compile(
            r"<tr[^>]*>.*?"
            r"<td[^>]*>.*?(\d{4}-\d{2}-\d{2}).*?</td>.*?"
            r'<td[^>]*>.*?<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>.*?</td>',
            re.DOTALL | re.IGNORECASE,
        )

        matches = row_pattern.findall(html)
        for date_str, href, title in matches:
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            year_match = re.search(r"(20\d{2}|19\d{2})", title_clean)
            year = int(year_match.group(1)) if year_match else None
            adjunct_url = self._build_pdf_url(href)

            announcements.append(
                {
                    "announcementTitle": title_clean,
                    "adjunctUrl": adjunct_url,
                    "report_year": year,
                    "announcementTime": date_str,
                }
            )

        if not announcements:
            link_pattern = re.compile(
                r'<a[^>]*href=["\']([^"\']*(?:announcement|bulletin|PDF)[^"\']*)["\'][^>]*>'
                r"(.*?20\d{2}.*?)"
                r"</a>",
                re.DOTALL | re.IGNORECASE,
            )
            date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")

            links = link_pattern.findall(html)
            dates = date_pattern.findall(html)

            for i, (href, title) in enumerate(links):
                title_clean = re.sub(r"<[^>]+>", "", title).strip()
                year_match = re.search(r"(20\d{2}|19\d{2})", title_clean)
                year = int(year_match.group(1)) if year_match else None

                date_str = dates[i] if i < len(dates) else ""
                adjunct_url = self._build_pdf_url(href)

                announcements.append(
                    {
                        "announcementTitle": title_clean,
                        "adjunctUrl": adjunct_url,
                        "report_year": year,
                        "announcementTime": date_str,
                    }
                )

        return announcements

    def _build_pdf_url(self, href: str) -> str:
        """构建PDF下载链接"""
        if not href:
            return ""
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"https://vip.stock.finance.sina.com.cn{href}"
        return f"https://vip.stock.finance.sina.com.cn/{href}"

    def download_pdf(
        self,
        stock_code: str,
        save_name: str,
        adjunct_url: str,
        overwrite: bool = False,
    ) -> bool:
        """下载PDF文件"""
        if not adjunct_url:
            self._log(f"跳过 {save_name}: 无下载链接")
            return False

        save_path = self.save_dir / stock_code / f"{save_name}.pdf"
        if save_path.exists() and not overwrite:
            self._log(f"跳过 {save_name}: 文件已存在")
            return True

        save_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(self.MAX_RETRIES):
            try:
                self._log(f"下载 [{attempt+1}/{self.MAX_RETRIES}]: {save_name}")
                resp = self.session.get(adjunct_url, timeout=30)

                if resp.status_code == 200:
                    content_type = resp.headers.get("Content-Type", "")

                    if "text/html" in content_type and b"%PDF" not in resp.content[:100]:
                        pdf_match = re.search(
                            rb'href=["\']([^"\']*\.pdf[^"\']*)["\']',
                            resp.content,
                            re.IGNORECASE,
                        )
                        if pdf_match:
                            pdf_url = pdf_match.group(1).decode("utf-8", errors="ignore")
                            if not pdf_url.startswith("http"):
                                pdf_url = (
                                    "https://vip.stock.finance.sina.com.cn/"
                                    f"{pdf_url.lstrip('/')}"
                                )
                            resp = self.session.get(pdf_url, timeout=30)
                            resp.raise_for_status()

                if b"%PDF" not in resp.content[:100]:
                    if attempt + 1 < self.MAX_RETRIES:
                        time.sleep(self.delay * (attempt + 1))
                        continue
                    self._log(f"下载失败 {save_name}: 文件不是有效PDF")
                    return False

                with open(save_path, "wb") as f:
                    f.write(resp.content)
                self._log(f"下载完成: {save_path} ({len(resp.content) / 1024:.2f} KB)")
                return True
            except Exception as e:
                self._log(f"下载重试中... {str(e)}")
                time.sleep(self.delay * (attempt + 1))

        self._log(f"最终下载失败: {save_name}")
        return False

    def get_reports(
        self,
        stock_code: str,
        start_year: int | None = None,
        end_year: int | None = None,
        categories: List[str] | None = None,
    ) -> List[Dict]:
        """获取并下载指定股票的财报"""
        if categories is None:
            categories = ["年报"]

        start_date = f"{start_year}-01-01" if start_year else "1900-01-01"
        if end_year:
            end_date = f"{end_year + 1}-06-30"
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")

        category = categories[0] if categories else "年报"

        all_announcements = self.search_announcements(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            category=category,
            page=1,
        )

        if start_year:
            all_announcements = [
                a
                for a in all_announcements
                if a.get("report_year") and start_year <= a["report_year"] <= (end_year or 9999)
            ]

        seen_years: set = set()
        deduped = []
        for a in all_announcements:
            y = a.get("report_year")
            if y and y not in seen_years:
                seen_years.add(y)
                deduped.append(a)
        all_announcements = deduped

        downloaded = []
        for item in all_announcements:
            title = item.get("announcementTitle", "").replace("/", "_").replace(" ", "_")
            year_info = f"_{item.get('report_year')}" if item.get("report_year") else ""
            save_name = f"{stock_code}_{title[:50]}{year_info}"

            success = self.download_pdf(
                stock_code=stock_code,
                save_name=save_name,
                adjunct_url=item.get("adjunctUrl", ""),
            )
            if success:
                downloaded.append(
                    {
                        "stock_code": stock_code,
                        "title": title,
                        "file_path": str(self.save_dir / stock_code / f"{save_name}.pdf"),
                        "year": item.get("report_year"),
                        "source": self.source_name,
                    }
                )
            time.sleep(self.DOWNLOAD_DELAY)

        return downloaded


# 数据源优先级（按可靠性排序）
SOURCE_PRIORITY = [
    ("cninfo", CninfoSpider),
    ("eastmoney", EastmoneySpider),
    ("sina", SinaSpider),
]


# ==================== 多源下载引擎 ====================


def resolve_save_dir(save_dir: str | None) -> str:
    """相对路径一律相对项目根解析，避免 cwd 导致写到错误位置。"""
    if not save_dir:
        return DEFAULT_SAVE_DIR
    path = Path(save_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


def parse_stock_codes(raw: str) -> List[str]:
    """解析逗号分隔的股票代码，仅接受 6 位数字。"""
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


def _create_spider(source_name: str, save_dir: str | None, delay: float | None):
    """根据名称创建爬虫实例"""
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
    categories: List[str] | None = None,
    delay: float | None = None,
    preferred_source: str | None = None,
) -> tuple:
    """
    下载单只股票的财报，自动切换数据源

    Returns:
        (downloaded_list, used_source_name)
    """
    if categories is None:
        categories = ["年报"]

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
                categories=categories,
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
    categories: List[str] | None = None,
    delay: float | None = None,
    preferred_source: str | None = None,
) -> Dict[str, List[Dict]]:
    """批量下载多只股票的财报（多数据源自动切换）"""
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
            categories=categories,
            delay=delay,
            preferred_source=preferred_source,
        )
        results[code] = downloaded

        if source_used:
            source_stats[source_used] = source_stats.get(source_used, 0) + 1
        else:
            source_stats["失败"] = source_stats.get("失败", 0) + 1

        if i < len(stock_codes) - 1:
            time.sleep(delay or 1.0)

    print(f"\n{'=' * 50}")
    print("数据源使用统计:")
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"   {src}: {count} 只股票")
    print(f"{'=' * 50}")

    return results


def _ensure_utf8_stdio() -> None:
    """Windows GBK 控制台无法打印部分 Unicode，强制 stdout/stderr 用 UTF-8。"""
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
        description="A股上市公司财报下载工具（多数据源自动切换）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_report.py --stock 600519 --start 2020 --end 2024
  python scripts/download_report.py --stock 600519,000858 --start 2020 --end 2024 --category 年报
  python scripts/download_report.py --stock 002027 --source cninfo

数据源优先级: 巨潮资讯网 → 东方财富 → 新浪财经
某个数据源失败时自动切换下一数据源。
默认保存到 corpus/01_raw/annual_reports/{代码}/。
仅接受 6 位股票代码，不支持按名称查询。
        """,
    )

    parser.add_argument(
        "--stock",
        required=True,
        help="股票代码，多个用逗号分隔，如 600519 或 600519,000858",
    )
    parser.add_argument("--start", type=int, help="起始年份，默认当前年份-5")
    parser.add_argument("--end", type=int, help="结束年份，默认当前年份")
    parser.add_argument("--category", choices=["年报", "半年报", "季报"], default="年报")
    parser.add_argument("--max-results", type=int, default=None, help="每只股票最多下载数量")
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help=f"PDF保存目录（默认: {DEFAULT_SAVE_DIR}）",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="请求间隔秒数")
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

    save_dir = resolve_save_dir(args.save_dir)

    print("准备下载财报")
    print(f"   股票代码: {', '.join(codes)}")
    print(f"   报告类型: {args.category}")
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
        categories=[args.category],
        delay=args.delay,
        preferred_source=args.source,
    )

    if args.max_results:
        for code in results:
            results[code] = results[code][: args.max_results]

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
