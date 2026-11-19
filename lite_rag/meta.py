"""文档元信息：目录、文件名、文首 YAML，以及查询时的筛选条件。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import PurePosixPath

SCOPES = ("company", "knowledge")
DOC_TYPES = ("filing", "ir_qa", "survey", "research", "news", "investment_theory", "beginner")
DOC_TYPE_LABELS = {
    "filing": "定期报告",
    "ir_qa": "互动问答",
    "survey": "投资者调研",
    "research": "研报",
    "news": "新闻",
    "investment_theory": "投资理论",
    "beginner": "入门答疑",
}
TYPES_BY_SCOPE = {
    "company": frozenset({"filing", "ir_qa", "survey", "research", "news"}),
    "knowledge": frozenset({"investment_theory", "beginner"}),
}
FIGURES = ("唐朝", "buffett", "munger")
FIGURE_LABELS = {
    "唐朝": "唐朝",
    "buffett": "巴菲特",
    "munger": "芒格",
}
OCCASIONS = ("letter", "meeting", "speech", "article")
OCCASION_LABELS = {
    "letter": "致投资者的信",
    "meeting": "股东大会",
    "speech": "演讲",
    "article": "文章",
}
# 采集脚本当前的默认目录，和 company/ir_qa 一样记成互动问答。
_ROOT_ALIASES = {
    "sse": ("company", "ir_qa"),
    "irm": ("company", "ir_qa"),
}
SECTIONS = (
    "profile",
    "segment",
    "mda",
    "risk",
    "material",
    "governance",
    "financials",
    "notes",
)
SECTION_LABELS = {
    "profile": "公司与业务",
    "segment": "分产品分地区",
    "mda": "经营讨论",
    "risk": "风险因素",
    "material": "重要事项",
    "governance": "股东与治理",
    "financials": "财务报表",
    "notes": "附注",
}
ENTITIES = ("consolidated", "parent")
ENTITY_LABELS = {
    "consolidated": "合并",
    "parent": "母公司",
}
# 只有财务报表和附注会分成合并、母公司两套。
_ENTITY_SECTIONS = frozenset({"financials", "notes"})
_FRONT_KEYS = frozenset({
    "scope",
    "doc_type",
    "title",
    "company",
    "stock_code",
    "year",
    "period",
    "published_on",
    "author",
    "section",
    "entity",
    "figure",
    "occasion",
})
_CODE_NAME = re.compile(r"^(?P<code>\d{6})[_-](?P<company>[^_\-]+)")
_PERIOD_IN_NAME = re.compile(r"(20\d{2})\s*(Q[1-4]|H[12])", re.IGNORECASE)
_YEAR_IN_NAME = re.compile(r"20\d{2}")
_DATE_IN_NAME = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
_PERIOD_VALUE = re.compile(r"^20\d{2}(?:Q[1-4]|H[12])?$")
_CODE_SUFFIX = (".SZ", ".SH", ".BJ")


@dataclass(frozen=True)
class DocMeta:
    scope: str
    doc_type: str
    title: str
    company: str
    stock_code: str
    year: int
    period: str
    published_on: str
    author: str
    section: str = ""
    entity: str = ""
    figure: str = ""
    occasion: str = ""

    def chroma(self) -> dict:
        return {
            "scope": self.scope,
            "doc_type": self.doc_type,
            "title": self.title,
            "company": self.company,
            "stock_code": self.stock_code,
            "year": int(self.year),
            "period": self.period,
            "published_on": self.published_on,
            "author": self.author,
            "section": self.section,
            "entity": self.entity,
            "figure": self.figure,
            "occasion": self.occasion,
        }

    def same_as(self, row) -> bool:
        return (
            row["scope"] == self.scope
            and row["doc_type"] == self.doc_type
            and (row["title"] or "") == self.title
            and (row["company"] or "") == self.company
            and (row["stock_code"] or "") == self.stock_code
            and int(row["year"] or 0) == self.year
            and (row["period"] or "") == self.period
            and (row["published_on"] or "") == self.published_on
            and (row["author"] or "") == self.author
            and _row_text(row, "section") == self.section
            and _row_text(row, "entity") == self.entity
            and _row_text(row, "figure") == self.figure
            and _row_text(row, "occasion") == self.occasion
        )


def same_doc_rows(rows, doc: DocMeta) -> bool:
    return bool(rows) and all(doc.same_as(row) for row in rows)


def doc_from_row(row) -> DocMeta:
    return DocMeta(
        scope=row["scope"],
        doc_type=row["doc_type"],
        title=row["title"] or "",
        company=row["company"] or "",
        stock_code=row["stock_code"] or "",
        year=int(row["year"] or 0),
        period=row["period"] or "",
        published_on=row["published_on"] or "",
        author=row["author"] or "",
        section=_row_text(row, "section"),
        entity=_row_text(row, "entity"),
        figure=_row_text(row, "figure"),
        occasion=_row_text(row, "occasion"),
    )


def public_meta(row) -> dict:
    doc = doc_from_row(row)
    return doc.chroma()


def _blank_values() -> dict:
    return {
        "scope": "",
        "doc_type": "",
        "title": "",
        "company": "",
        "stock_code": "",
        "year": 0,
        "period": "",
        "published_on": "",
        "author": "",
        "section": "",
        "entity": "",
        "figure": "",
        "occasion": "",
    }


def _copy_values(doc: DocMeta) -> dict:
    return {
        "scope": doc.scope,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "company": doc.company,
        "stock_code": doc.stock_code,
        "year": doc.year,
        "period": doc.period,
        "published_on": doc.published_on,
        "author": doc.author,
        "section": doc.section,
        "entity": doc.entity,
        "figure": doc.figure,
        "occasion": doc.occasion,
    }


def doc_meta_from_fields(fields: dict, *, base: DocMeta | None = None, title_default: str = "") -> DocMeta:
    """用和文首相同的字段生成元信息。fields 覆盖 base 里已有的值。"""
    if not isinstance(fields, dict):
        raise ValueError("元信息必须是映射")
    if base is None and not fields.get("doc_type"):
        raise ValueError("需要 scope 和 doc_type")
    values = _copy_values(base) if base is not None else _blank_values()
    if fields:
        _apply_front(values, fields)
    if not values["scope"] or not values["doc_type"]:
        raise ValueError("需要 scope 和 doc_type")
    if base is None and not values["title"]:
        values["title"] = title_default
    _align_year_and_period(values)
    _check_section_entity(values)
    _check_figure_occasion(values)
    return DocMeta(**values)


def build_doc_meta(relative: str, stem: str, front: dict | None = None) -> tuple[DocMeta, str | None]:
    """由相对路径、文件名和可选文首生成元信息。返回值和目录不一致时带上警告。"""
    scope, doc_type, explicit = _classify_folder(relative)
    parsed = _from_filename(stem, scope)
    values = {
        "scope": scope,
        "doc_type": doc_type,
        "title": stem,
        "company": parsed.get("company", ""),
        "stock_code": parsed.get("stock_code", ""),
        "year": parsed.get("year", 0),
        "period": parsed.get("period", ""),
        "published_on": parsed.get("published_on", ""),
        "author": "",
        "section": "",
        "entity": "",
        "figure": "",
        "occasion": "",
    }
    provided = set(front or {})
    if front:
        _apply_front(values, front)
    if values["scope"] == "knowledge":
        if "company" not in provided:
            values["company"] = ""
        if "stock_code" not in provided:
            values["stock_code"] = ""
    if not values["title"]:
        values["title"] = stem
    _align_year_and_period(values)
    _check_section_entity(values)
    _check_figure_occasion(values)
    doc = DocMeta(**values)
    warning = None
    if explicit and (doc.scope != scope or doc.doc_type != doc_type):
        warning = (
            f"{relative} 的文首声明为 {doc.scope}/{doc.doc_type}，"
            f"与目录 {scope}/{doc_type} 不一致，已采用文首"
        )
    return doc, warning


def source_label(hit: dict) -> str:
    kind = DOC_TYPE_LABELS.get(hit.get("doc_type") or "", "")
    company = (hit.get("company") or "").strip()
    code = (hit.get("stock_code") or "").strip()
    period = (hit.get("period") or "").strip()
    if not period and int(hit.get("year") or 0):
        period = str(int(hit["year"]))
    published = (hit.get("published_on") or "").strip()
    author = (hit.get("author") or "").strip()
    figure = FIGURE_LABELS.get((hit.get("figure") or "").strip(), "")
    occasion = OCCASION_LABELS.get((hit.get("occasion") or "").strip(), "")
    section = SECTION_LABELS.get((hit.get("section") or "").strip(), "")
    title = (hit.get("title") or hit.get("source_name") or "").strip()
    if company and code:
        who = f"{company}（{code}）"
    else:
        who = company or code
    bits = [bit for bit in (who, kind, period or published, section, figure or author, occasion, title) if bit]
    return " · ".join(bits)


@dataclass(frozen=True)
class SearchFilter:
    scope: str = ""
    doc_type: str = ""
    company: str = ""
    stock_code: str = ""
    year: int = 0
    period: str = ""
    section: str = ""
    entity: str = ""
    figure: str = ""
    occasion: str = ""

    def active(self) -> bool:
        return bool(
            self.scope
            or self.doc_type
            or self.company
            or self.stock_code
            or self.year
            or self.period
            or self.section
            or self.entity
            or self.figure
            or self.occasion
        )

    def describe(self) -> str:
        kind = DOC_TYPE_LABELS.get(self.doc_type, self.doc_type)
        section = SECTION_LABELS.get(self.section, self.section)
        entity = ENTITY_LABELS.get(self.entity, self.entity)
        figure = FIGURE_LABELS.get(self.figure, self.figure)
        occasion = OCCASION_LABELS.get(self.occasion, self.occasion)
        parts = [
            self.scope,
            kind,
            figure,
            occasion,
            self.company,
            self.stock_code,
            self.period,
            section,
            entity,
            str(self.year) if self.year else "",
        ]
        return " ".join(part for part in parts if part)

    def sql(self) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []
        if self.scope:
            clauses.append("scope = ?")
            params.append(self.scope)
        if self.doc_type:
            clauses.append("doc_type = ?")
            params.append(self.doc_type)
        if self.company:
            clauses.append("company = ?")
            params.append(self.company)
        if self.stock_code:
            clauses.append("stock_code = ?")
            params.append(self.stock_code)
        if self.year:
            clauses.append("year = ?")
            params.append(self.year)
        if self.period:
            clauses.append("period = ?")
            params.append(self.period)
        if self.section:
            clauses.append("section = ?")
            params.append(self.section)
        if self.entity:
            clauses.append("entity = ?")
            params.append(self.entity)
        if self.figure:
            clauses.append("figure = ?")
            params.append(self.figure)
        if self.occasion:
            clauses.append("occasion = ?")
            params.append(self.occasion)
        return " AND ".join(clauses), params

    def chroma_where(self) -> dict | None:
        """只包含向量 metadata 上的筛选。scope 用来选择 collection。"""
        parts = []
        if self.doc_type == "investment_theory":
            parts.append({"doc_type": self.doc_type})
        if self.figure:
            parts.append({"figure": self.figure})
        if self.occasion:
            parts.append({"occasion": self.occasion})
        if self.company:
            parts.append({"company": self.company})
        if self.stock_code:
            parts.append({"stock_code": self.stock_code})
        if self.year:
            parts.append({"year": self.year})
        if self.period:
            parts.append({"period": self.period})
        if self.section:
            parts.append({"section": self.section})
        if self.entity:
            parts.append({"entity": self.entity})
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return {"$and": parts}

    @classmethod
    def parse(
        cls,
        *,
        scope: str | None = None,
        doc_type: str | None = None,
        company: str | None = None,
        stock_code: str | None = None,
        year: int | None = None,
        period: str | None = None,
        section: str | None = None,
        entity: str | None = None,
        figure: str | None = None,
        occasion: str | None = None,
    ) -> SearchFilter:
        scope_text = (scope or "").strip()
        type_text = (doc_type or "").strip()
        company_text = (company or "").strip()
        if scope_text and scope_text not in SCOPES:
            raise SystemExit("scope 只能是 company 或 knowledge")
        if type_text and type_text not in DOC_TYPE_LABELS:
            names = "、".join(DOC_TYPES)
            raise SystemExit(f"doc_type 只能是 {names}")
        if scope_text and type_text and type_text not in TYPES_BY_SCOPE[scope_text]:
            raise SystemExit(f"{scope_text} 下面没有 doc_type={type_text}")
        try:
            code = _code(stock_code) if stock_code not in (None, "") else ""
            number = _year(year) if year not in (None, "") else 0
            period_text = "" if period in (None, "") else _period(period)
            section_text = "" if section in (None, "") else _section(section)
            entity_text = "" if entity in (None, "") else _entity(entity)
            figure_text = "" if figure in (None, "") else _figure(figure)
            occasion_text = "" if occasion in (None, "") else _occasion(occasion)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if year not in (None, "") and number == 0:
            raise SystemExit("year 需要是 1900 到 2100 之间的年份")
        if number and _query_includes_filing(scope_text, type_text):
            raise SystemExit("查财报请用 period，不要用 year")
        in_theory = type_text == "investment_theory" or (
            scope_text == "knowledge" and type_text in ("", "investment_theory")
        )
        if (figure_text or occasion_text) and not in_theory:
            raise SystemExit("figure 和 occasion 只用于 investment_theory")
        return cls(
            scope=scope_text,
            doc_type=type_text,
            company=company_text,
            stock_code=code,
            year=number,
            period=period_text,
            section=section_text,
            entity=entity_text,
            figure=figure_text,
            occasion=occasion_text,
        )


def _classify_folder(relative: str) -> tuple[str, str, bool]:
    parts = PurePosixPath(relative).parts
    if not parts:
        raise ValueError("空路径")
    if parts[0] in _ROOT_ALIASES:
        scope, doc_type = _ROOT_ALIASES[parts[0]]
        return scope, doc_type, True
    if parts[0] not in {"company", "knowledge"}:
        return "company", "filing", False
    if len(parts) < 3:
        raise ValueError(
            f"{relative} 需要放在类型子目录下，例如 company/filing/ 或 knowledge/investment_theory/"
        )
    kind = parts[1]
    allowed = TYPES_BY_SCOPE[parts[0]]
    if kind not in allowed:
        names = "、".join(sorted(allowed))
        raise ValueError(f"不认识的目录 {parts[0]}/{kind}。这里可用的类型是 {names}")
    return parts[0], kind, True


def _from_filename(stem: str, scope: str) -> dict:
    found: dict = {}
    if scope == "company":
        matched = _CODE_NAME.match(stem)
        if matched:
            found["stock_code"] = matched.group("code")
            company = re.sub(r"(20\d{2}).*$", "", matched.group("company")).strip(" _-")
            if company:
                found["company"] = company
    period_match = _PERIOD_IN_NAME.search(stem)
    if period_match:
        found["year"] = int(period_match.group(1))
        found["period"] = f"{period_match.group(1)}{period_match.group(2).upper()}"
    else:
        years = _YEAR_IN_NAME.findall(stem)
        if years:
            found["year"] = int(years[-1])
        year = found.get("year")
        if year and ("半年" in stem or "中报" in stem):
            found["period"] = f"{year}H1"
        elif year and "一季" in stem:
            found["period"] = f"{year}Q1"
        elif year and "三季" in stem:
            found["period"] = f"{year}Q3"
        elif year and ("年报" in stem or "年度报告" in stem):
            found["period"] = str(year)
    date_match = _DATE_IN_NAME.search(stem)
    if date_match:
        published = "-".join(date_match.groups())
        try:
            datetime.strptime(published, "%Y-%m-%d")
        except ValueError:
            published = ""
        if published:
            found["published_on"] = published
            found.setdefault("year", int(date_match.group(1)))
    return found


def _apply_front(values: dict, front: dict) -> None:
    if not isinstance(front, dict):
        raise ValueError("文首信息必须是 YAML 映射")
    unknown = sorted(set(front) - _FRONT_KEYS)
    if unknown:
        raise ValueError("不认识的文首字段: " + "、".join(unknown))
    if "scope" in front and front["scope"] not in SCOPES:
        raise ValueError("scope 只能是 company 或 knowledge")
    if "doc_type" in front and front["doc_type"] not in DOC_TYPE_LABELS:
        names = "、".join(DOC_TYPES)
        raise ValueError(f"doc_type 只能是 {names}")
    if "scope" in front and front["scope"]:
        values["scope"] = front["scope"]
    if "doc_type" in front and front["doc_type"]:
        values["doc_type"] = front["doc_type"]
        if "scope" not in front or not front.get("scope"):
            values["scope"] = _scope_for(front["doc_type"])
    if values["doc_type"] not in TYPES_BY_SCOPE[values["scope"]]:
        raise ValueError(f"{values['scope']} 不能使用 doc_type={values['doc_type']}")
    if "title" in front:
        values["title"] = "" if front["title"] is None else str(front["title"]).strip()
    if "company" in front:
        values["company"] = "" if front["company"] is None else str(front["company"]).strip()
    if "stock_code" in front:
        values["stock_code"] = "" if front["stock_code"] in (None, "") else _code(front["stock_code"])
    if "year" in front:
        values["year"] = 0 if front["year"] in (None, "") else _year(front["year"])
    if "period" in front:
        values["period"] = "" if front["period"] in (None, "") else _period(front["period"])
    if "published_on" in front:
        values["published_on"] = _date(front["published_on"])
    if "author" in front:
        values["author"] = "" if front["author"] is None else str(front["author"]).strip()
    if "section" in front:
        values["section"] = "" if front["section"] in (None, "") else _section(front["section"])
    if "entity" in front:
        values["entity"] = "" if front["entity"] in (None, "") else _entity(front["entity"])
    if "figure" in front:
        values["figure"] = "" if front["figure"] in (None, "") else _figure(front["figure"])
    if "occasion" in front:
        values["occasion"] = "" if front["occasion"] in (None, "") else _occasion(front["occasion"])


def _align_year_and_period(values: dict) -> None:
    period = values["period"]
    year = int(values["year"] or 0)
    if not period:
        values["year"] = year
        return
    period_year = int(period[:4])
    if year and year != period_year:
        raise ValueError(f"year={year} 与 period={period} 不是同一年")
    values["year"] = year or period_year


def _row_text(row, key: str) -> str:
    if key not in row.keys():
        return ""
    return row[key] or ""


def _query_includes_filing(scope: str, doc_type: str) -> bool:
    """未指定类型时，公司范围和未限定范围都会查到财报。"""
    if doc_type == "filing":
        return True
    if doc_type:
        return False
    return scope != "knowledge"


def _check_section_entity(values: dict) -> None:
    if values["entity"] and values["section"] not in _ENTITY_SECTIONS:
        raise ValueError("entity 只用于 section 为 financials 或 notes")


def _check_figure_occasion(values: dict) -> None:
    figure = values["figure"]
    occasion = values["occasion"]
    if values["doc_type"] == "investment_theory":
        if values["author"]:
            raise ValueError("investment_theory 用 figure 表示人物，不要写 author")
        if not figure:
            raise ValueError("investment_theory 需要 figure")
        if not occasion:
            raise ValueError("investment_theory 需要 occasion")
        return
    if figure or occasion:
        raise ValueError("figure 和 occasion 只用于 investment_theory")


def _figure(value) -> str:
    text = str(value).strip()
    if text not in FIGURE_LABELS:
        names = "、".join(FIGURES)
        raise ValueError(f"figure 只能是 {names}")
    return text


def _occasion(value) -> str:
    text = str(value).strip()
    if text not in OCCASION_LABELS:
        names = "、".join(OCCASIONS)
        raise ValueError(f"occasion 只能是 {names}")
    return text


def _section(value) -> str:
    text = str(value).strip()
    if text not in SECTION_LABELS:
        names = "、".join(SECTIONS)
        raise ValueError(f"section 只能是 {names}")
    return text


def _entity(value) -> str:
    text = str(value).strip()
    if text not in ENTITY_LABELS:
        raise ValueError("entity 只能是 consolidated 或 parent")
    return text


def _scope_for(doc_type: str) -> str:
    for scope, types in TYPES_BY_SCOPE.items():
        if doc_type in types:
            return scope
    raise ValueError(f"不认识的 doc_type: {doc_type}")


def _year(value) -> int:
    if isinstance(value, bool):
        raise ValueError("year 需要是年份整数")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise ValueError("year 需要是年份整数")
    if number != 0 and not 1900 <= number <= 2100:
        raise ValueError("year 需要在 1900 到 2100 之间")
    return number


def _code(value) -> str:
    if isinstance(value, bool):
        raise ValueError(f"股票代码需要是 6 位数字: {value}")
    if isinstance(value, int):
        text = f"{value:06d}"
    else:
        text = str(value).strip().upper()
        for suffix in _CODE_SUFFIX:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError(f"股票代码需要是 6 位数字: {value}")
    return text


def _period(value) -> str:
    text = str(value).strip().upper()
    if not _PERIOD_VALUE.fullmatch(text):
        raise ValueError("period 需要是 2024、2024Q1 或 2024H1 这种形式")
    return text


def _date(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        return value.date().strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("published_on 需要是 YYYY-MM-DD") from exc
    return text
