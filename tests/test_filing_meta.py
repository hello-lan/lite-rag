"""财报向量元信息。不加载 embedding 模型。"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from lite_rag.meta import SearchFilter, doc_meta_from_fields
from lite_rag.store import CHUNK_TABLES, chroma_meta, init_meta_db, required_columns


def filing(**extra):
    fields = {
        "scope": "financial_report",
        "doc_type": "filing",
        "stock_name": "贵州茅台",
        "stock_code": "600519",
        "year": 2024,
        "period": "2024",
    }
    fields.update(extra)
    return doc_meta_from_fields(fields)


class FilingMetaTests(unittest.TestCase):
    def test_filing_chroma_uses_period_section_and_entity(self) -> None:
        doc = filing(section="financials", entity="consolidated")
        meta = chroma_meta("doc-1", doc)
        self.assertEqual(meta["period"], "2024")
        self.assertEqual(meta["section"], "financials")
        self.assertEqual(meta["entity"], "consolidated")
        self.assertEqual(meta["stock_name"], "贵州茅台")
        self.assertNotIn("year", meta)

    def test_empty_section_and_entity_are_omitted(self) -> None:
        meta = chroma_meta("doc-1", filing())
        self.assertEqual(meta["period"], "2024")
        self.assertNotIn("section", meta)
        self.assertNotIn("entity", meta)
        self.assertNotIn("year", meta)

    def test_theory_chroma_keeps_figure_occasion_and_year(self) -> None:
        doc = doc_meta_from_fields(
            {
                "scope": "knowledge",
                "doc_type": "investment_theory",
                "figure": "唐朝",
                "occasion": "article",
                "year": 2018,
            }
        )
        meta = chroma_meta("doc-2", doc)
        self.assertEqual(meta["doc_type"], "investment_theory")
        self.assertEqual(meta["figure"], "唐朝")
        self.assertEqual(meta["occasion"], "article")
        self.assertEqual(meta["year"], 2018)
        self.assertNotIn("period", meta)
        self.assertNotIn("section", meta)

    def test_investment_theory_requires_figure_and_occasion(self) -> None:
        with self.assertRaises(ValueError):
            doc_meta_from_fields({"scope": "knowledge", "doc_type": "investment_theory", "year": 2018})
        with self.assertRaises(ValueError):
            doc_meta_from_fields(
                {
                    "scope": "knowledge",
                    "doc_type": "investment_theory",
                    "figure": "唐朝",
                    "year": 2018,
                }
            )

    def test_section_must_be_known(self) -> None:
        with self.assertRaises(ValueError):
            filing(section="balance")

    def test_narrative_sections_cannot_carry_entity(self) -> None:
        with self.assertRaises(ValueError):
            filing(section="profile", entity="consolidated")
        with self.assertRaises(ValueError):
            filing(section="mda", entity="parent")

    def test_entity_requires_financials_or_notes(self) -> None:
        with self.assertRaises(ValueError):
            filing(entity="consolidated")
        doc = filing(section="notes", entity="parent")
        self.assertEqual(doc.entity, "parent")
        self.assertEqual(chroma_meta("doc-3", doc)["entity"], "parent")

    def test_filing_query_rejects_year(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            SearchFilter.parse(scope="financial_report", doc_type="filing", year=2024)
        self.assertEqual(caught.exception.code, "查财报请用 period，不要用 year")
        with self.assertRaises(SystemExit):
            SearchFilter.parse(scope="financial_report", year=2024)
        with self.assertRaises(SystemExit):
            SearchFilter.parse(year=2024)

    def test_knowledge_query_keeps_year(self) -> None:
        filt = SearchFilter.parse(
            scope="knowledge",
            doc_type="investment_theory",
            figure="buffett",
            occasion="meeting",
            year=2018,
        )
        self.assertEqual(filt.year, 2018)
        self.assertEqual(
            filt.chroma_where(),
            {
                "$and": [
                    {"doc_type": "investment_theory"},
                    {"figure": "buffett"},
                    {"occasion": "meeting"},
                    {"year": 2018},
                ]
            },
        )

    def test_investor_materials_chroma_uses_doc_type_stock_and_year(self) -> None:
        doc = doc_meta_from_fields(
            {
                "scope": "investor_materials",
                "doc_type": "ir_activities",
                "title": "美的集团投资者关系活动记录表",
                "stock_name": "美的集团",
                "stock_code": "000333",
                "year": 2024,
            }
        )
        meta = chroma_meta("doc-ir", doc)
        self.assertEqual(meta["doc_type"], "ir_activities")
        self.assertEqual(meta["stock_name"], "美的集团")
        self.assertEqual(meta["stock_code"], "000333")
        self.assertEqual(meta["year"], 2024)
        self.assertNotIn("period", meta)
        self.assertNotIn("published_on", meta)

    def test_investor_materials_query_keeps_year(self) -> None:
        filt = SearchFilter.parse(
            scope="investor_materials",
            doc_type="research",
            stock_code="002027",
            year=2025,
        )
        self.assertEqual(filt.year, 2025)
        self.assertEqual(
            filt.chroma_where(),
            {
                "$and": [
                    {"doc_type": "research"},
                    {"stock_code": "002027"},
                    {"year": 2025},
                ]
            },
        )

    def test_investor_materials_rejects_filing_filters(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            SearchFilter.parse(scope="investor_materials", period="2024")
        self.assertEqual(caught.exception.code, "investor_materials 不支持 period、section、entity")

    def test_filing_filter_uses_period_section_and_entity(self) -> None:
        with self.assertRaises(SystemExit):
            SearchFilter.parse(doc_type="filing", section="balance")
        filt = SearchFilter.parse(
            doc_type="filing",
            period="2024Q3",
            section="notes",
            entity="parent",
        )
        self.assertEqual(
            filt.chroma_where(),
            {"$and": [{"period": "2024Q3"}, {"section": "notes"}, {"entity": "parent"}]},
        )

    def test_incomplete_schema_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.db"
            conn = sqlite3.connect(path)
            conn.execute(
                """
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    source_doc_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    split_config TEXT NOT NULL,
                    embed_model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scope TEXT NOT NULL DEFAULT '',
                    doc_type TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    stock_name TEXT NOT NULL DEFAULT '',
                    stock_code TEXT NOT NULL DEFAULT '',
                    year INTEGER NOT NULL DEFAULT 0,
                    period TEXT NOT NULL DEFAULT '',
                    published_on TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                INSERT INTO chunks (
                    source_doc_id, source_name, source_path, chunk_text, chunk_idx,
                    split_config, embed_model_name, year, period
                ) VALUES ('a', 'n', 'p', 't', 0, 's', 'm', 2024, '2024')
                """
            )
            conn.commit()
            conn.close()

            reset = init_meta_db(path)
            self.assertTrue(reset)
            conn = sqlite3.connect(path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertEqual(set(CHUNK_TABLES.values()), tables & set(CHUNK_TABLES.values()))
            self.assertNotIn("chunks", tables)
            for scope, table in CHUNK_TABLES.items():
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                self.assertEqual(count, 0)
                self.assertEqual(columns, required_columns(scope))
            company_cols = required_columns("financial_report")
            investor_cols = required_columns("investor_materials")
            knowledge_cols = required_columns("knowledge")
            self.assertTrue({"stock_name", "stock_code", "period", "section", "entity"} <= company_cols)
            self.assertFalse({"scope", "year", "figure", "occasion"} & company_cols)
            self.assertTrue({"stock_name", "stock_code", "year"} <= investor_cols)
            self.assertFalse(
                {"scope", "period", "published_on", "section", "entity", "figure", "occasion"}
                & investor_cols
            )
            self.assertTrue({"figure", "occasion", "year"} <= knowledge_cols)
            self.assertFalse({"scope", "stock_name", "stock_code", "period", "section", "entity"} & knowledge_cols)
            conn.close()
            self.assertFalse(init_meta_db(path))


if __name__ == "__main__":
    unittest.main()
