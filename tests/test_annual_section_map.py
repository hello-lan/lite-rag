"""年报书签 → section / entity 映射。"""

import unittest
from pathlib import Path

from scripts.split_annual_reports import (
    SectionMapError,
    Segment,
    map_entity,
    map_section,
    require_sections,
)


class AnnualSectionMapTests(unittest.TestCase):
    def test_standard_chapters(self) -> None:
        cases = {
            ("第一节 重要提示、目录和释义",): "profile",
            ("第二节 公司简介和主要财务指标", "一、公司信息"): "profile",
            ("第二节 公司概况",): "profile",
            ("第三节 会计数据和财务指标",): "profile",
            ("第三节 管理层讨论与分析", "四、主营业务分析"): "mda",
            ("董事会报告",): "mda",
            ("致股东",): "mda",
            ("第四节 公司治理",): "governance",
            ("第七节 股份变动及股东情况",): "governance",
            ("第八节 优先股相关情况",): "governance",
            ("第五节 环境和社会责任",): "material",
            ("第六节 重要事项",): "material",
            ("第五节 重大事件",): "material",
            ("第九节 债券相关情况",): "material",
            ("第十节 财务报告",): "notes",
            ("第十一节 财务会计报告",): "notes",
        }
        for path, want in cases.items():
            self.assertEqual(map_section(path), want, path)

    def test_combined_governance_esg_splits_by_subtitle(self) -> None:
        root = "第四节 公司治理、环境和社会"
        self.assertEqual(map_section((root,)), "governance")
        self.assertEqual(map_section((root, "一、公司治理的基本状况")), "governance")
        self.assertEqual(map_section((root, "十五、环境信息披露情况")), "material")
        self.assertEqual(map_section((root, "十六、社会责任情况")), "material")
        self.assertEqual(
            map_section((root, "十七、巩固拓展脱贫攻坚成果、乡村振兴的情况")),
            "material",
        )

    def test_financial_statements_vs_notes(self) -> None:
        root = "第十节 财务报告"
        self.assertEqual(map_section((root, "一、审计报告")), "financials")
        self.assertEqual(map_section((root, "二、财务报表")), "financials")
        self.assertEqual(
            map_section((root, "二、财务报表", "1、合并资产负债表")),
            "financials",
        )
        self.assertEqual(
            map_section((root, "二、财务报表", "5、母公司利润表")),
            "financials",
        )
        self.assertEqual(
            map_section((root, "七、合并财务报表项目注释")),
            "notes",
        )
        self.assertEqual(
            map_section((root, "十七、资产负债表日后事项")),
            "notes",
        )
        self.assertEqual(
            map_section((root, "五、重要会计政策及会计估计")),
            "notes",
        )
        # 附注里带「现金流量表/资产负债表」字样，不能误判成主表。
        self.assertEqual(
            map_section((root, "七、合并财务报表项目注释", "53、现金流量表项目")),
            "notes",
        )
        self.assertEqual(
            map_section((root, "四、财务报表的编制基础")),
            "notes",
        )
        self.assertEqual(
            map_section((root, "十五、承诺及或有事项", "（1）资产负债表日存在的重要或有事项")),
            "notes",
        )

    def test_orphan_note_bookmarks_without_financial_parent(self) -> None:
        """美的等 TOC 把附注抬成一级时，仍应标成 notes / financials。"""
        self.assertEqual(map_section(("一 公司基本情况",)), "notes")
        self.assertEqual(map_section(("二 主要会计政策和会计估计",)), "notes")
        self.assertEqual(map_section(("四 合并财务报表项目附注",)), "notes")
        self.assertEqual(map_section(("九 关联方及重大关联交易",)), "notes")
        self.assertEqual(map_section(("十三 资产负债表日后事项",)), "notes")
        self.assertEqual(map_section(("三 境内外会计准则下会计数据差异",)), "notes")
        self.assertEqual(map_section(("一、审计报告",)), "financials")
        self.assertEqual(map_section(("1、合并资产负债表",)), "financials")
        self.assertEqual(map_section(("合并资产负债表",)), "financials")
        self.assertEqual(map_section(("四、其他",)), "other")
        # 仅当去编号后恰为「其他」；「其他说明」等仍不映射。
        self.assertEqual(map_section(("五、其他说明",)), "")
        self.assertEqual(
            map_section(("第十节 财务报告", "十八、其他重要事项", "4、其他")),
            "notes",
        )

    def test_entity(self) -> None:
        root = "第十节 财务报告"
        consolidated = (root, "二、财务报表", "1、合并资产负债表")
        parent = (root, "二、财务报表", "5、母公司利润表")
        notes_parent = (root, "十九、母公司财务报表主要项目注释")
        notes_parent_alias = ("十八 公司财务报表主要项目附注",)
        self.assertEqual(map_entity(consolidated, "financials"), "consolidated")
        self.assertEqual(map_entity(parent, "financials"), "parent")
        self.assertEqual(map_entity(notes_parent, "notes"), "parent")
        self.assertEqual(map_entity(notes_parent_alias, "notes"), "parent")
        self.assertEqual(map_entity(("第三节 管理层讨论与分析",), "mda"), "")

    def test_require_sections_fails_when_unmapped(self) -> None:
        ok = Segment(path=("第三节 管理层讨论与分析",), start_page=0, end_page=1, text="x")
        bad = Segment(path=("未知章节",), start_page=10, end_page=11, text="y")
        require_sections([ok], pdf=Path("demo.pdf"))
        require_sections(
            [Segment(path=("四、其他",), start_page=10, end_page=11, text="y")],
            pdf=Path("demo.pdf"),
        )
        with self.assertRaises(SectionMapError) as caught:
            require_sections([ok, bad], pdf=Path("demo.pdf"))
        message = str(caught.exception)
        self.assertIn("demo.pdf", message)
        self.assertIn("未知章节", message)
        self.assertIn("p11-12", message)
        self.assertIn("map_section", message)


if __name__ == "__main__":
    unittest.main()
