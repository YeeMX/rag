"""单元测试：规则抽取字段、日期/数字规范化、与 GT 的比较。

运行（在 03-vision-ocr-demo 目录下）::

    python tests/test_extract.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract import (  # noqa: E402
    compare_value,
    extract_fields,
    normalize_date,
    normalize_int,
)

# 模拟 PaddleOCR 区域识别结果（结构同 outputs/ocr_texts.json 的 region_pipeline 字段）
SAMPLE_REGIONS = [
    {"class_name": "title", "rank": 0, "conf": 0.99, "texts": ["劳动合同书"]},
    {
        "class_name": "paragraph",
        "rank": 1,
        "conf": 0.97,
        "texts": [
            "甲方（用人单位）：北京华信科技有限公司",
            "住　　所：北京市朝阳区建国路88号院3号楼",
            "法定代表人：张建国",
        ],
    },
    {
        "class_name": "paragraph",
        "rank": 2,
        "conf": 0.96,
        "texts": [
            "乙方（劳动者）：张伟",
            "身份证号：110101199001011234",
            "住　　所：上海市浦东新区张江路168号2幢",
        ],
    },
    {
        "class_name": "paragraph",
        "rank": 3,
        "conf": 0.95,
        "texts": ["第一条 本合同为固定期限劳动合同，合同期限为3年，自2024年03月15日起至2027年03月14日止。"],
    },
    {
        "class_name": "paragraph",
        "rank": 4,
        "conf": 0.94,
        "texts": ["第二条 乙方同意在甲方从事财务会计岗位工作，工作地点为北京市。"],
    },
    {
        "class_name": "paragraph",
        "rank": 5,
        "conf": 0.93,
        "texts": ["第三条 甲方于每月15日前支付乙方工资，月工资为人民币8000元。"],
    },
    {
        "class_name": "paragraph",
        "rank": 6,
        "conf": 0.92,
        "texts": ["第四条 乙方试用期为2个月，试用期月工资为人民币6400元。"],
    },
    {"class_name": "table", "rank": 7, "conf": 0.90, "texts": ["项目", "金额（元/月）", "备注", "合计", "8000", "税前"]},
    {
        "class_name": "paragraph",
        "rank": 8,
        "conf": 0.89,
        "texts": ["甲方（盖章）：北京华信科技有限公司", "乙方（签字）：张伟", "签订日期：2024年03月01日"],
    },
    {"class_name": "stamp", "rank": 9, "conf": 0.80, "texts": ["华信科技", "合同专用章"]},
]


class TestExtractFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fields = extract_fields(SAMPLE_REGIONS)

    def test_party_a(self) -> None:
        self.assertEqual(self.fields["party_a"]["value"], "北京华信科技有限公司")

    def test_party_b(self) -> None:
        self.assertEqual(self.fields["party_b"]["value"], "张伟")

    def test_contract_years(self) -> None:
        self.assertEqual(self.fields["contract_years"]["value"], 3)

    def test_contract_dates(self) -> None:
        self.assertEqual(self.fields["contract_start"]["value"], "2024-03-15")
        self.assertEqual(self.fields["contract_end"]["value"], "2027-03-14")

    def test_monthly_salary_not_confused_with_probation(self) -> None:
        """月工资不能被「试用期月工资」抢走。"""
        self.assertEqual(self.fields["monthly_salary"]["value"], 8000)

    def test_probation(self) -> None:
        self.assertEqual(self.fields["probation_months"]["value"], 2)
        self.assertEqual(self.fields["probation_salary"]["value"], 6400)

    def test_job(self) -> None:
        self.assertEqual(self.fields["job"]["value"], "财务会计")

    def test_sign_date(self) -> None:
        self.assertEqual(self.fields["sign_date"]["value"], "2024-03-01")

    def test_stamp_company(self) -> None:
        self.assertEqual(self.fields["stamp_company"]["value"], "华信科技")

    def test_document_type(self) -> None:
        self.assertEqual(self.fields["document_type"]["value"], "劳动合同书")

    def test_source_provenance(self) -> None:
        """每个抽到的字段都要能追溯到来源区域（可解释性）。"""
        self.assertEqual(self.fields["party_a"]["source"]["class_name"], "paragraph")
        self.assertIsNotNone(self.fields["stamp_company"]["source"])


class TestCrossLineFields(unittest.TestCase):
    """OCR 把长条款折成多行时，抽取仍要能跨行匹配到字段。"""

    def test_contract_dates_across_lines(self) -> None:
        regions = [
            {
                "class_name": "paragraph",
                "rank": 0,
                "conf": 0.9,
                "texts": [
                    "第一条 本合同为固定期限劳动合同，合同期限为3年，自2024年03月15日起至2027年03",
                    "月14日止。",
                ],
            }
        ]
        fields = extract_fields(regions)
        self.assertEqual(fields["contract_start"]["value"], "2024-03-15")
        self.assertEqual(fields["contract_end"]["value"], "2027-03-14")

    def test_sign_date(self) -> None:
        regions = [
            {
                "class_name": "paragraph",
                "rank": 0,
                "conf": 0.9,
                "texts": ["甲方（盖章）：某公司", "签订日期：2024年03月01日"],
            }
        ]
        fields = extract_fields(regions)
        self.assertEqual(fields["sign_date"]["value"], "2024-03-01")


class TestMissingFields(unittest.TestCase):
    def test_empty_regions(self) -> None:
        fields = extract_fields([])
        self.assertIsNone(fields["party_a"]["value"])
        self.assertIsNone(fields["monthly_salary"]["value"])
        self.assertIsNone(fields["document_type"]["value"])

    def test_no_date(self) -> None:
        fields = extract_fields([{"class_name": "paragraph", "rank": 0, "conf": 0.9, "texts": ["这里没有日期"]}])
        self.assertIsNone(fields["contract_start"]["value"])
        self.assertIsNone(fields["sign_date"]["value"])


class TestNormalization(unittest.TestCase):
    def test_normalize_date(self) -> None:
        self.assertEqual(normalize_date("2024年3月5日"), "2024-03-05")
        self.assertEqual(normalize_date("2024 年 03 月 15 日"), "2024-03-15")
        self.assertEqual(normalize_date("2024-03-15"), "2024-03-15")  # 已是规范格式也要能比较
        self.assertEqual(normalize_date("2024/3/5"), "2024-03-05")
        self.assertIsNone(normalize_date("这不是日期"))

    def test_normalize_int(self) -> None:
        self.assertEqual(normalize_int("18,000"), 18000)
        self.assertEqual(normalize_int(6400), 6400)
        self.assertIsNone(normalize_int(None))

    def test_compare_value(self) -> None:
        self.assertTrue(compare_value("monthly_salary", "8000", 8000))
        self.assertTrue(compare_value("sign_date", "2024年3月1日", "2024年03月01日"))
        self.assertFalse(compare_value("contract_years", 2, 3))
        self.assertFalse(compare_value("party_a", None, "北京华信科技有限公司"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
