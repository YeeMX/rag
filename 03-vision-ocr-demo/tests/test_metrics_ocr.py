"""单元测试：Levenshtein / CER / 文本规范化（OCR 指标的核心逻辑）。

运行（在 03-vision-ocr-demo 目录下，paddle 或 vision 环境均可，不加载模型）::

    python tests/test_metrics_ocr.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_paddle import (  # noqa: E402
    character_error_rate,
    levenshtein_distance,
    line_exact_match_rate,
    normalize_text,
)


class TestLevenshtein(unittest.TestCase):
    def test_equal_strings(self) -> None:
        self.assertEqual(levenshtein_distance("劳动合同", "劳动合同"), 0)

    def test_empty_strings(self) -> None:
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_single_substitution(self) -> None:
        self.assertEqual(levenshtein_distance("劳动合同", "劳动合用"), 1)

    def test_insertion(self) -> None:
        self.assertEqual(levenshtein_distance("工资", "工资标准"), 2)

    def test_deletion(self) -> None:
        self.assertEqual(levenshtein_distance("劳动合同书", "劳动合同"), 1)

    def test_classic_kitten(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)

    def test_symmetric(self) -> None:
        a, b = "甲方用人单位", "甲万用人单位"
        self.assertEqual(levenshtein_distance(a, b), levenshtein_distance(b, a))

    def test_common_prefix_suffix_optimization_keeps_result(self) -> None:
        # 长公共前缀 + 长公共后缀，内部有 2 处替换
        a = "这是一段很长的公共前缀文本ABC很长的公共后缀文本"
        b = "这是一段很长的公共前缀文本AXC很长的公共后缀文本"
        self.assertEqual(levenshtein_distance(a, b), 1)


class TestNormalizeAndCer(unittest.TestCase):
    def test_normalize_removes_whitespace(self) -> None:
        self.assertEqual(normalize_text("劳动 合同\n第一条\u3000 期限"), "劳动合同第一条期限")

    def test_cer_zero_when_identical(self) -> None:
        self.assertAlmostEqual(character_error_rate("劳动合同书", "劳动合同书"), 0.0)

    def test_cer_ignores_whitespace(self) -> None:
        self.assertAlmostEqual(character_error_rate("劳动合同书", "劳动 合同\n书"), 0.0)

    def test_cer_one_when_all_wrong(self) -> None:
        self.assertAlmostEqual(character_error_rate("甲乙丙丁", "戊己庚辛"), 1.0)

    def test_cer_single_char_error(self) -> None:
        # 4 个字错 1 个 -> CER = 0.25
        self.assertAlmostEqual(character_error_rate("甲乙丙丁", "甲乙丙戊"), 0.25)

    def test_cer_can_exceed_one(self) -> None:
        self.assertGreater(character_error_rate("短", "很长很长的识别结果"), 1.0)

    def test_cer_with_empty_reference(self) -> None:
        self.assertAlmostEqual(character_error_rate("", ""), 0.0)
        self.assertAlmostEqual(character_error_rate("", "多出来的字"), 1.0)


class TestLineExactMatch(unittest.TestCase):
    def test_all_lines_hit(self) -> None:
        gt = "劳动合同书\n第一条 合同期限为三年"
        hits, total = line_exact_match_rate(gt, ["劳动合同书", "第一条 合同期限为三年"])
        self.assertEqual((hits, total), (2, 2))

    def test_partial_lines(self) -> None:
        gt = "劳动合同书\n第一条 合同期限为三年\n第二条 月工资"
        hits, total = line_exact_match_rate(gt, ["劳动合同书", "第一条 合同期限为3年"])
        self.assertEqual((hits, total), (1, 3))

    def test_whitespace_insensitive(self) -> None:
        hits, total = line_exact_match_rate("甲方 用人单位", ["甲方用人单位"])
        self.assertEqual((hits, total), (1, 1))

    def test_empty_reference(self) -> None:
        hits, total = line_exact_match_rate("", [])
        self.assertEqual((hits, total), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
