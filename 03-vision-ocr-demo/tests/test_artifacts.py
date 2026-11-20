"""产物一致性测试：README 引用的图片存在、metrics JSON 结构合理、README 数字与 JSON 一致。

运行（在 03-vision-ocr-demo 目录下，需先跑过 run_demo.py）::

    python tests/test_artifacts.py

没有 outputs 产物时自动跳过（fresh clone 里先跑 `run_demo.py --skip-train`）。
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [
            OUT_DIR / "metrics_yolo.json",
            OUT_DIR / "metrics_ocr.json",
            OUT_DIR / "metrics_extract.json",
            PROJECT_ROOT / "README.md",
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise unittest.SkipTest("缺少运行产物，先执行 run_demo.py: " + ", ".join(missing))
        cls.yolo = read_json(OUT_DIR / "metrics_yolo.json")
        cls.ocr = read_json(OUT_DIR / "metrics_ocr.json")
        cls.extract = read_json(OUT_DIR / "metrics_extract.json")
        cls.readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    # -- 数据结构 -----------------------------------------------------------

    def test_yolo_metrics_structure(self) -> None:
        official = self.yolo["ultralytics_val"]
        manual = self.yolo["manual_iou_eval"]
        for value in (official["mAP50"], official["mAP50_95"], manual["mAP50"], manual["mAP50_95"]):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        # 手写评估与官方评估应当接近（同一批 val 预测）
        self.assertLess(abs(official["mAP50"] - manual["mAP50"]), 0.05)
        self.assertLess(abs(official["mAP50_95"] - manual["mAP50_95"]), 0.05)
        speed = self.yolo["speed"]
        self.assertGreater(speed["median_ms"], 0)
        self.assertGreater(speed["runs"], 0)

    def test_ocr_metrics_structure(self) -> None:
        config = self.ocr["config"]
        self.assertEqual(len(self.ocr["per_image"]), config["num_images"])
        for item in self.ocr["per_image"]:
            self.assertGreaterEqual(item["full_image"]["cer"], 0.0)
            self.assertGreaterEqual(item["region_pipeline"]["cer"], 0.0)
            self.assertGreater(item["full_image"]["time_ms"], 0)
        # 总体 CER 必须等于 总编辑距离 / 总字符数
        full = self.ocr["full_image"]
        self.assertAlmostEqual(
            full["cer_overall"],
            round(full["total_edit_distance"] / full["total_gt_chars"], 4),
            places=4,
        )

    def test_extract_metrics_structure(self) -> None:
        micro = self.extract["headline_micro"]
        self.assertAlmostEqual(
            self.extract["headline_field_accuracy"],
            round(micro["correct"] / micro["total"], 4),
            places=4,
        )

    # -- README 一致性 ------------------------------------------------------

    def test_readme_image_links_exist(self) -> None:
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", self.readme)
        self.assertGreaterEqual(len(links), 1, "README 里应引用真实输出图片")
        for link in links:
            self.assertTrue((PROJECT_ROOT / link).is_file(), f"README 引用的图片不存在: {link}")

    def test_readme_numbers_match_metrics(self) -> None:
        """README 里的关键数字必须能在 metrics JSON 中找到（避免手写指标漂移）。"""
        tokens = {
            "mAP50": f"{self.yolo['ultralytics_val']['mAP50']:.4f}",
            "mAP50_95": f"{self.yolo['ultralytics_val']['mAP50_95']:.4f}",
            "ocr_full_cer": f"{self.ocr['full_image']['cer_overall']:.4f}",
            "ocr_region_cer": f"{self.ocr['region_pipeline']['cer_overall']:.4f}",
            "extract_accuracy": f"{self.extract['headline_field_accuracy']:.4f}",
        }
        for name, token in tokens.items():
            self.assertIn(token, self.readme, f"README 缺少 {name}={token}，或数字与 metrics JSON 不一致")

    def test_data_yaml_not_empty(self) -> None:
        path = DATA_DIR / "data.yaml"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("names:", text)
        self.assertIn("train:", text)
        self.assertIn("val:", text)
        self.assertGreater(path.stat().st_size, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
