"""单元测试：合成数据集的自洽性（标注 bbox / YOLO 标签 / 全文 / 字段 GT）。

运行（在 03-vision-ocr-demo 目录下，需先生成数据）::

    python scripts/make_dataset.py
    python tests/test_dataset.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
CLASS_NAMES = ["title", "paragraph", "table", "stamp"]


def reading_order_key(region: dict) -> tuple[float, float]:
    return (round(float(region["bbox"][1]), 1), round(float(region["bbox"][0]), 1))


class TestDatasetAnnotations(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = DATA_DIR / "annotations.json"
        if not path.is_file():
            raise unittest.SkipTest(f"找不到 {path}，请先运行 scripts/make_dataset.py")
        cls.annotations = json.loads(path.read_text(encoding="utf-8"))
        cls.images = cls.annotations["images"]

    def test_image_count_and_splits(self) -> None:
        self.assertGreaterEqual(len(self.images), 24)
        train = [img for img in self.images if img["split"] == "train"]
        val = [img for img in self.images if img["split"] == "val"]
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertEqual(len(train) + len(val), len(self.images))

    def test_image_files_exist(self) -> None:
        for img in self.images:
            self.assertTrue((DATA_DIR / img["file"]).is_file(), img["file"])

    def test_bboxes_inside_image(self) -> None:
        for img in self.images:
            w, h = img["width"], img["height"]
            for region in img["regions"]:
                x1, y1, x2, y2 = region["bbox"]
                self.assertTrue(0 <= x1 < x2 <= w, f"{img['id']} {region}")
                self.assertTrue(0 <= y1 < y2 <= h, f"{img['id']} {region}")
                self.assertIn(region["class"], CLASS_NAMES)
                self.assertTrue(region["text"].strip(), f"{img['id']} 空文本区域")

    def test_full_text_matches_reading_order(self) -> None:
        """整图全文必须等于按阅读顺序拼接的区域文本（OCR CER 的 GT 依赖这一点）。"""
        for img in self.images:
            ordered = sorted(img["regions"], key=reading_order_key)
            expected = "\n".join(region["text"] for region in ordered)
            self.assertEqual(img["full_text"], expected, img["id"])

    def test_yolo_labels_consistent(self) -> None:
        """YOLO 标签的类别计数应与 annotations 中「尺寸 >= 2px」的区域计数一致。"""
        for img in self.images:
            label_path = DATA_DIR / img["label"]
            self.assertTrue(label_path.is_file(), str(label_path))
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            expected = sum(
                1
                for region in img["regions"]
                if region["bbox"][2] - region["bbox"][0] >= 2 and region["bbox"][3] - region["bbox"][1] >= 2
            )
            self.assertEqual(len(lines), expected, img["id"])
            for line in lines:
                parts = line.split()
                self.assertEqual(len(parts), 5, line)
                cls_id = int(parts[0])
                self.assertIn(cls_id, range(len(CLASS_NAMES)))
                values = [float(v) for v in parts[1:]]
                for value in values:
                    self.assertGreaterEqual(value, 0.0, line)
                    self.assertLessEqual(value, 1.0, line)

    def test_hard_cases_exist(self) -> None:
        hard = [img for img in self.images if img["hard_case"]]
        self.assertGreaterEqual(len(hard), 2)
        types = {img["hard_case"] for img in hard}
        for expected in ("rotation", "low_contrast", "stamp_occlusion"):
            self.assertIn(expected, types)
        rotated = [img for img in hard if img["hard_case"] == "rotation"]
        for img in rotated:
            self.assertGreaterEqual(abs(img["rotation_deg"]), 2.0)
            self.assertLessEqual(abs(img["rotation_deg"]), 5.0)

    def test_fields_ground_truth(self) -> None:
        required = [
            "party_a",
            "party_b",
            "contract_years",
            "contract_start",
            "contract_end",
            "monthly_salary",
            "probation_months",
            "job",
            "sign_date",
        ]
        for img in self.images:
            for name in required:
                self.assertIn(name, img["fields"], img["id"])
                self.assertIsNotNone(img["fields"][name], f"{img['id']} {name}")
            self.assertGreaterEqual(int(img["fields"]["monthly_salary"]), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
