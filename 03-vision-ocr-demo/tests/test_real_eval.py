"""单元测试：真实数据评测脚本的纯函数（GT 文本顺序、类别无关覆盖率计算）。

运行（在 03-vision-ocr-demo 目录下）::

    python tests/test_real_eval.py

不加载 paddle / ultralytics 模型；没有 XFUND 数据时跳过数据集相关用例。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_real_dataset as ev  # noqa: E402


class TestGtTextOrder(unittest.TestCase):
    def test_reading_order(self) -> None:
        entities = [
            {"box": [300, 100, 500, 120], "text": "第二行"},
            {"box": [100, 10, 200, 30], "text": "第一行"},
            {"box": [10, 100, 50, 120], "text": "第三行左"},
        ]
        text = ev.build_gt_text(entities)
        self.assertEqual(text, "第一行\n第三行左\n第二行")

    def test_skip_empty(self) -> None:
        entities = [{"box": [0, 0, 10, 10], "text": "  "}, {"box": [0, 20, 10, 30], "text": "有字"}]
        self.assertEqual(ev.build_gt_text(entities), "有字")


class TestCoverageMetrics(unittest.TestCase):
    def test_perfect_match(self) -> None:
        gts = [[0, 0, 10, 10], [100, 100, 110, 110]]
        dets = [
            {"bbox": [0, 0, 10, 10], "conf": 0.9, "class_id": 0},
            {"bbox": [100, 100, 110, 110], "conf": 0.8, "class_id": 1},
        ]
        m = ev.gt_coverage_metrics(gts, dets)
        self.assertEqual(m["recall_iou50"], 1.0)
        self.assertEqual(m["center_hit_rate"], 1.0)
        self.assertEqual(m["center_hits_per_gt"], 1.0)  # 每个 GT 中心恰好落在 1 个框内
        self.assertEqual(m["det_gt_ratio"], 1.0)
        self.assertEqual(m["mean_gt_area_coverage"], 1.0)
        self.assertAlmostEqual(m["mean_conf_all_dets"], 0.85, places=4)

    def test_no_detections(self) -> None:
        m = ev.gt_coverage_metrics([[0, 0, 10, 10]], [])
        self.assertEqual(m["recall_iou50"], 0.0)
        self.assertEqual(m["center_hit_rate"], 0.0)
        self.assertEqual(m["mean_conf_all_dets"], None)
        self.assertEqual(m["det_gt_ratio"], 0.0)

    def test_partial_iou(self) -> None:
        # GT 框 10x10，检测框覆盖一半 -> IoU=0.5 恰好命中；中心点命中
        m = ev.gt_coverage_metrics([[0, 0, 10, 10]], [{"bbox": [0, 0, 5, 10], "conf": 0.7, "class_id": 0}])
        self.assertEqual(m["recall_iou50"], 1.0)
        self.assertEqual(m["center_hit_rate"], 1.0)
        self.assertAlmostEqual(m["mean_gt_area_coverage"], 0.5, places=4)

    def test_center_counted_once(self) -> None:
        """多个检测框都盖住 GT 中心时，中心命中率不能超过 1。"""
        gts = [[0, 0, 100, 100]]
        dets = [
            {"bbox": [0, 0, 100, 100], "conf": 0.9, "class_id": 0},
            {"bbox": [10, 10, 90, 90], "conf": 0.8, "class_id": 1},
            {"bbox": [20, 20, 80, 80], "conf": 0.7, "class_id": 2},
        ]
        m = ev.gt_coverage_metrics(gts, dets)
        self.assertEqual(m["center_hit_rate"], 1.0)
        self.assertEqual(m["center_hits_per_gt"], 3.0)  # 3 个框都盖住同一个 GT 中心 -> 可 >1
        self.assertEqual(m["recall_iou50"], 1.0)

    def test_large_block_vs_small_gt(self) -> None:
        """合成模型学到的是版面块，GT 是行级框：IOU 很低但中心/覆盖率高（域差距的核心）。"""
        gt = [[0, 0, 100, 20]]  # 一行文字
        det = [{"bbox": [0, 0, 400, 200], "conf": 0.6, "class_id": 1}]  # 一个大段落块
        m = ev.gt_coverage_metrics(gt, det)
        self.assertEqual(m["recall_iou50"], 0.0)       # 粒度不匹配，IoU 命中失败
        self.assertEqual(m["center_hit_rate"], 1.0)    # 但中心点在里面
        self.assertEqual(m["center_hits_per_gt"], 1.0)
        self.assertAlmostEqual(m["mean_gt_area_coverage"], 1.0, places=4)  # GT 面积被完全覆盖


class TestLoadPages(unittest.TestCase):
    def test_load_pages_if_dataset_available(self) -> None:
        data_dir = PROJECT_ROOT / "data" / "real"
        annotations = data_dir / "zh.val.json"
        images_dir = data_dir / "xfund_zh_val"
        if not annotations.is_file() or not images_dir.is_dir():
            raise unittest.SkipTest("未下载 XFUND 数据（python scripts/eval_real_dataset.py --download）")
        pages = ev.load_pages(annotations, images_dir, max_pages=3)
        self.assertEqual(len(pages), 3)
        self.assertTrue(all(page["gt_text"].strip() for page in pages))
        self.assertTrue(all(len(page["gt_boxes"]) >= 10 for page in pages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
