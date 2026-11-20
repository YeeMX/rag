"""单元测试：IoU 计算、贪心匹配、AP 计算（手写指标的核心逻辑）。

运行（在 03-vision-ocr-demo 目录下，vision 环境）::

    python tests/test_metrics_yolo.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.yolo_detect import average_precision, greedy_match_image, iou_xyxy  # noqa: E402


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1.0, places=6)

    def test_disjoint_boxes(self) -> None:
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0, places=6)

    def test_half_overlap(self) -> None:
        # 两个 10x10 框水平重叠一半：交集 50，并集 150
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150, places=6)

    def test_contained_box(self) -> None:
        # 小框完全在大框内：交集 25，并集 100
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [2, 2, 7, 7]), 0.25, places=6)

    def test_touching_boxes_is_zero(self) -> None:
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [10, 0, 20, 10]), 0.0, places=6)


class TestGreedyMatching(unittest.TestCase):
    def test_perfect_match(self) -> None:
        dets = [{"bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]
        gts = [{"bbox": [0, 0, 10, 10], "cls": 0}]
        matched, matched_gt = greedy_match_image(dets, gts, 0.5)
        self.assertEqual(matched, [(0, 0)])
        self.assertEqual(matched_gt, {0})

    def test_class_mismatch_is_fp(self) -> None:
        dets = [{"bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 1}]
        gts = [{"bbox": [0, 0, 10, 10], "cls": 0}]
        matched, _ = greedy_match_image(dets, gts, 0.5)
        self.assertEqual(matched, [])

    def test_low_iou_is_fp(self) -> None:
        dets = [{"bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]
        gts = [{"bbox": [0, 0, 6, 10], "cls": 0}]  # IoU = 60/100 = 0.6 -> 0.5 阈值算 TP
        matched, _ = greedy_match_image(dets, gts, 0.7)
        self.assertEqual(matched, [])
        matched, _ = greedy_match_image(dets, gts, 0.5)
        self.assertEqual(len(matched), 1)

    def test_one_gt_two_dets(self) -> None:
        """一个 GT 只能匹配一个检测框，另一个检测框算 FP。"""
        dets = [
            {"bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0},
            {"bbox": [1, 1, 11, 11], "conf": 0.8, "cls": 0},
        ]
        gts = [{"bbox": [0, 0, 10, 10], "cls": 0}]
        matched, matched_gt = greedy_match_image(dets, gts, 0.5)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][0], 0)  # 高置信度的那个被匹配
        self.assertEqual(len(matched_gt), 1)

    def test_conf_order_matters(self) -> None:
        """按置信度从高到低匹配：低置信度但 IoU 更高的框不会抢走 GT。"""
        dets = [
            {"bbox": [0, 0, 10, 10], "conf": 0.5, "cls": 0},   # IoU=1.0 但置信度低
            {"bbox": [0, 0, 9, 10], "conf": 0.95, "cls": 0},  # IoU=0.9 先匹配
        ]
        gts = [{"bbox": [0, 0, 10, 10], "cls": 0}]
        matched, _ = greedy_match_image(dets, gts, 0.5)
        self.assertEqual(matched, [(1, 0)])


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ap(self) -> None:
        self.assertAlmostEqual(average_precision([True, True], 2), 1.0, places=6)

    def test_half_recall(self) -> None:
        # 2 个 GT，只召回 1 个且无 FP：召回率 0.5 -> AP = 0.5
        self.assertAlmostEqual(average_precision([True], 2), 0.5, places=6)

    def test_fp_before_tp(self) -> None:
        # 第一个是 FP，第二个是 TP；2 个 GT
        # recall=0.5, precision=0.5 -> AP=0.25
        self.assertAlmostEqual(average_precision([False, True], 2), 0.25, places=6)

    def test_no_detections(self) -> None:
        self.assertAlmostEqual(average_precision([], 3), 0.0, places=6)

    def test_no_gt(self) -> None:
        self.assertAlmostEqual(average_precision([True], 0), 0.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
