"""YOLO 版面区域检测：训练、推理、真实指标计算、CPU 速度测量。

用法（在 03-vision-ocr-demo 目录下，必须用 .venv-vision 解释器）::

    # 训练 + 评估 + 推理出图（完整）
    python src/yolo_detect.py --stage all --data-dir data --out outputs --epochs 40

    # 已有 models/best.pt，跳过训练，只做评估 + 推理
    python src/yolo_detect.py --stage all --skip-train

指标说明（面试可讲）：
- ``ultralytics_val``：调用 ``model.val()`` 得到的官方 mAP@0.5 / mAP@0.5:0.95 / P / R；
- ``manual_iou_eval``：**自己手写**的 IoU 匹配评估（贪心匹配 + TP/FP/FN + 全点插值 AP），
  用来交叉验证官方指标，同时把「TP/FP/FN 怎么定」讲清楚；
- ``speed``：CPU 上单张图片端到端推理耗时（多次取中位数）与 FPS。

所有数字均由本脚本实际运行产生，写入 ``outputs/metrics_yolo.json``。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

CLASS_NAMES: list[str] = ["title", "paragraph", "table", "stamp"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def load_annotations(data_dir: Path) -> dict[str, Any]:
    """读取 make_dataset.py 生成的 annotations.json。"""
    path = data_dir / "annotations.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到标注文件 {path}，请先运行 scripts/make_dataset.py")
    return json.loads(path.read_text(encoding="utf-8"))


def images_of_split(annotations: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [img for img in annotations["images"] if img["split"] == split]


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """计算两个 xyxy 框的 IoU（intersection over union）。

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        IoU，范围 [0, 1]；不相交返回 0。
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_match_image(
    detections: Sequence[dict[str, Any]],
    ground_truths: Sequence[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int]]:
    """单张图的贪心匹配：按置信度从高到低，把检测框匹配到同类且 IoU 最高且未匹配的 GT。

    匹配成功（IoU >= 阈值）的检测算 TP，否则算 FP；没被匹配到的 GT 算 FN。

    Args:
        detections: [{"bbox": [x1,y1,x2,y2], "conf": float, "cls": int}, ...]
        ground_truths: [{"bbox": [x1,y1,x2,y2], "cls": int}, ...]
        iou_threshold: IoU 阈值（mAP@0.5 用 0.5）

    Returns:
        (matches, matched_gt_indices)。matches 为 [(det_index, gt_index), ...]，
        只包含匹配成功的检测；未匹配检测不在其中（即 FP）。
    """
    order = sorted(range(len(detections)), key=lambda i: -float(detections[i]["conf"]))
    matched: list[tuple[int, int]] = []
    matched_gt: set[int] = set()
    for det_idx in order:
        det = detections[det_idx]
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt or int(gt["cls"]) != int(det["cls"]):
                continue
            value = iou_xyxy(det["bbox"], gt["bbox"])
            if value > best_iou:
                best_iou = value
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            matched.append((det_idx, best_gt))
    return matched, matched_gt


def average_precision(tp_flags: Sequence[bool], num_gt: int) -> float:
    """根据按置信度降序排列的 TP/FP 序列，用「全点插值」计算 AP。

    参考 VOC/COCO 的积分方式：先把 precision 做单调递减修正，再对 recall 变化点求和。
    """
    if num_gt == 0 or not tp_flags:
        return 0.0
    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp = np.cumsum(1.0 - np.asarray(tp_flags, dtype=np.float64))
    recall = tp / float(num_gt)
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def resolve_pretrained(base_weights: str) -> tuple[str, bool]:
    """尽量使用预训练权重。

    顺序：本地文件 -> ultralytics 自动下载 -> 从零训练的 yaml（如实标记 pretrained=False）。
    """
    candidate = Path(base_weights)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_file():
        return str(candidate), True
    # 本地没有：尝试让 ultralytics 下载官方权重（需要网络）
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = attempt_download_asset(candidate.name)
        if downloaded and Path(downloaded).is_file():
            return str(downloaded), True
    except Exception as exc:  # noqa: BLE001 - 无网络 / 下载失败时降级
        print(f"[yolo] 预训练权重下载失败（{exc}），将使用 {candidate.name.replace('.pt', '.yaml')} 从零训练")
    return "yolo11n.yaml", False


def train_model(
    data_yaml: Path,
    *,
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int = 0,
    base_weights: str = "yolo11n.pt",
    project_dir: Path = PROJECT_ROOT / "runs",
    run_name: str = "doc_layout",
    workers: int = 2,
) -> dict[str, Any]:
    """训练 YOLO 小模型（CPU）。返回训练元信息。"""
    from ultralytics import YOLO

    weights, pretrained = resolve_pretrained(base_weights)
    print(f"[yolo] 训练配置: epochs={epochs} imgsz={imgsz} batch={batch} "
          f"base={weights} pretrained={pretrained}")
    model = YOLO(weights)
    t0 = time.perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=workers,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        seed=seed,
        deterministic=True,
        amp=False,          # CPU 没有 AMP
        cache="ram",        # 小数据集直接缓存，省 IO
        plots=False,
        val=True,
        verbose=True,
    )
    seconds = time.perf_counter() - t0
    best = project_dir / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"训练结束但找不到最优权重 {best}")
    # 拷贝到稳定路径，便于 --skip-train 使用
    stable = PROJECT_ROOT / "models" / "best.pt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(best.read_bytes())
    print(f"[yolo] 训练耗时 {seconds:.1f}s，权重已复制到 {stable}")
    return {
        "base_weights": weights,
        "pretrained": pretrained,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "train_seconds": round(seconds, 1),
        "best_weights": str(stable),
        "ultralytics_best": str(best),
    }


# ---------------------------------------------------------------------------
# 官方 model.val() 指标
# ---------------------------------------------------------------------------


def evaluate_with_ultralytics(
    weights: Path, data_yaml: Path, *, imgsz: int, split: str = "val"
) -> dict[str, Any]:
    """调用 ultralytics 的 model.val() 得到官方指标。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=8,
        device="cpu",
        plots=False,
        verbose=False,
        workers=0,
    )
    box = metrics.box
    names: dict[int, str] = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    per_class: dict[str, Any] = {}
    for idx, name in names.items():
        try:
            per_class[name] = {
                "ap50": round(float(box.ap50[idx]), 4),
                "ap50_95": round(float(box.ap[idx]), 4),
                "precision": round(float(box.p[idx]), 4),
                "recall": round(float(box.r[idx]), 4),
            }
        except Exception:  # noqa: BLE001 - 类别数组可能不存在
            continue
    speed = getattr(metrics, "speed", {}) or {}
    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(float(v), 3) for k, v in speed.items()},
    }


# ---------------------------------------------------------------------------
# 手写 IoU 匹配评估（交叉验证）
# ---------------------------------------------------------------------------


def _predict_detections(
    model: Any, image_path: Path, *, imgsz: int, conf: float, iou_nms: float
) -> list[dict[str, Any]]:
    """对单张图推理，返回 [{bbox, conf, cls}, ...]（原图坐标系）。"""
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou_nms,
        device="cpu",
        verbose=False,
        max_det=300,
    )
    dets: list[dict[str, Any]] = []
    if not results:
        return dets
    boxes = results[0].boxes
    if boxes is None:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for bbox, score, cls in zip(xyxy, confs, classes):
        dets.append(
            {
                "bbox": [float(v) for v in bbox],
                "conf": float(score),
                "cls": int(cls),
            }
        )
    return dets


def evaluate_with_iou_matching(
    weights: Path,
    annotations: dict[str, Any],
    data_dir: Path,
    *,
    split: str,
    imgsz: int,
    conf: float = 0.25,
    iou_nms: float = 0.7,
    iou_thresholds: Sequence[float] = (0.5,),
) -> dict[str, Any]:
    """手写评估：贪心 IoU 匹配 → TP/FP/FN → P/R/AP。

    Args:
        weights: best.pt 路径
        annotations: annotations.json 内容
        data_dir: 数据集根目录
        split: train / val
        imgsz: 推理尺寸
        conf: 计算 P/R 时使用的部署置信度阈值
        iou_nms: NMS 的 IoU 阈值（与 ultralytics 官方默认保持一致以便对比）
        iou_thresholds: 计算 mAP 时使用的 IoU 阈值列表

    Returns:
        指标字典（含每张图、每类别的 TP/FP/FN 与 AP）。
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = images_of_split(annotations, split)

    # 1) 逐图推理 + 贪心匹配（det_conf=0.001 保留全部检测用于算 AP）
    per_image: list[dict[str, Any]] = []
    # 每个 IoU 阈值下：per_class -> list[(conf, is_tp)]，以及每类 GT 数
    all_dets: dict[float, dict[str, list[tuple[float, bool]]]] = {
        float(t): {name: [] for name in CLASS_NAMES} for t in iou_thresholds
    }
    gt_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    # 部署阈值下的微平均
    deploy_tp = deploy_fp = deploy_fn = 0

    for img in images:
        img_path = data_dir / img["file"]
        dets = _predict_detections(model, img_path, imgsz=imgsz, conf=0.001, iou_nms=iou_nms)
        gts = [
            {"bbox": region["bbox"], "cls": CLASS_NAMES.index(region["class"])}
            for region in img["regions"]
        ]
        for gt in gts:
            gt_counts[CLASS_NAMES[gt["cls"]]] += 1

        per_threshold_result: dict[str, dict[str, int]] = {}
        for thr in iou_thresholds:
            thr = float(thr)
            matched, matched_gt = greedy_match_image(dets, gts, thr)
            tp = len(matched)
            fp = len(dets) - tp
            fn = len(gts) - tp
            per_threshold_result[str(thr)] = {"tp": tp, "fp": fp, "fn": fn}

            matched_det_indices = {d for d, _ in matched}
            for det_idx, det in enumerate(dets):
                class_name = CLASS_NAMES[det["cls"]]
                all_dets[thr][class_name].append(
                    (det["conf"], det_idx in matched_det_indices)
                )

            if thr == 0.5:
                # 部署阈值下的 TP/FP/FN（detections 先过滤 conf）
                deploy_dets = [d for d in dets if d["conf"] >= conf]
                deploy_matched, _ = greedy_match_image(deploy_dets, gts, thr)
                deploy_tp += len(deploy_matched)
                deploy_fp += len(deploy_dets) - len(deploy_matched)
                deploy_fn += len(gts) - len(deploy_matched)

        per_image.append(
            {
                "id": img["id"],
                "num_gt": len(gts),
                "num_det": len(dets),
                "matching": per_threshold_result,
            }
        )

    # 2) 每个 IoU 阈值、每个类别计算 AP
    ap_by_threshold: dict[str, dict[str, float]] = {}
    ap50_by_class: dict[str, float] = {}
    stats_by_class: dict[str, Any] = {}
    for thr in (float(t) for t in iou_thresholds):
        per_class_ap: dict[str, float] = {}
        for name in CLASS_NAMES:
            pairs = all_dets[thr][name]
            pairs.sort(key=lambda item: -item[0])  # 置信度降序
            tp_flags = [is_tp for _, is_tp in pairs]
            ap = average_precision(tp_flags, gt_counts[name])
            per_class_ap[name] = round(ap, 4)
            if thr == 0.5:
                ap50_by_class[name] = round(ap, 4)
                n_tp = sum(1 for flag in tp_flags if flag)
                n_det = len(tp_flags)
                stats_by_class[name] = {
                    "num_gt": gt_counts[name],
                    "num_det": n_det,
                    "tp": n_tp,
                    "fp": n_det - n_tp,
                    "fn": gt_counts[name] - n_tp,
                    "ap50": round(ap, 4),
                    "precision_at_all_det": round(n_tp / n_det, 4) if n_det else 0.0,
                    "recall_at_all_det": round(n_tp / gt_counts[name], 4) if gt_counts[name] else 0.0,
                }
        ap_by_threshold[str(thr)] = per_class_ap

    map50 = float(np.mean([ap50_by_class[name] for name in CLASS_NAMES]))
    map50_95 = float(
        np.mean(
            [
                np.mean([ap_by_threshold[str(thr)][name] for name in CLASS_NAMES])
                for thr in (float(t) for t in iou_thresholds)
            ]
        )
    )
    precision = deploy_tp / (deploy_tp + deploy_fp) if (deploy_tp + deploy_fp) else 0.0
    recall = deploy_tp / (deploy_tp + deploy_fn) if (deploy_tp + deploy_fn) else 0.0

    return {
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "ap50_per_class": ap50_by_class,
        "ap_per_threshold": ap_by_threshold,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "deployment_conf": conf,
        "precision_at_conf": round(precision, 4),
        "recall_at_conf": round(recall, 4),
        "totals_at_conf": {
            "tp": deploy_tp,
            "fp": deploy_fp,
            "fn": deploy_fn,
        },
        "per_class": stats_by_class,
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# CPU 速度测量
# ---------------------------------------------------------------------------


def benchmark_speed(
    weights: Path,
    sample_image: Path,
    *,
    imgsz: int,
    runs: int = 15,
    warmup: int = 3,
) -> dict[str, Any]:
    """测量 CPU 单张图端到端推理耗时（预处理 + 推理 + 后处理）。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for _ in range(warmup):
        model.predict(source=str(sample_image), imgsz=imgsz, device="cpu", verbose=False)

    times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=str(sample_image), imgsz=imgsz, device="cpu", verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    median = statistics.median(times_ms)
    mean = statistics.fmean(times_ms)
    sorted_ms = sorted(times_ms)
    p90 = sorted_ms[min(len(sorted_ms) - 1, int(round(0.9 * (len(sorted_ms) - 1))))]
    return {
        "runs": runs,
        "warmup": warmup,
        "imgsz": imgsz,
        "median_ms": round(median, 2),
        "mean_ms": round(mean, 2),
        "p90_ms": round(p90, 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "fps_from_median": round(1000.0 / median, 2) if median > 0 else 0.0,
        "device": "cpu",
        "note": "端到端耗时 = 预处理 + 前向 + NMS/后处理；单进程单张，不含批量优势。",
    }


# ---------------------------------------------------------------------------
# 推理出图 + 供 OCR 流水线使用的预测结果
# ---------------------------------------------------------------------------


def _read_image_bgr(path: Path) -> np.ndarray:
    """用 PIL 读图并转成 BGR ndarray（避免额外依赖 cv2 的 imread 编码问题）。"""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _write_image_bgr(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr[:, :, ::-1]).save(path)


def save_predictions(
    weights: Path,
    annotations: dict[str, Any],
    data_dir: Path,
    out_dir: Path,
    *,
    split: str,
    imgsz: int,
    conf: float = 0.25,
    max_visuals: int = 8,
) -> dict[str, Any]:
    """跑推理，保存带框图片到 outputs/images/，并写出 yolo_predictions.json。

    OCR 流水线会读取这里的预测框做「区域裁剪 → 逐区域识别」。
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images = images_of_split(annotations, split)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for idx, img in enumerate(images):
        img_path = data_dir / img["file"]
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            device="cpu",
            verbose=False,
        )
        result = results[0]
        dets: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for bbox, score, cls in zip(xyxy, confs, classes):
                dets.append(
                    {
                        "class_id": int(cls),
                        "class_name": CLASS_NAMES[int(cls)],
                        "conf": round(float(score), 4),
                        "bbox": [round(float(v), 1) for v in bbox],
                    }
                )
        records.append(
            {
                "id": img["id"],
                "image": img["file"],
                "width": img["width"],
                "height": img["height"],
                "hard_case": img["hard_case"],
                "detections": dets,
                "visual": f"images/det_{img['id']}.png" if idx < max_visuals else None,
            }
        )
        if idx < max_visuals:
            plotted = result.plot()  # BGR ndarray，已画框 + 类别 + 置信度
            _write_image_bgr(out_images / f"det_{img['id']}.png", plotted)

    payload = {
        "config": {
            "weights": str(weights),
            "imgsz": imgsz,
            "conf": conf,
            "split": split,
            "device": "cpu",
        },
        "images": records,
    }
    (out_dir / "yolo_predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def make_gt_pred_visual(
    weights: Path,
    annotations: dict[str, Any],
    data_dir: Path,
    out_dir: Path,
    *,
    img_id: str,
    imgsz: int,
) -> Path:
    """生成「GT（绿）vs 预测（红）」对比图，方便 README 展示。"""
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    model = YOLO(str(weights))
    img_meta = next(img for img in annotations["images"] if img["id"] == img_id)
    img_path = data_dir / img_meta["file"]
    results = model.predict(source=str(img_path), imgsz=imgsz, conf=0.25, device="cpu", verbose=False)
    dets = []
    boxes = results[0].boxes
    if boxes is not None:
        for bbox, score, cls in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
        ):
            dets.append(([float(v) for v in bbox], float(score), int(cls)))

    with Image.open(img_path) as im:
        canvas = im.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for region in img_meta["regions"]:
        x1, y1, x2, y2 = region["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 170, 0), width=3)
    for bbox, score, cls in dets:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=3)
        draw.text((x1 + 4, max(0, y1 + 2)), f"{CLASS_NAMES[cls]} {score:.2f}", fill=(220, 30, 30))
    out = out_dir / "images" / f"gt_vs_pred_{img_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ---------------------------------------------------------------------------
# 环境信息
# ---------------------------------------------------------------------------


def environment_info() -> dict[str, Any]:
    """收集可复现的环境信息（硬件 / 版本）。"""
    import torch
    import ultralytics

    try:
        from ultralytics.utils.torch_utils import get_cpu_info

        cpu = get_cpu_info()
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 版面检测：训练/评估/推理")
    parser.add_argument("--stage", choices=["train", "eval", "predict", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--base-weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25, help="部署置信度阈值")
    parser.add_argument("--split", default="val")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--speed-runs", type=int, default=15)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = data_dir / "data.yaml"
    annotations = load_annotations(data_dir)
    weights: Path = args.weights

    train_info: dict[str, Any] | None = None
    if args.stage in ("train", "all") and not args.skip_train:
        train_info = train_model(
            data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            base_weights=args.base_weights,
        )
        weights = Path(train_info["best_weights"])
    if not weights.is_file():
        raise FileNotFoundError(
            f"找不到权重 {weights}。请先训练（去掉 --skip-train）或用 --weights 指定。"
        )

    if args.stage == "train":
        return 0

    if args.stage in ("eval", "all"):
        print("[yolo] 1/3 调用 ultralytics model.val() 计算官方指标 ...")
        official = evaluate_with_ultralytics(weights, data_yaml, imgsz=args.imgsz, split=args.split)
        print(f"        official: mAP50={official['mAP50']:.4f} "
              f"mAP50-95={official['mAP50_95']:.4f} P={official['precision']:.4f} "
              f"R={official['recall']:.4f}")

        print("[yolo] 2/3 手写 IoU 匹配评估（交叉验证）...")
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
        manual = evaluate_with_iou_matching(
            weights,
            annotations,
            data_dir,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_nms=0.7,
            iou_thresholds=iou_thresholds,
        )
        print(f"        manual  : mAP50={manual['mAP50']:.4f} "
              f"mAP50-95={manual['mAP50_95']:.4f} "
              f"P@{args.conf}={manual['precision_at_conf']:.4f} "
              f"R@{args.conf}={manual['recall_at_conf']:.4f}")

        sample = images_of_split(annotations, args.split)[0]
        print("[yolo] 3/3 CPU 速度测量 ...")
        speed = benchmark_speed(
            weights,
            data_dir / sample["file"],
            imgsz=args.imgsz,
            runs=args.speed_runs,
        )
        print(f"        median={speed['median_ms']:.1f}ms FPS={speed['fps_from_median']:.1f}")

        cross = {
            "mAP50_abs_diff": round(abs(official["mAP50"] - manual["mAP50"]), 4),
            "mAP50_95_abs_diff": round(abs(official["mAP50_95"] - manual["mAP50_95"]), 4),
            "note": "手动实现与 ultralytics 在 NMS 后处理/letterbox 细节上略有差异，数值接近即为一致。",
        }
        metrics = {
            "config": {
                "weights": str(weights),
                "data_yaml": str(data_yaml),
                "split": args.split,
                "num_eval_images": len(images_of_split(annotations, args.split)),
                "num_regions": sum(len(img["regions"]) for img in images_of_split(annotations, args.split)),
                "imgsz": args.imgsz,
                "conf_deployment": args.conf,
                "nms_iou": 0.7,
                "device": "cpu",
                "train": train_info,
            },
            "environment": environment_info(),
            "ultralytics_val": official,
            "manual_iou_eval": manual,
            "cross_check": cross,
            "speed": speed,
        }
        metrics_path = out_dir / "metrics_yolo.json"
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[yolo] 指标已写入 {metrics_path}")

    if args.stage in ("predict", "all"):
        print("[yolo] 保存推理结果与可视化 ...")
        payload = save_predictions(
            weights,
            annotations,
            data_dir,
            out_dir,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
        )
        # 额外生成一张「GT vs 预测」对比图（优先选正常样本）
        normal = [img for img in images_of_split(annotations, args.split) if not img["hard_case"]]
        visual_id = (normal or images_of_split(annotations, args.split))[0]["id"]
        gt_pred = make_gt_pred_visual(
            weights, annotations, data_dir, out_dir, img_id=visual_id, imgsz=args.imgsz
        )
        print(f"[yolo] 预测结果: {out_dir / 'yolo_predictions.json'}（{len(payload['images'])} 张）")
        print(f"[yolo] 对比图: {gt_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
