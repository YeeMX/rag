"""PaddleOCR 3.x 法律文书 OCR：整图识别 vs YOLO 区域裁剪后识别，并计算真实 CER。

用法（在 03-vision-ocr-demo 目录下，必须用 .venv-paddle 解释器）::

    python src/ocr_paddle.py --data-dir data --out outputs --split val

做什么：
1. 对 val 集的每张图做**整图 OCR**（PaddleOCR 直接识别整页）；
2. 读取 YOLO 的检测结果（``outputs/yolo_predictions.json``），把每个版面区域裁剪出来，
   逐个区域 OCR，再按阅读顺序拼接 —— 这是本 demo 的核心流水线；
3. 两条路线分别计算 CER（自己实现的 Levenshtein 编辑距离）、整句准确率、单图平均耗时；
4. 结果写入 ``outputs/metrics_ocr.json`` 与 ``outputs/ocr_texts.json``。

为什么分区域识别可能更好（面试可讲）：
- 整页 OCR 时，检测阶段会把整页缩放到固定边长，小字号正文被缩小 → 识别率下降；
- 按区域裁剪后，每个区域按自身尺度识别，小字不会因为整页缩放而丢细节；
- 同时区域类别还能给下游结构化抽取提供「这段文字是什么」的先验。
代价：需要先跑 YOLO 检测，且多次调用 OCR 会增加总耗时。

环境说明：本机 Paddle 3.3.1 + PaddleOCR 3.7.0 在 CPU 上存在 oneDNN/PIR 兼容性
bug（``ConvertPirAttribute2RuntimeAttribute not support``），因此初始化时关闭
MKLDNN（``enable_mkldnn=False``）。所有数字都在该设置下测得，metrics 里已注明。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

CLASS_NAMES: list[str] = ["title", "paragraph", "table", "stamp"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 默认使用 PP-OCRv5 mobile（CPU 上的速度/精度折中）；
#: 想换更准更慢的模型可传 --det-model PP-OCRv6_medium_det --rec-model PP-OCRv6_medium_rec。
DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_REC_MODEL = "PP-OCRv5_mobile_rec"

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


# ---------------------------------------------------------------------------
# 文本规范化与 CER（自己实现，不依赖第三方）
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """CER 计算前的规范化：去掉所有空白（含换行、全角空格）。

    中文 OCR 评测一般忽略空白排版差异，只比较可见字符；标点**保留**并参与计分。
    """
    return _WHITESPACE_RE.sub("", text.replace("\u3000", " "))


def _strip_common_prefix_suffix(a: str, b: str) -> tuple[str, str]:
    """去掉公共前后缀，缩小编辑距离 DP 的规模（不改变编辑距离）。"""
    start = 0
    limit = min(len(a), len(b))
    while start < limit and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    return a[start:end_a], b[start:end_b]


def levenshtein_distance(a: str, b: str) -> int:
    """计算两个字符串的编辑距离（插入 / 删除 / 替换，代价均为 1）。

    经典动态规划，时间复杂度 O(len(a) * len(b))，空间 O(min(len(a), len(b)))。

    Args:
        a: 参考文本
        b: 识别文本

    Returns:
        最少编辑次数，等于插入 + 删除 + 替换的字符数。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    a, b = _strip_common_prefix_suffix(a, b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):  # 让 b 是较短的那个，内存更省
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j] + 1,      # 删除 a[i-1]
                    current[j - 1] + 1,   # 插入 b[j-1]
                    previous[j - 1] + cost,  # 替换 / 匹配
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """字符错误率 CER = 编辑距离 / 参考文本字符数。

    参考文本为空时：识别结果也为空记 0，否则记 1。CER 可以 > 1（识别结果比原文长很多）。
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


def line_exact_match_rate(reference_text: str, hypothesis_lines: Sequence[str]) -> tuple[int, int]:
    """整句（行级）准确率：GT 的每一非空行，是否在 OCR 结果中被一字不差地识别出来。

    Returns:
        (命中行数, GT 非空行数)
    """
    gt_lines = [normalize_text(line) for line in reference_text.split("\n")]
    gt_lines = [line for line in gt_lines if line]
    hyp_set = {normalize_text(line) for line in hypothesis_lines}
    hyp_set.discard("")
    hits = sum(1 for line in gt_lines if line in hyp_set)
    return hits, len(gt_lines)


# ---------------------------------------------------------------------------
# PaddleOCR 封装
# ---------------------------------------------------------------------------


def init_ocr(
    *,
    det_model: str = DEFAULT_DET_MODEL,
    rec_model: str = DEFAULT_REC_MODEL,
    enable_mkldnn: bool = False,
) -> Any:
    """初始化 PaddleOCR（3.x API）。

    关闭 doc orientation / unwarping / textline orientation 三个子模块：
    本场景输入是正向平整的扫描件，这些模块只会增加耗时。
    """
    import paddle

    if not enable_mkldnn:
        # 绕过 Paddle 3.3.1 + PaddleOCR 3.7 CPU oneDNN/PIR 兼容性 bug
        paddle.set_flags({"FLAGS_use_mkldnn": False})

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=enable_mkldnn,
    )
    return ocr


def _polygon_to_bbox(poly: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def ocr_one_image(ocr: Any, image_path: Path) -> dict[str, Any]:
    """对单张图片跑一次 OCR，返回文本、置信度、文本框与耗时。"""
    t0 = time.perf_counter()
    results = ocr.predict(str(image_path))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    texts: list[str] = []
    scores: list[float] = []
    boxes: list[list[float]] = []
    if results:
        result = results[0]
        texts = [str(t) for t in result["rec_texts"]]
        scores = [round(float(s), 4) for s in result["rec_scores"]]
        polys = result["rec_polys"]
        boxes = [_polygon_to_bbox(poly) for poly in polys]
    return {
        "texts": texts,
        "scores": scores,
        "boxes": boxes,
        "time_ms": round(elapsed_ms, 2),
    }


def sort_detections(detections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按阅读顺序排序 YOLO 检测框：先上沿、再左沿（与数据集标注的排序规则一致）。"""
    return sorted(
        detections,
        key=lambda det: (round(float(det["bbox"][1]), 1), round(float(det["bbox"][0]), 1)),
    )


def crop_region(
    image_path: Path,
    bbox: Sequence[float],
    out_path: Path,
    *,
    pad: int = 8,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """裁剪一个区域（带 padding，裁到图片边界内），保存 PNG，返回尺寸与裁剪框。"""
    from PIL import Image

    with Image.open(image_path) as im:
        img = im.convert("RGB")
        width, height = img.size
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(width, x2 + pad)
        cy2 = min(height, y2 + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError(f"非法裁剪框 bbox={bbox} 图片={image_path}")
        crop = img.crop((cx1, cy1, cx2, cy2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return crop.width, crop.height, (cx1, cy1, cx2, cy2)


# ---------------------------------------------------------------------------
# 两条 OCR 路线
# ---------------------------------------------------------------------------


def run_full_image_pipeline(
    ocr: Any, image_meta: dict[str, Any], data_dir: Path
) -> dict[str, Any]:
    """整图直接 OCR。"""
    image_path = data_dir / image_meta["file"]
    result = ocr_one_image(ocr, image_path)
    return {
        "text": "\n".join(result["texts"]),
        "lines": result["texts"],
        "scores": result["scores"],
        "boxes": result["boxes"],
        "time_ms": result["time_ms"],
        "num_lines": len(result["texts"]),
    }


def run_region_pipeline(
    ocr: Any,
    image_meta: dict[str, Any],
    predictions: dict[str, Any],
    data_dir: Path,
    crops_dir: Path,
) -> dict[str, Any]:
    """YOLO 区域裁剪 → 逐区域 OCR → 阅读顺序拼接。"""
    detections = predictions["detections"]
    ordered = sort_detections(detections)
    image_path = data_dir / image_meta["file"]
    regions: list[dict[str, Any]] = []
    all_lines: list[str] = []
    total_ms = 0.0
    for rank, det in enumerate(ordered):
        class_name = det["class_name"]
        crop_path = crops_dir / image_meta["id"] / f"{rank:02d}_{class_name}.png"
        width, height, crop_box = crop_region(image_path, det["bbox"], crop_path)
        result = ocr_one_image(ocr, crop_path)
        total_ms += result["time_ms"]
        all_lines.extend(result["texts"])
        regions.append(
            {
                "rank": rank,
                "class_name": class_name,
                "conf": det["conf"],
                "bbox": det["bbox"],
                "crop_box": list(crop_box),
                "crop_size": [width, height],
                "crop": crop_path.relative_to(PROJECT_ROOT).as_posix(),
                "texts": result["texts"],
                "scores": result["scores"],
                "time_ms": result["time_ms"],
            }
        )
    return {
        "text": "\n".join(all_lines),
        "lines": all_lines,
        "regions": regions,
        "num_regions": len(ordered),
        "ocr_time_ms": round(total_ms, 2),
    }


# ---------------------------------------------------------------------------
# 可视化：把一个区域的裁剪结果拼成图（README 展示用）
# ---------------------------------------------------------------------------


def save_crop_montage(image_meta: dict[str, Any], crops: Sequence[dict[str, Any]], out_path: Path) -> Path | None:
    """把一张图的各区域裁剪竖向拼接，左侧标注类别 / 置信度。"""
    from PIL import Image, ImageDraw, ImageFont

    if not crops:
        return None
    font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
    label_w = 250
    pad = 10
    loaded: list[tuple[Image.Image, str]] = []
    for region in crops:
        path = PROJECT_ROOT / region["crop"]
        try:
            with Image.open(path) as im:
                loaded.append((im.convert("RGB"), f"{region['class_name']}  conf={region['conf']:.2f}"))
        except FileNotFoundError:
            continue
    if not loaded:
        return None
    width = max(im.width for im, _ in loaded) + label_w + pad * 2
    height = sum(im.height for im, _ in loaded) + pad * (len(loaded) + 1)
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    y = pad
    for im, label in loaded:
        draw.text((pad, y + im.height // 2 - 8), label, font=font, fill=(30, 30, 30))
        canvas.paste(im, (label_w + pad, y))
        y += im.height + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 主评估流程
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def environment_info() -> dict[str, Any]:
    import paddle
    import paddleocr
    import platform

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "paddle": paddle.__version__,
        "paddleocr": paddleocr.__version__,
        "device": "cpu",
        "mkldnn": False,
        "note": "mkldnn=False 是为绕过 Paddle 3.3.1 + PaddleOCR 3.7 的 oneDNN/PIR bug。",
    }


def evaluate(
    *,
    data_dir: Path,
    out_dir: Path,
    predictions_path: Path,
    metrics_yolo_path: Path,
    split: str,
    det_model: str,
    rec_model: str,
    enable_mkldnn: bool,
    max_images: int | None = None,
    save_crops: bool = True,
) -> dict[str, Any]:
    """跑两条 OCR 路线并汇总指标。"""
    # 统一成绝对路径：crop 记录里要用相对项目根目录的路径
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    predictions_path = predictions_path.resolve()
    metrics_yolo_path = metrics_yolo_path.resolve()
    annotations = load_json(data_dir / "annotations.json")
    predictions = load_json(predictions_path)
    images = [img for img in annotations["images"] if img["split"] == split]
    if max_images:
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"split={split} 没有图片")

    # 每个区域裁剪后用于区域路线的 OCR 耗时，需要把检测耗时单独拿出来看
    detection_ms: float | None = None
    if metrics_yolo_path.is_file():
        try:
            detection_ms = float(load_json(metrics_yolo_path)["speed"]["median_ms"])
        except Exception:  # noqa: BLE001
            detection_ms = None

    print(f"[ocr] 初始化 PaddleOCR: det={det_model} rec={rec_model} mkldnn={enable_mkldnn}")
    t_init = time.perf_counter()
    ocr = init_ocr(det_model=det_model, rec_model=rec_model, enable_mkldnn=enable_mkldnn)
    init_seconds = time.perf_counter() - t_init
    print(f"[ocr] 初始化耗时 {init_seconds:.1f}s")

    crops_dir = out_dir / "crops"
    pred_by_id = {item["id"]: item for item in predictions["images"]}

    # ---- 预热：模型第一次 predict 会做图优化 / 内存分配，不计入耗时 ----
    first_img = images[0]
    warmup_path = data_dir / first_img["file"]
    _ = ocr_one_image(ocr, warmup_path)
    first_dets = pred_by_id.get(first_img["id"], {}).get("detections", [])
    if first_dets:
        warmup_crop = crops_dir / "_warmup" / "crop.png"
        try:
            crop_region(warmup_path, first_dets[0]["bbox"], warmup_crop)
            _ = ocr_one_image(ocr, warmup_crop)
        except ValueError:
            pass
    print("[ocr] 预热完成，开始正式测量 ...")

    per_image: list[dict[str, Any]] = []
    full_cer_sum = 0.0
    region_cer_sum = 0.0
    full_ref_chars = 0
    region_ref_chars = 0
    full_edit_distance = 0
    region_edit_distance = 0
    full_hits = 0
    region_hits = 0
    gt_line_total = 0
    full_exact = 0
    region_exact = 0
    full_times: list[float] = []
    region_ocr_times: list[float] = []
    region_counts: list[int] = []

    texts_payload: list[dict[str, Any]] = []
    montage_saved = False

    for img in images:
        gt_text = img["full_text"]
        image_path = data_dir / img["file"]

        # 路线 A：整图 OCR
        full = run_full_image_pipeline(ocr, img, data_dir)
        full_norm = normalize_text(full["text"])
        gt_norm = normalize_text(gt_text)
        ref_chars = len(gt_norm)
        dist_full = levenshtein_distance(gt_norm, full_norm)
        cer_full = dist_full / ref_chars if ref_chars else 0.0
        hits_full, n_gt_lines = line_exact_match_rate(gt_text, full["lines"])

        # 路线 B：区域裁剪流水线
        pred = pred_by_id.get(img["id"])
        if pred is None:
            region = {"text": "", "lines": [], "regions": [], "num_regions": 0, "ocr_time_ms": 0.0}
        else:
            region = run_region_pipeline(ocr, img, pred, data_dir, crops_dir)
        region_norm = normalize_text(region["text"])
        dist_region = levenshtein_distance(gt_norm, region_norm)
        cer_region = dist_region / ref_chars if ref_chars else 0.0
        hits_region, _ = line_exact_match_rate(gt_text, region["lines"])

        full_cer_sum += cer_full
        region_cer_sum += cer_region
        full_ref_chars += ref_chars
        region_ref_chars += ref_chars
        full_edit_distance += dist_full
        region_edit_distance += dist_region
        full_hits += hits_full
        region_hits += hits_region
        gt_line_total += n_gt_lines
        full_exact += 1 if full_norm == gt_norm else 0
        region_exact += 1 if region_norm == gt_norm else 0
        full_times.append(full["time_ms"])
        region_ocr_times.append(region["ocr_time_ms"])
        region_counts.append(region["num_regions"])

        item = {
            "id": img["id"],
            "hard_case": img["hard_case"],
            "gt_chars": ref_chars,
            "gt_lines": n_gt_lines,
            "full_image": {
                "cer": round(cer_full, 4),
                "edit_distance": dist_full,
                "num_lines": full["num_lines"],
                "time_ms": full["time_ms"],
                "line_hits": hits_full,
            },
            "region_pipeline": {
                "cer": round(cer_region, 4),
                "edit_distance": dist_region,
                "num_regions": region["num_regions"],
                "num_lines": len(region["lines"]),
                "ocr_time_ms": region["ocr_time_ms"],
                "line_hits": hits_region,
            },
        }
        per_image.append(item)
        texts_payload.append(
            {
                "id": img["id"],
                "hard_case": img["hard_case"],
                "gt_text": gt_text,
                "full_image_text": full["text"],
                "region_pipeline_text": region["text"],
                "region_pipeline": region["regions"],
            }
        )

        if save_crops and not montage_saved and not img["hard_case"] and region["regions"]:
            montage = save_crop_montage(
                img, region["regions"], out_dir / "images" / "region_crops_example.png"
            )
            montage_saved = montage is not None
        print(
            f"[ocr] {img['id']} ({img['hard_case'] or 'normal'}): "
            f"整图 CER={cer_full:.3f} ({full['time_ms']:.0f}ms) | "
            f"区域 CER={cer_region:.3f} ({region['ocr_time_ms']:.0f}ms, {region['num_regions']} 区域)"
        )

    n = len(per_image)
    full_time_median = statistics.median(full_times)
    region_time_median = statistics.median(region_ocr_times)
    full_ms = statistics.fmean(full_times)
    region_ms = statistics.fmean(region_ocr_times)
    region_end_to_end = region_ms + (detection_ms or 0.0)

    metrics: dict[str, Any] = {
        "config": {
            "split": split,
            "num_images": n,
            "det_model": det_model,
            "rec_model": rec_model,
            "lang": "ch",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "enable_mkldnn": enable_mkldnn,
            "init_seconds": round(init_seconds, 2),
            "crop_padding_px": 8,
            "device": "cpu",
            "warmup": "第一张图 + 第一个裁剪区域各跑一次，不计入耗时。",
        },
        "full_image": {
            "cer_overall": round(full_edit_distance / full_ref_chars, 4) if full_ref_chars else 0.0,
            "cer_mean_per_image": round(full_cer_sum / n, 4),
            "total_edit_distance": full_edit_distance,
            "total_gt_chars": full_ref_chars,
            "sentence_accuracy": round(full_hits / gt_line_total, 4) if gt_line_total else 0.0,
            "sentence_hits": full_hits,
            "sentence_total": gt_line_total,
            "image_exact_match": round(full_exact / n, 4),
            "avg_time_ms": round(full_ms, 2),
            "median_time_ms": round(full_time_median, 2),
            "fps_from_median": round(1000.0 / full_time_median, 2) if full_time_median else 0.0,
        },
        "region_pipeline": {
            "cer_overall": round(region_edit_distance / region_ref_chars, 4) if region_ref_chars else 0.0,
            "cer_mean_per_image": round(region_cer_sum / n, 4),
            "total_edit_distance": region_edit_distance,
            "total_gt_chars": region_ref_chars,
            "sentence_accuracy": round(region_hits / gt_line_total, 4) if gt_line_total else 0.0,
            "sentence_hits": region_hits,
            "sentence_total": gt_line_total,
            "image_exact_match": round(region_exact / n, 4),
            "avg_regions_per_image": round(statistics.fmean(region_counts), 2) if region_counts else 0.0,
            "avg_ocr_time_ms": round(region_ms, 2),
            "median_ocr_time_ms": round(region_time_median, 2),
            "avg_detection_ms": round(detection_ms, 2) if detection_ms is not None else None,
            "avg_end_to_end_ms": round(region_end_to_end, 2),
            "end_to_end_fps": round(1000.0 / region_end_to_end, 2) if region_end_to_end else 0.0,
            "note": "end_to_end = YOLO 检测中位耗时（来自 metrics_yolo.json）+ 各区域 OCR 耗时之和。",
        },
        "comparison": {},
        "per_image": per_image,
        "environment": environment_info(),
    }

    full = metrics["full_image"]
    region = metrics["region_pipeline"]
    metrics["comparison"] = {
        "cer_absolute_drop": round(full["cer_overall"] - region["cer_overall"], 4),
        "cer_relative_drop_pct": round(
            (full["cer_overall"] - region["cer_overall"]) / full["cer_overall"] * 100.0, 2
        )
        if full["cer_overall"] > 0
        else 0.0,
        "sentence_accuracy_gain_pct": round(
            (region["sentence_accuracy"] - full["sentence_accuracy"]) * 100.0, 2
        ),
        "time_cost_ratio": round(region["avg_ocr_time_ms"] / full["avg_time_ms"], 2)
        if full["avg_time_ms"]
        else None,
        "note": "CER 越低越好；时间比 = 区域路线 OCR 总耗时 / 整图 OCR 耗时（不含检测）。",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_ocr.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "ocr_texts.json").write_text(
        json.dumps({"config": metrics["config"], "images": texts_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 68)
    print(f"整图 OCR  : CER={full['cer_overall']:.4f}  整句准确率={full['sentence_accuracy']:.4f}  "
          f"平均 {full['avg_time_ms']:.0f}ms/图")
    print(f"分区域 OCR: CER={region['cer_overall']:.4f}  整句准确率={region['sentence_accuracy']:.4f}  "
          f"平均 OCR {region['avg_ocr_time_ms']:.0f}ms + 检测 {region['avg_detection_ms'] or 0:.0f}ms")
    print(f"指标写入: {out_dir / 'metrics_ocr.json'}")
    print("=" * 68)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR 整图 vs 区域裁剪 OCR + CER 评测")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--preds", type=Path, default=None, help="默认 outputs/yolo_predictions.json")
    parser.add_argument("--metrics-yolo", type=Path, default=None, help="默认 outputs/metrics_yolo.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    parser.add_argument("--rec-model", default=DEFAULT_REC_MODEL)
    parser.add_argument("--enable-mkldnn", action="store_true", help="默认关闭（兼容性）")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    predictions_path = args.preds or (args.out / "yolo_predictions.json")
    metrics_yolo_path = args.metrics_yolo or (args.out / "metrics_yolo.json")
    evaluate(
        data_dir=args.data_dir,
        out_dir=args.out,
        predictions_path=predictions_path,
        metrics_yolo_path=metrics_yolo_path,
        split=args.split,
        det_model=args.det_model,
        rec_model=args.rec_model,
        enable_mkldnn=args.enable_mkldnn,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
