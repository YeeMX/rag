"""在真实扫描件（XFUND 中文验证集）上验证合成数据上的结论是否成立。

任务背景：之前 `run_demo.py` 在**合成**法律文书上得到「分区域 OCR 优于整图 OCR」等结论，
本脚本用**真实扫描表单**做域外验证，如实量化域差距（domain gap）：

1. **OCR CER**：PaddleOCR（关 MKLDNN，PP-OCRv5 mobile）对真实扫描页做整图 OCR，
   与 XFUND 的实体级 GT 文本计算 CER（复用 `src/ocr_paddle.py` 里自己实现的 Levenshtein）；
2. **YOLO 域差距**：用合成数据训练的 `models/best.pt`（4 类）在真实页上推理，
   做**类别无关**评估：GT 文本框被 IoU≥0.5 覆盖的 recall、中心点命中率、
   GT 面积被覆盖比例、平均置信度、检测框数 / GT 框数；
   因为 XFUND 的标签语义是 header/question/answer/other，与 title/paragraph/table/stamp
   不可直接映射，所以**不报 mAP**；
3. **结构化抽取**：XFUND 不是劳动合同模板，字段抽取不可直接套用，如实标注「不适用」。

用法（推荐用 .venv-vision 启动，脚本会自动用 .venv-paddle 跑 OCR 子步骤）::

    # 首次运行：自动下载 XFUND zh.val（约 66MB + 1.7MB）
    python scripts/eval_real_dataset.py --download

    # 已有数据后复现（8 页；OCR 约 27s/页，总耗时约 5 分钟）
    python scripts/eval_real_dataset.py --max-pages 8

    # 只跑某一步（调试用）
    python scripts/eval_real_dataset.py --stage yolo
    python scripts/eval_real_dataset.py --stage ocr
    python scripts/eval_real_dataset.py --stage aggregate

数据来源：XFUND v1.0（ACL 2022 Findings，微软多语言表单理解基准）
    https://github.com/doc-analysis/XFUND/releases/tag/v1.0
    zh.val 集：50 页真实扫描中文表单（本脚本默认取前 8 页）。
    官方仓库未附明确 LICENSE 文件（HuggingFace 镜像标注 CC BY-NC-SA 4.0），
    本项目只做本地研究/作品集验证，不再分发数据（`data/real/` 已在 .gitignore 中忽略）。

所有数字均为实际运行结果，写入 `outputs/real_dataset/metrics_real.json`。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "real"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "real_dataset"

XFUND_IMAGES_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/zh.val.zip"
XFUND_ANNOTATIONS_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/zh.val.json"
XFUND_IMAGES_BYTES = 69_217_820
XFUND_ANNOTATIONS_BYTES = 1_711_142
DATASET_NAME = "XFUND v1.0 zh.val（真实扫描中文表单）"
DATASET_URL = "https://github.com/doc-analysis/XFUND/releases/tag/v1.0"

#: 合成数据上的对照值（由 outputs/metrics_ocr.json / metrics_yolo.json 动态读取，读不到则为 None）
SYNTHETIC_METRICS_OCR = PROJECT_ROOT / "outputs" / "metrics_ocr.json"
SYNTHETIC_METRICS_YOLO = PROJECT_ROOT / "outputs" / "metrics_yolo.json"


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def _download(url: str, target: Path, expected_bytes: int | None = None) -> None:
    """带进度提示的简单下载（只用标准库；对 GitHub release 302 跳转有效）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[real] 下载 {url} -> {target} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 dsh-vision-demo"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as fh:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    seconds = time.perf_counter() - t0
    size = target.stat().st_size
    print(f"[real] 下载完成: {size / 1024 / 1024:.1f} MB，{seconds:.1f}s")
    if expected_bytes is not None and size != expected_bytes:
        print(f"[real] 警告：文件大小 {size} 与预期 {expected_bytes} 不一致（仍继续，请检查来源）")


def ensure_dataset(data_dir: Path, *, download: bool) -> tuple[Path, Path]:
    """确保 XFUND zh.val 的图片目录与标注文件存在。返回 (images_dir, annotations_path)。"""
    images_dir = data_dir / "xfund_zh_val"
    annotations_path = data_dir / "zh.val.json"
    zip_path = data_dir / "zh.val.zip"
    have_images = images_dir.is_dir() and len(list(images_dir.glob("*.jpg"))) >= 1
    if have_images and annotations_path.is_file():
        return images_dir, annotations_path
    if not download:
        raise FileNotFoundError(
            "找不到 XFUND 数据。请先运行一次：\n"
            "    python scripts/eval_real_dataset.py --download\n"
            f"或手动下载:\n    {XFUND_IMAGES_URL}\n    {XFUND_ANNOTATIONS_URL}\n"
            f"解压图片到 {images_dir}，标注放到 {annotations_path}"
        )
    if not zip_path.is_file():
        _download(XFUND_IMAGES_URL, zip_path, XFUND_IMAGES_BYTES)
    if not annotations_path.is_file():
        _download(XFUND_ANNOTATIONS_URL, annotations_path, XFUND_ANNOTATIONS_BYTES)
    if not images_dir.is_dir():
        print(f"[real] 解压 {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
    if not images_dir.is_dir():
        # 兼容 zip 内多一层目录的情况
        candidates = [p for p in data_dir.iterdir() if p.is_dir() and list(p.glob("*.jpg"))]
        if not candidates:
            raise RuntimeError(f"解压后仍找不到图片目录: {data_dir}")
        images_dir = candidates[0]
    return images_dir, annotations_path


def build_gt_text(entities: Sequence[dict[str, Any]]) -> str:
    """把实体级标注按阅读顺序（先上沿、再左沿）拼成整页 GT 文本。

    XFUND 的 GT 是实体级（header/question/answer/other）标注，不是逐字全文转录；
    页面装饰性文字（页码等）可能未被标注，这是 CER 的一个已知偏差，README 中有说明。
    """
    ordered = sorted(entities, key=lambda e: (float(e["box"][1]), float(e["box"][0])))
    return "\n".join(str(e["text"]) for e in ordered if str(e["text"]).strip())


def _page_number(doc_id: str) -> int:
    try:
        return int(str(doc_id).rsplit("_", 1)[-1])
    except ValueError:
        return 0


def load_pages(
    annotations_path: Path,
    images_dir: Path,
    max_pages: int,
    *,
    min_entities: int = 10,
) -> list[dict[str, Any]]:
    """读取并选择评测页：按页号排序，过滤掉实体太少的页，取前 max_pages 页。"""
    ann = json.loads(annotations_path.read_text(encoding="utf-8"))
    documents = ann["documents"]
    documents = sorted(documents, key=lambda d: _page_number(d["id"]))
    pages: list[dict[str, Any]] = []
    for doc in documents:
        entities = doc.get("document") or []
        if len(entities) < min_entities:
            continue
        image_name = doc["img"]["fname"]
        image_path = images_dir / image_name
        if not image_path.is_file():
            continue
        gt_text = build_gt_text(entities)
        if not gt_text.strip():
            continue
        pages.append(
            {
                "id": doc["id"],
                "image": image_name,
                "image_path": str(image_path),
                "width": int(doc["img"].get("width") or 0),
                "height": int(doc["img"].get("height") or 0),
                "num_entities": len(entities),
                "gt_text": gt_text,
                "gt_boxes": [[float(v) for v in e["box"]] for e in entities],
                "entity_labels": [str(e.get("label", "")) for e in entities],
            }
        )
        if len(pages) >= max_pages:
            break
    if not pages:
        raise RuntimeError("没有找到可评测的页面（检查 images 目录与 annotations 是否匹配）")
    return pages


# ---------------------------------------------------------------------------
# YOLO 域差距（类别无关）
# ---------------------------------------------------------------------------


def _relative_path(path: Path) -> str:
    """转成相对项目根目录的路径（报告/指标可能被公开分享，避免泄露本机绝对路径）。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def gt_coverage_metrics(
    gt_boxes: Sequence[Sequence[float]],
    detections: Sequence[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """类别无关地衡量「检测框能不能把 GT 文本框框住」。

    Args:
        gt_boxes: GT 文本框（xyxy），XFUND 的实体级 bbox
        detections: YOLO 预测框 [{"bbox":..., "conf":..., "class_id":...}, ...]
        iou_threshold: IoU 命中阈值（默认 0.5）

    Returns:
        本页指标：
        - ``recall_iou50``：IoU≥0.5 的 GT 占比（严格框住率）；
        - ``center_hit_rate``：中心点至少落在 1 个检测框内的 GT 占比（0~1）；
        - ``center_hits_per_gt``：每个 GT 中心平均落在几个检测框内（可 >1，反映重复检测）；
        - ``mean_gt_area_coverage``：GT 框面积被最佳检测框覆盖的平均比例；
        - 平均置信度等。
    """
    from src.yolo_detect import iou_xyxy

    hits_iou = 0
    gt_with_center_hit = 0
    center_hits_total = 0
    coverages: list[float] = []
    matched_det_indices: set[int] = set()
    for gt in gt_boxes:
        gx1, gy1, gx2, gy2 = (float(v) for v in gt)
        cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
        gt_area = max(1e-6, _area(gt))
        best_iou = 0.0
        best_cov = 0.0
        best_index = -1
        center_hits_this_gt = 0
        for di, det in enumerate(detections):
            iou = iou_xyxy(gt, det["bbox"])
            if iou > best_iou:
                best_iou, best_index = iou, di
            inter_w = max(0.0, min(gx2, float(det["bbox"][2])) - max(gx1, float(det["bbox"][0])))
            inter_h = max(0.0, min(gy2, float(det["bbox"][3])) - max(gy1, float(det["bbox"][1])))
            cov = inter_w * inter_h / gt_area
            if cov > best_cov:
                best_cov = cov
            if (
                float(det["bbox"][0]) <= cx <= float(det["bbox"][2])
                and float(det["bbox"][1]) <= cy <= float(det["bbox"][3])
            ):
                center_hits_this_gt += 1
        if best_iou >= iou_threshold:
            hits_iou += 1
            if best_index >= 0:
                matched_det_indices.add(best_index)
        if center_hits_this_gt > 0:
            gt_with_center_hit += 1
        center_hits_total += center_hits_this_gt
        coverages.append(best_cov)

    matched_confs = [
        float(det["conf"]) for di, det in enumerate(detections) if di in matched_det_indices
    ]
    num_gt = len(gt_boxes)
    return {
        "num_gt": num_gt,
        "num_det": len(detections),
        "det_gt_ratio": round(len(detections) / num_gt, 3) if num_gt else None,
        "recall_iou50": round(hits_iou / num_gt, 4) if num_gt else None,
        "center_hit_rate": round(gt_with_center_hit / num_gt, 4) if num_gt else None,
        "center_hits_per_gt": round(center_hits_total / num_gt, 4) if num_gt else None,
        "center_hits_total": center_hits_total,
        "mean_gt_area_coverage": round(statistics.fmean(coverages), 4) if coverages else None,
        "mean_conf_all_dets": round(statistics.fmean(float(d["conf"]) for d in detections), 4)
        if detections
        else None,
        "mean_conf_iou_matched": round(statistics.fmean(matched_confs), 4) if matched_confs else None,
    }


def run_yolo_stage(
    pages: Sequence[dict[str, Any]],
    *,
    weights: Path,
    imgsz: int,
    conf: float,
    out_dir: Path,
) -> dict[str, Any]:
    """用合成数据训练的 YOLO 在真实页上推理，做类别无关的域差距评估。"""
    import platform

    import torch
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    from src.yolo_detect import CLASS_NAMES

    print(f"[real] 加载 YOLO 权重: {weights}")
    model = YOLO(str(weights))
    per_page: list[dict[str, Any]] = []
    class_distribution: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sampled = 0

    for page in pages:
        image_path = Path(page["image_path"])
        result = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None:
            for bbox, score, cls in zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
            ):
                detections.append(
                    {
                        "bbox": [round(float(v), 1) for v in bbox],
                        "conf": round(float(score), 4),
                        "class_id": int(cls),
                    }
                )
                class_distribution[CLASS_NAMES[int(cls)]] += 1

        metrics = gt_coverage_metrics(page["gt_boxes"], detections)
        per_page.append(
            {
                "id": page["id"],
                "image": page["image"],
                "width": page["width"],
                "height": page["height"],
                **metrics,
                "detections": detections,
            }
        )
        print(
            f"[real] YOLO {page['id']}: GT={metrics['num_gt']} det={metrics['num_det']} "
            f"recall_iou50={metrics['recall_iou50']} center_hit_rate={metrics['center_hit_rate']} "
            f"coverage={metrics['mean_gt_area_coverage']}"
        )

        # 保存前 3 页标注样例（GT 绿框 / 预测红框）
        if sampled < 3:
            with Image.open(image_path) as im:
                canvas = im.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for gt in page["gt_boxes"]:
                draw.rectangle([gt[0], gt[1], gt[2], gt[3]], outline=(0, 160, 0), width=4)
            for det in detections:
                b = det["bbox"]
                draw.rectangle([b[0], b[1], b[2], b[3]], outline=(220, 30, 30), width=5)
            max_w = 1200
            if canvas.width > max_w:
                ratio = max_w / canvas.width
                canvas = canvas.resize((max_w, int(canvas.height * ratio)), Image.LANCZOS)
            sample_path = samples_dir / f"{page['id']}_gt_vs_pred.png"
            canvas.save(sample_path)
            sampled += 1
            print(f"[real] 保存样例图: {sample_path}")

    # 聚合（按 GT 数加权）
    total_gt = sum(p["num_gt"] for p in per_page)
    total_det = sum(p["num_det"] for p in per_page)
    aggregate = {
        "num_pages": len(per_page),
        "total_gt_boxes": total_gt,
        "total_detections": total_det,
        "det_gt_ratio": round(total_det / total_gt, 3) if total_gt else None,
        "recall_iou50": round(sum(p["recall_iou50"] * p["num_gt"] for p in per_page) / total_gt, 4)
        if total_gt
        else None,
        "center_hit_rate": round(sum(p["center_hit_rate"] * p["num_gt"] for p in per_page) / total_gt, 4)
        if total_gt
        else None,
        "center_hits_per_gt": round(sum(p["center_hits_total"] for p in per_page) / total_gt, 4)
        if total_gt
        else None,
        "mean_gt_area_coverage": round(sum(p["mean_gt_area_coverage"] * p["num_gt"] for p in per_page) / total_gt, 4)
        if total_gt
        else None,
        "mean_conf_all_dets": round(statistics.fmean(p["mean_conf_all_dets"] for p in per_page if p["mean_conf_all_dets"] is not None), 4)
        if any(p["mean_conf_all_dets"] is not None for p in per_page)
        else None,
        "mean_conf_iou_matched": round(
            statistics.fmean(p["mean_conf_iou_matched"] for p in per_page if p["mean_conf_iou_matched"] is not None), 4
        )
        if any(p["mean_conf_iou_matched"] is not None for p in per_page)
        else None,
        "class_distribution": class_distribution,
        "iou_threshold": 0.5,
        "note": "类别无关评估：XFUND 标签为 header/question/answer/other，与合成数据 4 类不可直接映射，因此不报 mAP。",
    }
    payload = {
        "config": {
            "weights": _relative_path(weights),
            "trained_on": "合成法律文书（title/paragraph/table/stamp）",
            "imgsz": imgsz,
            "conf": conf,
            "device": "cpu",
            "dataset": DATASET_NAME,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "aggregate": aggregate,
        "pages": per_page,
    }
    (out_dir / "yolo_real_predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# OCR：整图识别 + CER
# ---------------------------------------------------------------------------


def run_ocr_stage(
    pages: Sequence[dict[str, Any]],
    *,
    out_dir: Path,
    det_model: str,
    rec_model: str,
    enable_mkldnn: bool,
) -> dict[str, Any]:
    """整图 OCR（PaddleOCR）→ CER。需要在 .venv-paddle 环境执行。"""
    import platform

    from src.ocr_paddle import (
        init_ocr,
        levenshtein_distance,
        normalize_text,
        ocr_one_image,
    )

    print(f"[real] 初始化 PaddleOCR: det={det_model} rec={rec_model} mkldnn={enable_mkldnn}")
    t0 = time.perf_counter()
    ocr = init_ocr(det_model=det_model, rec_model=rec_model, enable_mkldnn=enable_mkldnn)
    init_seconds = time.perf_counter() - t0
    print(f"[real] 初始化耗时 {init_seconds:.1f}s")
    # 用小裁剪图预热，避免第一页整页 OCR 的额外初始化开销影响计时
    try:
        from PIL import Image

        with Image.open(pages[0]["image_path"]) as im:
            warm = im.convert("RGB").crop((0, 0, min(800, im.width), min(400, im.height)))
        warm_path = out_dir / "_warmup.png"
        warm_path.parent.mkdir(parents=True, exist_ok=True)
        warm.save(warm_path)
        ocr_one_image(ocr, warm_path)
        warm_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[real] 预热跳过: {exc}")
    print("[real] 预热完成，开始逐页 OCR ...")

    per_page: list[dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        result = ocr_one_image(ocr, Path(page["image_path"]))
        hyp_text = "\n".join(result["texts"])
        ref = normalize_text(page["gt_text"])
        hyp = normalize_text(hyp_text)
        distance = levenshtein_distance(ref, hyp)
        cer = distance / len(ref) if ref else 0.0
        per_page.append(
            {
                "id": page["id"],
                "image": page["image"],
                "gt_chars": len(ref),
                "ocr_chars": len(hyp),
                "ocr_lines": len(result["texts"]),
                "edit_distance": distance,
                "cer": round(cer, 4),
                "time_ms": result["time_ms"],
                "ocr_text": hyp_text,
            }
        )
        print(
            f"[real] OCR {index}/{len(pages)} {page['id']}: CER={cer:.4f} "
            f"(edit={distance}/{len(ref)}) time={result['time_ms'] / 1000:.1f}s lines={len(result['texts'])}"
        )

    total_ref = sum(p["gt_chars"] for p in per_page)
    total_edit = sum(p["edit_distance"] for p in per_page)
    times = [p["time_ms"] for p in per_page]
    aggregate = {
        "num_pages": len(per_page),
        "cer_weighted": round(total_edit / total_ref, 4) if total_ref else None,
        "cer_mean_per_page": round(statistics.fmean(p["cer"] for p in per_page), 4) if per_page else None,
        "cer_min": round(min(p["cer"] for p in per_page), 4) if per_page else None,
        "cer_max": round(max(p["cer"] for p in per_page), 4) if per_page else None,
        "total_edit_distance": total_edit,
        "total_gt_chars": total_ref,
        "avg_time_ms": round(statistics.fmean(times), 1) if times else None,
        "median_time_ms": round(statistics.median(times), 1) if times else None,
        "total_ocr_seconds": round(sum(times) / 1000.0, 1),
    }
    payload = {
        "config": {
            "pipeline": "整图直接 OCR（不裁剪）",
            "det_model": det_model,
            "rec_model": rec_model,
            "lang": "ch",
            "enable_mkldnn": enable_mkldnn,
            "init_seconds": round(init_seconds, 2),
            "warmup": "用第一页左上角小裁剪图预热一次，不计入逐页耗时。",
            "device": "cpu",
            "dataset": DATASET_NAME,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "aggregate": aggregate,
        "pages": per_page,
    }
    (out_dir / "ocr_real.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[real] OCR 汇总: CER(加权)={aggregate['cer_weighted']} "
        f"平均耗时={aggregate['avg_time_ms']}ms/页 共 {aggregate['num_pages']} 页"
    )
    return payload


# ---------------------------------------------------------------------------
# 汇总：metrics_real.json + report_real.md
# ---------------------------------------------------------------------------


def load_synthetic_reference() -> dict[str, Any]:
    """读取合成数据集上的对照指标（读不到就返回 None）。"""
    reference: dict[str, Any] = {"ocr_synthetic_full_image_cer": None, "yolo_synthetic_mAP50": None}
    try:
        ocr = json.loads(SYNTHETIC_METRICS_OCR.read_text(encoding="utf-8"))
        reference["ocr_synthetic_full_image_cer"] = ocr["full_image"]["cer_overall"]
        reference["ocr_synthetic_region_cer"] = ocr["region_pipeline"]["cer_overall"]
        reference["ocr_synthetic_avg_time_ms"] = ocr["full_image"]["avg_time_ms"]
        reference["ocr_synthetic_config"] = {
            "models": f"{ocr['config']['det_model']} + {ocr['config']['rec_model']}",
            "eval_pages": ocr["config"]["num_images"],
            "data": "合成法律文书（1000x1414，A4 比例）",
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    try:
        yolo = json.loads(SYNTHETIC_METRICS_YOLO.read_text(encoding="utf-8"))
        reference["yolo_synthetic_mAP50"] = yolo["ultralytics_val"]["mAP50"]
        reference["yolo_synthetic_mAP50_95"] = yolo["ultralytics_val"]["mAP50_95"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return reference


def build_metrics(
    out_dir: Path,
    data_dir: Path,
    *,
    max_pages: int,
    pages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """合并 YOLO / OCR 两阶段结果，写 metrics_real.json。"""
    yolo_payload = json.loads((out_dir / "yolo_real_predictions.json").read_text(encoding="utf-8"))
    ocr_payload = json.loads((out_dir / "ocr_real.json").read_text(encoding="utf-8"))
    synthetic = load_synthetic_reference()

    # 下载文件大小（真实证据）
    zip_path = data_dir / "zh.val.zip"
    dataset_files = {}
    for name in ("zh.val.zip", "zh.val.json"):
        path = data_dir / name
        dataset_files[name] = path.stat().st_size if path.is_file() else None

    metrics = {
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_URL,
            "license_note": "官方仓库未附明确 LICENSE 文件；HuggingFace 镜像标注为 CC BY-NC-SA 4.0。"
            "本项目仅用于本地研究/作品集验证，不再分发数据（data/real/ 已被 .gitignore 忽略）。",
            "split": "zh.val",
            "total_pages_in_split": 50,
            "evaluated_pages": len(pages),
            "page_ids": [p["id"] for p in pages],
            "image_size": [pages[0]["width"], pages[0]["height"]],
            "dataset_files_bytes": dataset_files,
            "is_synthetic": False,
        },
        "ocr": ocr_payload,
        "yolo_domain_gap": {
            "config": yolo_payload["config"],
            "aggregate": yolo_payload["aggregate"],
            "per_page": [
                {
                    key: value
                    for key, value in page.items()
                    if key != "detections"
                }
                for page in yolo_payload["pages"]
            ],
        },
        "structured_extraction": {
            "applicable": False,
            "reason": "XFUND 是真实表单（header/question/answer 语义的问答对），不是劳动合同模板；"
            "现有规则抽取器（合同期限/工资/试用期等字段）无法直接套用，硬套会得到没有意义的数字。"
            "如需迁移，需要重新定义字段 schema 并标注少量真实样本后训练/写规则。",
        },
        "comparison_with_synthetic": synthetic,
        "notes": [
            "XFUND GT 是实体级标注，不是逐字全文转录；页面装饰文字（如页码）可能未被标注，"
            "整图 OCR 读到的这类文字会作为插入错误计入 CER，因此真实 CER 略偏高。",
            "YOLO 用合成数据训练、真实数据推理，本质上是一次 OOD（out-of-distribution）测试；"
            "XFUND 的 GT 是行/词级实体框，而合成模型学的是版面块级框，粒度不匹配是 recall_iou50 极低的主因。",
            "本评测不报告 mAP：类别语义不可映射。",
        ],
    }
    (out_dir / "metrics_real.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report_real(out_dir, metrics)
    return metrics


def write_report_real(out_dir: Path, metrics: dict[str, Any]) -> None:
    """生成 outputs/real_dataset/report_real.md（README 引用的证据文件）。"""
    ocr = metrics["ocr"]
    yolo = metrics["yolo_domain_gap"]
    dataset = metrics["dataset"]
    ref = metrics["comparison_with_synthetic"]
    ocr_agg = ocr["aggregate"]
    yolo_agg = yolo["aggregate"]

    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "未测量"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# 真实扫描件验证报告（XFUND zh.val）",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　|　数据：{dataset['name']}　|　"
        f"评测 {dataset['evaluated_pages']} 页　|　环境：Windows + CPU",
        f"> 样本量：仅 {dataset['evaluated_pages']} 页（真实页整图 OCR 约 30~90s/页，样本量有限，指标为方向性结论，不能当统计显著性）。",
        ">",
        f"> 数据来源：{dataset['source']}",
        "> **这是真实扫描件（非合成数据）**；所有数字来自实际运行。",
        "",
        "## 1. OCR：真实页整图识别 CER",
        "",
        f"- 模型：{ocr['config']['det_model']} + {ocr['config']['rec_model']}（MKLDNN={ocr['config']['enable_mkldnn']}，CPU）",
        f"- 机制：{ocr['config']['pipeline']}；{ocr['config']['warmup']}",
        "",
        "| 指标 | 真实扫描页 | 合成数据（对照） |",
        "|---|---|---|",
        f"| CER（加权） | **{fmt(ocr_agg['cer_weighted'])}** | "
        f"{fmt(ref.get('ocr_synthetic_full_image_cer'))} |",
        f"| CER 均值 | {fmt(ocr_agg['cer_mean_per_page'])} | - |",
        f"| 平均单页 OCR 耗时 | {fmt(ocr_agg['avg_time_ms'], 0)} ms | "
        f"{fmt(ref.get('ocr_synthetic_avg_time_ms'), 0)} ms |",
        f"| 评测页数 | {ocr_agg['num_pages']} | 8（val 集） |",
        "",
        "逐页明细：",
        "",
        "| 页面 | GT 字数 | OCR 字数 | 编辑距离 | CER | 耗时 |",
        "|---|---|---|---|---|---|",
    ]
    for page in ocr["pages"]:
        lines.append(
            f"| {page['id']} | {page['gt_chars']} | {page['ocr_chars']} | "
            f"{page['edit_distance']} | {fmt(page['cer'])} | {fmt(page['time_ms'], 0)} ms |"
        )
    lines += [
        "",
        f"共 {ocr_agg['num_pages']} 页、{ocr_agg['total_gt_chars']} 个 GT 字符，"
        f"总 OCR 耗时 {ocr_agg['total_ocr_seconds']} s。",
        "",
        "## 2. YOLO 域差距（类别无关，不报 mAP）",
        "",
        f"- 模型：{yolo['config']['weights']}（训练数据：{yolo['config']['trained_on']}）",
        f"- 推理：imgsz={yolo['config']['imgsz']}，conf={yolo['config']['conf']}，device=cpu",
        f"- 说明：{yolo_agg['note']}",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| GT 文本框数 | {yolo_agg['total_gt_boxes']} |",
        f"| 检测框数 | {yolo_agg['total_detections']} |",
        f"| 检测框/GT框 数量比 | {fmt(yolo_agg['det_gt_ratio'], 2)} |",
        f"| IoU≥0.5 覆盖率（类别无关 recall） | **{fmt(yolo_agg['recall_iou50'])}** |",
        f"| 中心点命中率（至少 1 个检测框覆盖 GT 中心，0~1） | {fmt(yolo_agg['center_hit_rate'])} |",
        f"| 中心点平均命中个数（每 GT 落在几个检测框内，可 >1） | {fmt(yolo_agg['center_hits_per_gt'])} |",
        f"| GT 面积平均被覆盖比例 | {fmt(yolo_agg['mean_gt_area_coverage'])} |",
        f"| 检测框平均置信度 | {fmt(yolo_agg['mean_conf_all_dets'])} |",
        f"| 命中检测框平均置信度（IoU≥0.5） | {fmt(yolo_agg['mean_conf_iou_matched'])} |",
        f"| 检测类别分布 | {json.dumps(yolo_agg['class_distribution'], ensure_ascii=False)} |",
        "",
        "逐页明细：",
        "",
        "| 页面 | GT 数 | 检测数 | 数量比 | recall_iou50 | 中心命中率 | 面积覆盖 | 平均置信度 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for page in yolo["per_page"]:
        lines.append(
            f"| {page['id']} | {page['num_gt']} | {page['num_det']} | {fmt(page['det_gt_ratio'], 2)} | "
            f"{fmt(page['recall_iou50'])} | {fmt(page['center_hit_rate'])} | "
            f"{fmt(page['mean_gt_area_coverage'])} | {fmt(page['mean_conf_all_dets'])} |"
        )
    lines += [
        "",
        "## 3. 结构化抽取",
        "",
        f"**不适用**：{metrics['structured_extraction']['reason']}",
        "",
        "## 4. 结论与局限",
        "",
    ]
    lines += [f"- {note}" for note in metrics["notes"]]
    synth_cer = ref.get("ocr_synthetic_full_image_cer")
    ratio = (ocr_agg["cer_weighted"] / synth_cer) if synth_cer else None
    lines += [
        f"- **域差距**：真实扫描件整图 OCR CER = {fmt(ocr_agg['cer_weighted'])}，合成数据为 {fmt(synth_cer)}，"
        f"约 {fmt(ratio, 2)} 倍；说明合成数据上的绝对值不能直接外推。",
        "- **结论是否迁移**：合成数据上「检测→区域裁剪→OCR 提升质量」的结论在真实域**无法直接复现**："
        "合成训练的检测器在真实表单上 IoU≥0.5 的文本区域覆盖率仅 "
        f"{fmt(yolo_agg['recall_iou50'])}，裁剪区域会错位，瓶颈在检测器的域泛化能力，而不是 OCR 模型本身。",
        "- 真实 CER 偏高的原因：真实扫描噪声/倾斜/中英混排、GT 是实体级标注（页码等装饰文字未标注），"
        "以及 OCR 对真实表单密集小字/表格线的识别难度。",
        "",
        "## 5. 样例图（GT 绿框 / YOLO 预测红框）",
        "",
    ]
    samples = sorted((out_dir / "samples").glob("*_gt_vs_pred.png"))
    for sample in samples:
        lines.append(f"![{sample.stem}](samples/{sample.name})")
        lines.append("")
    (out_dir / "report_real.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在真实扫描件（XFUND zh.val）上验证合成数据的结论")
    parser.add_argument("--stage", choices=["all", "yolo", "ocr", "aggregate"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-pages", type=int, default=8, help="评测页数（默认 8，OCR 约 27s/页）")
    parser.add_argument("--min-entities", type=int, default=10)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--det-model", default="PP-OCRv5_mobile_det")
    parser.add_argument("--rec-model", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--enable-mkldnn", action="store_true")
    parser.add_argument("--paddle-python", default=None)
    parser.add_argument("--download", action="store_true", help="数据缺失时自动下载 XFUND zh.val")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "aggregate":
        pages = load_pages(
            data_dir / "zh.val.json",
            data_dir / "xfund_zh_val",
            args.max_pages,
            min_entities=args.min_entities,
        )
        metrics = build_metrics(out_dir, data_dir, max_pages=args.max_pages, pages=pages)
        print(f"[real] 指标写入 {out_dir / 'metrics_real.json'} 与 {out_dir / 'report_real.md'}")
        return 0

    images_dir, annotations_path = ensure_dataset(data_dir, download=args.download)
    pages = load_pages(annotations_path, images_dir, args.max_pages, min_entities=args.min_entities)
    print(f"[real] 评测页: {[p['id'] for p in pages]}（共 {len(pages)} 页，真实扫描件）")

    if args.stage in ("yolo", "all"):
        run_yolo_stage(pages, weights=args.weights, imgsz=args.imgsz, conf=args.conf, out_dir=out_dir)

    if args.stage in ("ocr", "all"):
        if args.stage == "all":
            # 自动用 .venv-paddle 子进程跑 OCR，避免 torch / paddle 依赖冲突
            from run_demo import find_paddle_python

            paddle_python = find_paddle_python(args.paddle_python)
            cmd = [
                str(paddle_python),
                str(Path(__file__).resolve()),
                "--stage",
                "ocr",
                "--data-dir",
                str(data_dir),
                "--out",
                str(out_dir),
                "--max-pages",
                str(args.max_pages),
                "--min-entities",
                str(args.min_entities),
                "--det-model",
                args.det_model,
                "--rec-model",
                args.rec_model,
            ]
            if args.enable_mkldnn:
                cmd.append("--enable-mkldnn")
            print("[real] 启动 OCR 子步骤（paddle 环境）: " + " ".join(cmd))
            t0 = time.perf_counter()
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            print(f"[real] OCR 子步骤完成: {time.perf_counter() - t0:.1f}s, returncode={result.returncode}")
            if result.returncode != 0:
                raise RuntimeError("OCR 子步骤失败")
        else:
            run_ocr_stage(
                pages,
                out_dir=out_dir,
                det_model=args.det_model,
                rec_model=args.rec_model,
                enable_mkldnn=args.enable_mkldnn,
            )

    if args.stage == "all":
        # OCR 子步骤只写 ocr_real.json，汇总统一由父进程做，避免重复计算
        build_metrics(out_dir, data_dir, max_pages=args.max_pages, pages=pages)
        print(f"[real] 指标写入 {out_dir / 'metrics_real.json'} 与 {out_dir / 'report_real.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

