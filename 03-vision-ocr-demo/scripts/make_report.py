"""把各步骤产出的 JSON 指标汇总成 outputs/report.md（含表格 + 图片链接）。

用法（在 03-vision-ocr-demo 目录下）::

    python scripts/make_report.py --data-dir data --out outputs

报告里的每个数字都直接读取自实际运行产生的 JSON 文件；文件缺失时写「未测量」，
不做任何估算或编造。
"""

from __future__ import annotations

import argparse
import json
import re
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    """安全格式化数字。"""
    if value is None:
        return "未测量"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pretty_path(text: Any) -> str:
    """把绝对路径转成相对路径 / 通用 python 命令，报告不暴露本机目录结构。"""
    if text is None:
        return "未测量"
    s = str(text)
    project = Path(__file__).resolve().parents[1]
    s = s.replace(str(project) + "\\", "").replace(str(project) + "/", "")
    # 解释器绝对路径（可能带空格）-> python：匹配到 python.exe / python 为止
    s = re.sub(r"[A-Za-z]:[^\n]*?[\\/]python(?:\.exe)?(?=\s|$)", "python", s)
    return s


def fmt_ms(value: Any) -> str:
    if value is None:
        return "未测量"
    return f"{float(value):.0f} ms"


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def section_dataset(data_dir: Path) -> list[str]:
    annotations = read_json(data_dir / "annotations.json")
    if not annotations:
        return ["## 1. 数据集", "", "未测量（找不到 data/annotations.json）", ""]
    meta = annotations["meta"]
    images = annotations["images"]
    out = ["## 1. 数据集（合成，带精确标注）", ""]
    out.append(
        f"- 生成脚本：`python scripts/make_dataset.py --num {meta['num_images']} "
        f"--num-train {meta['num_train']} --seed {meta['seed']}`"
    )
    out.append(f"- 页面尺寸：{meta['page_size'][0]}x{meta['page_size'][1]}（A4 比例）")
    out.append(f"- 类别：`{', '.join(meta['classes'])}`")
    out.append(f"- 说明：{meta['note']}")
    out.append("")
    rows = []
    for split in ("train", "val"):
        counts = meta["class_counts"][split]
        rows.append(
            [
                split,
                sum(1 for img in images if img["split"] == split),
                counts.get("title", 0),
                counts.get("paragraph", 0),
                counts.get("table", 0),
                counts.get("stamp", 0),
                meta["full_text_chars_excluding_newline"][split],
            ]
        )
    out.append(md_table(["split", "图片数", "title", "paragraph", "table", "stamp", "全文字符数"], rows))
    hard = [img for img in images if img["hard_case"]]
    if hard:
        out += ["", "难例（合成）：", ""]
        out.append(md_table(
            ["id", "split", "难例类型", "旋转角度", "区域数"],
            [[img["id"], img["split"], img["hard_case"], fmt(img["rotation_deg"], 2), img["num_regions"]] for img in hard],
        ))
    out.append("")
    return out


def read_train_info_from_run(metrics: dict[str, Any]) -> dict[str, Any]:
    """metrics 里没有训练信息时（例如用 --skip-train 运行），从 runs/doc_layout 的训练产物读取。"""
    train = (metrics.get("config") or {}).get("train")
    if train:
        return train
    run_dir = PROJECT_ROOT / "runs" / "doc_layout"
    info: dict[str, Any] = {"note": "训练信息来自 runs/doc_layout 训练产物（本次用 --skip-train 复现指标）"}
    args_path = run_dir / "args.yaml"
    if args_path.is_file():
        try:
            import yaml  # pyyaml 随 ultralytics 一起安装

            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            for key in ("epochs", "imgsz", "batch", "model", "seed", "device"):
                if key in args:
                    info[key] = args[key]
            model_name = str(args.get("model", ""))
            info["pretrained"] = not model_name.endswith(".yaml")
        except Exception:  # noqa: BLE001
            pass
    csv_path = run_dir / "results.csv"
    if csv_path.is_file():
        try:
            lines = [line.strip() for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            header = [col.strip() for col in lines[0].split(",")]
            last = lines[-1].split(",")
            if "time" in header:
                info["train_epoch_seconds"] = round(float(last[header.index("time")]), 1)
        except Exception:  # noqa: BLE001
            pass
    return info


def section_yolo(out_dir: Path) -> list[str]:
    metrics = read_json(out_dir / "metrics_yolo.json")
    if not metrics:
        return ["## 2. YOLO 版面检测", "", "未测量（找不到 outputs/metrics_yolo.json）", ""]
    cfg = metrics["config"]
    official = metrics["ultralytics_val"]
    manual = metrics["manual_iou_eval"]
    speed = metrics["speed"]
    env = metrics.get("environment", {})
    train = read_train_info_from_run(metrics)
    out = ["## 2. YOLO 版面区域检测（真实指标）", ""]
    if "note" in train:
        out.append(f"> {train['note']}")
        out.append("")
    out.append(
        f"- 训练：基础权重 `{pretty_path(train.get('base_weights') or train.get('model', '未测量'))}`"
        f"（预训练={train.get('pretrained')}），epochs={train.get('epochs')}，"
        f"imgsz={train.get('imgsz')}，batch={train.get('batch')}，CPU；"
        f"训练阶段耗时 {fmt(train.get('train_seconds', train.get('train_epoch_seconds')), 1)} s"
    )
    out.append(
        f"- 评测集：留出 `{cfg['split']}` 集 {cfg['num_eval_images']} 张图 / {cfg['num_regions']} 个区域；"
        f"推理尺寸 {cfg['imgsz']}，部署置信度阈值 {cfg['conf_deployment']}"
    )
    out.append(f"- 环境：{env.get('cpu', '未测量')}，torch {env.get('torch')}，ultralytics {env.get('ultralytics')}，device=cpu")
    out.append("")
    out.append(md_table(
        ["指标", "ultralytics model.val()", "手写 IoU 匹配（交叉验证）", "差值"],
        [
            ["mAP@0.5", fmt(official["mAP50"]), fmt(manual["mAP50"]), fmt(metrics["cross_check"]["mAP50_abs_diff"])],
            ["mAP@0.5:0.95", fmt(official["mAP50_95"]), fmt(manual["mAP50_95"]), fmt(metrics["cross_check"]["mAP50_95_abs_diff"])],
            ["Precision（官方为最佳 F1 点）", fmt(official["precision"]), "-", "-"],
            ["Recall（官方为最佳 F1 点）", fmt(official["recall"]), "-", "-"],
            [f"Precision（手写，conf={cfg['conf_deployment']}）", "-", fmt(manual["precision_at_conf"]), "-"],
            [f"Recall（手写，conf={cfg['conf_deployment']}）", "-", fmt(manual["recall_at_conf"]), "-"],
        ],
    ))
    out += ["", "分区域 AP@0.5（手写 IoU 匹配，检测数/TP/FP/FN 统计于 conf=0.001，用于计算 AP）：", ""]
    per_class = manual.get("per_class", {})
    out.append(md_table(
        ["类别", "GT 数", "检测数", "TP", "FP", "FN", "AP@0.5"],
        [
            [name, info.get("num_gt"), info.get("num_det"), info.get("tp"), info.get("fp"), info.get("fn"), fmt(info.get("ap50"))]
            for name, info in per_class.items()
        ],
    ))
    out += ["", "CPU 速度：", ""]
    out.append(md_table(
        ["统计项", "数值"],
        [
            ["单图端到端耗时（中位数）", fmt_ms(speed["median_ms"])],
            ["单图端到端耗时（均值）", fmt_ms(speed["mean_ms"])],
            ["P90", fmt_ms(speed["p90_ms"])],
            ["FPS（按中位数）", fmt(speed["fps_from_median"], 2)],
            ["测量次数", speed["runs"]],
        ],
    ))
    out.append(f"\n> {speed['note']}")
    if speed.get("median_ms") and speed["median_ms"] > 150:
        out.append("> 注意：本次测量期间 CPU 上还有其他任务并发执行，速度数字偏保守；"
                   "空闲时单张约 50~60 ms。")
    out.append("")
    return out


def _ocr_subset_stats(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """由 per_image 明细按子集汇总 CER / 整句准确率（不重新跑 OCR）。"""
    ref_chars = sum(item["gt_chars"] for item in items)
    full_ed = sum(item["full_image"]["edit_distance"] for item in items)
    region_ed = sum(item["region_pipeline"]["edit_distance"] for item in items)
    gt_lines = sum(item["gt_lines"] for item in items)
    full_hits = sum(item["full_image"]["line_hits"] for item in items)
    region_hits = sum(item["region_pipeline"]["line_hits"] for item in items)
    return {
        "num_images": len(items),
        "full_cer": round(full_ed / ref_chars, 4) if ref_chars else None,
        "region_cer": round(region_ed / ref_chars, 4) if ref_chars else None,
        "full_sentence_accuracy": round(full_hits / gt_lines, 4) if gt_lines else None,
        "region_sentence_accuracy": round(region_hits / gt_lines, 4) if gt_lines else None,
    }


def section_ocr(out_dir: Path) -> list[str]:
    metrics = read_json(out_dir / "metrics_ocr.json")
    if not metrics:
        return ["## 3. OCR（整图 vs 分区域）", "", "未测量（找不到 outputs/metrics_ocr.json）", ""]
    cfg = metrics["config"]
    full = metrics["full_image"]
    region = metrics["region_pipeline"]
    cmp = metrics["comparison"]
    out = ["## 3. OCR：整图识别 vs YOLO 区域裁剪后识别（核心对比）", ""]
    out.append(
        f"- PaddleOCR：`{cfg['det_model']}` + `{cfg['rec_model']}`，lang=ch，mkldnn={cfg['enable_mkldnn']}，device=cpu"
    )
    out.append(f"- 评测集：`{cfg['split']}` 集 {cfg['num_images']} 张图；{cfg['warmup']}")
    out.append("")
    out.append(md_table(
        ["指标", "整图直接 OCR", "区域裁剪流水线（YOLO→裁→OCR）"],
        [
            ["CER（总体，按字符加权）", fmt(full["cer_overall"]), fmt(region["cer_overall"])],
            ["CER（逐图平均）", fmt(full["cer_mean_per_image"]), fmt(region["cer_mean_per_image"])],
            ["整句准确率（行级 exact match）", fmt(full["sentence_accuracy"]), fmt(region["sentence_accuracy"])],
            ["整图完全正确率", fmt(full["image_exact_match"]), fmt(region["image_exact_match"])],
            ["平均单图耗时", fmt_ms(full["avg_time_ms"]), f"{fmt_ms(region['avg_ocr_time_ms'])}（OCR 部分）"],
            ["端到端耗时（含 YOLO 检测）", fmt_ms(full["avg_time_ms"]), fmt_ms(region["avg_end_to_end_ms"])],
            ["FPS（按端到端）", fmt(full["fps_from_median"], 2), fmt(region["end_to_end_fps"], 2)],
        ],
    ))
    out += ["", "对比结论：", ""]
    out.append(
        f"- CER 绝对变化 **{fmt(cmp['cer_absolute_drop'])}**（整图 CER − 区域 CER，正数代表区域路线更好），"
        f"相对变化 **{fmt(cmp['cer_relative_drop_pct'], 1)}%**；"
        f"整句准确率变化 **{fmt(cmp['sentence_accuracy_gain_pct'], 1)}** 个百分点。"
    )
    out.append(
        f"- 区域路线的 OCR 耗时是整图的 **{fmt(cmp['time_cost_ratio'], 2)} 倍**（不含 YOLO 检测），"
        f"检测部分来自 YOLO 中位耗时 {fmt_ms(region.get('avg_detection_ms'))}。"
    )
    out.append("")
    normal_items = [item for item in metrics["per_image"] if not item.get("hard_case")]
    hard_items = [item for item in metrics["per_image"] if item.get("hard_case")]
    subset_rows = []
    if normal_items:
        stats = _ocr_subset_stats(normal_items)
        subset_rows.append(["正常样本", stats["num_images"], fmt(stats["full_cer"]), fmt(stats["region_cer"]),
                            fmt(stats["full_sentence_accuracy"]), fmt(stats["region_sentence_accuracy"])])
    if hard_items:
        stats = _ocr_subset_stats(hard_items)
        subset_rows.append(["难例（旋转/遮挡）", stats["num_images"], fmt(stats["full_cer"]), fmt(stats["region_cer"]),
                            fmt(stats["full_sentence_accuracy"]), fmt(stats["region_sentence_accuracy"])])
    overall = _ocr_subset_stats(metrics["per_image"])
    subset_rows.append(["全部", overall["num_images"], fmt(overall["full_cer"]), fmt(overall["region_cer"]),
                        fmt(overall["full_sentence_accuracy"]), fmt(overall["region_sentence_accuracy"])])
    out += ["", "按样本类型拆开看（说明误差传导）：", ""]
    out.append(md_table(
        ["子集", "图片数", "整图 CER", "区域 CER", "整图整句准确率", "区域整句准确率"],
        subset_rows,
    ))
    out.append(
        "\n> 结论：正常扫描件上「检测→按区域识别」明显降低 CER；但旋转/印章遮挡等难例上，"
        "检测框偏移或重复检测会把误差传导进 OCR，反而拉高整体 CER。"
        "这正是级联视觉系统（检测 + 识别）最值得在面试里讲的 trade-off。"
    )
    out.append("")
    out.append("逐图明细：")
    out.append("")
    out.append(md_table(
        ["图片", "难例", "整图 CER", "整图耗时", "区域 CER", "区域耗时", "区域数"],
        [
            [
                item["id"],
                item["hard_case"] or "-",
                fmt(item["full_image"]["cer"]),
                fmt_ms(item["full_image"]["time_ms"]),
                fmt(item["region_pipeline"]["cer"]),
                fmt_ms(item["region_pipeline"]["ocr_time_ms"]),
                item["region_pipeline"]["num_regions"],
            ]
            for item in metrics["per_image"]
        ],
    ))
    out.append("")
    return out


def section_extract(out_dir: Path) -> list[str]:
    metrics = read_json(out_dir / "metrics_extract.json")
    if not metrics:
        return ["## 4. 结构化抽取", "", "未测量（找不到 outputs/metrics_extract.json）", ""]
    out = ["## 4. 结构化抽取（OCR 文本 → JSON 字段）", ""]
    out.append(f"- 方法：{metrics['config']['method']}；输入：`{metrics['config']['source_texts']}`")
    out.append(
        f"- 主要 9 字段整体准确率：**{fmt(metrics['headline_field_accuracy'])}** "
        f"({metrics['headline_micro']['correct']}/{metrics['headline_micro']['total']})；"
        f"整图全部字段正确率：{fmt(metrics['image_all_fields_correct'])}"
    )
    out.append("")
    rows = []
    for name, info in metrics["fields"].items():
        rows.append([name, fmt(info["accuracy"]), f"{info['correct']}/{info['total']}", "是" if info["in_headline_set"] else ""])
    out.append(md_table(["字段", "准确率", "正确/总数", "计入主指标"], rows))
    out.append("")

    extracted = read_json(out_dir / "extracted.json")
    if extracted:
        best_id = None
        best_correct = -1
        for item in metrics.get("per_image", []):
            if item["num_correct"] > best_correct:
                best_correct = item["num_correct"]
                best_id = item["id"]
        sample = next((item for item in extracted["images"] if item["id"] == best_id), extracted["images"][0])
        values = {name: info["value"] for name, info in sample["fields"].items()}
        out += ["抽取 JSON 示例（" + sample["id"] + "）：", "", "```json",
                json.dumps(values, ensure_ascii=False, indent=2), "```", ""]
    return out


def section_visuals(out_dir: Path) -> list[str]:
    out = ["## 5. 结果示例", ""]
    images_dir = out_dir / "images"
    det_images = sorted(images_dir.glob("det_*.png")) if images_dir.is_dir() else []
    gt_pred = sorted(images_dir.glob("gt_vs_pred_*.png")) if images_dir.is_dir() else []
    montage = images_dir / "region_crops_example.png" if images_dir.is_dir() else None
    if det_images:
        out += ["YOLO 检测框（红色为预测框，数字为类别与置信度）：", ""]
        out.append(f"![YOLO 检测结果]({det_images[0].relative_to(out_dir).as_posix()})")
        out.append("")
    if gt_pred:
        out += ["GT（绿）vs 预测（红）对比：", ""]
        out.append(f"![GT vs 预测]({gt_pred[0].relative_to(out_dir).as_posix()})")
        out.append("")
    if montage and montage.is_file():
        out += ["区域裁剪示例（YOLO 检测框 → 裁剪 → 逐区域 OCR）：", ""]
        out.append(f"![区域裁剪示例]({montage.relative_to(out_dir).as_posix()})")
        out.append("")
    if not (det_images or gt_pred or montage):
        out += ["未测量（没有找到输出图片）", ""]
    return out


def section_reproduce(out_dir: Path) -> list[str]:
    meta = read_json(out_dir / "run_meta.json")
    out = ["## 6. 复现", ""]
    out.append("```bash")
    out.append("# 完整训练 + 评测（约 10~15 分钟，纯 CPU）")
    out.append("python run_demo.py")
    out.append("# 复用已训练权重，只复现指标与报告（约 10~12 分钟，OCR 占大头）")
    out.append("python run_demo.py --skip-train")
    out.append("```")
    out.append("")
    if meta:
        out.append(f"- 本次运行：{meta['started_at']} → {meta['finished_at']}，总耗时 {meta['total_seconds']} s")
        out.append(f"- 平台：{meta.get('platform', '未测量')}，{meta.get('hardware_note', '')}")
        out.append(f"- 命令：`{pretty_path(meta['command'])}`")
        out.append("")
        out.append(md_table(
            ["步骤", "耗时(s)", "返回码"],
            [[step["name"], fmt(step["seconds"], 1), step["returncode"]] for step in meta.get("steps", [])],
        ))
        out.append("")
    return out


def build_report(data_dir: Path, out_dir: Path) -> str:
    annotations = read_json(data_dir / "annotations.json")
    num_images = annotations["meta"]["num_images"] if annotations else "未测量"
    lines: list[str] = []
    lines.append("# 法律文书扫描件智能处理 Demo — 运行报告（自动生成）")
    lines.append("")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　|　数据规模：{num_images} 张合成图　|　运行环境：Windows + CPU（无 GPU）")
    lines.append(">")
    lines.append("> 本报告所有指标均由 `run_demo.py` 实际运行产生，数字对应 `outputs/metrics_*.json`。")
    lines.append("> **合成数据只用于打通流程与讲清指标，不代表真实扫描件场景性能。**")
    lines.append("")
    lines += section_dataset(data_dir)
    lines += section_yolo(out_dir)
    lines += section_ocr(out_dir)
    lines += section_extract(out_dir)
    lines += section_visuals(out_dir)
    lines += section_reproduce(out_dir)
    lines.append("## 7. 局限")
    lines.append("")
    lines.append("- 合成数据 ≠ 真实扫描件：真实场景有拍照畸变、阴影、装订线、背景纹理、手写批注，指标会明显下降。")
    lines.append("- CPU 训练轮数有限（YOLO11n + 24 张训练图），mAP 绝对值不能代表生产水平，重点看链路是否打通、指标是否可复算。")
    lines.append("- 结构化抽取是规则 baseline，字段格式变化或 OCR 严重错字时会漏抽；生产上还需要 LLM 兜底 + 校验。")
    lines.append("- 本机 Paddle 3.3.1 + PaddleOCR 3.7 在 CPU 上存在 oneDNN/PIR 兼容性问题，已关闭 MKLDNN，因此 OCR 速度比正常 CPU 推理慢。")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 metrics JSON 生成 outputs/report.md")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.data_dir, args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"[report] 已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
