"""一条命令跑通「法律文书扫描件智能处理」demo。

流程：数据生成 → YOLO 训练/推理 → YOLO 指标 → OCR 指标（整图 vs 分区域）→
结构化抽取 → 生成 outputs/report.md。

必须用 **vision 环境**的解释器启动（它负责调度两个环境）::

    D:\\Vibe Coding Projects\\.venv-vision\\Scripts\\python.exe run_demo.py

    # 已经训练过、只想快速复现指标和报告
    D:\\Vibe Coding Projects\\.venv-vision\\Scripts\\python.exe run_demo.py --skip-train

为什么需要两个环境：ultralytics/torch 在 .venv-vision，paddlepaddle/paddleocr 在
.venv-paddle（两者依赖冲突，分开装更稳）。本脚本用 subprocess 分别调用对应解释器。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
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

PROJECT_ROOT = Path(__file__).resolve().parent
VISION_PYTHON = Path(sys.executable)
OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def find_paddle_python(explicit: str | None) -> Path:
    """定位 .venv-paddle 解释器：命令行 > 环境变量 > 兄弟目录默认位置。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PADDLE_PYTHON"):
        candidates.append(Path(os.environ["PADDLE_PYTHON"]))
    candidates.append(PROJECT_ROOT.parent / ".venv-paddle" / "Scripts" / "python.exe")
    candidates.append(Path("D:/Vibe Coding Projects/.venv-paddle/Scripts/python.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "找不到 .venv-paddle 的 python.exe，请用 --paddle-python 指定。尝试过: "
        + ", ".join(str(c) for c in candidates)
    )


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {text}")
    print("=" * 72, flush=True)


def run_step(
    name: str,
    cmd: Sequence[str],
    *,
    cwd: Path,
    steps: list[dict[str, Any]],
) -> None:
    """执行一个子步骤，记录耗时；失败时抛出带上下文的异常。"""
    print(f"\n[run] 步骤: {name}")
    print("[run] 命令: " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.perf_counter()
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    seconds = time.perf_counter() - t0
    steps.append(
        {
            "name": name,
            "cmd": [str(c) for c in cmd],
            "seconds": round(seconds, 1),
            "returncode": result.returncode,
        }
    )
    print(f"[run] 步骤完成: {name} ({seconds:.1f}s, returncode={result.returncode})", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"步骤「{name}」失败，returncode={result.returncode}，命令: {cmd}")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def print_final_summary() -> None:
    """从 metrics 文件里读取关键数字打印，方便一眼看到真实结果。"""
    yolo = read_json(OUT_DIR / "metrics_yolo.json")
    ocr = read_json(OUT_DIR / "metrics_ocr.json")
    ext = read_json(OUT_DIR / "metrics_extract.json")
    print("\n" + "=" * 72)
    print("  真实指标汇总（详见 outputs/*.json 与 outputs/report.md）")
    print("=" * 72)
    if yolo:
        official = yolo["ultralytics_val"]
        manual = yolo["manual_iou_eval"]
        print(f"  YOLO mAP@0.5      : {official['mAP50']:.4f}（官方） / {manual['mAP50']:.4f}（手写 IoU 匹配）")
        print(f"  YOLO mAP@0.5:0.95 : {official['mAP50_95']:.4f}（官方） / {manual['mAP50_95']:.4f}（手写）")
        print(f"  YOLO P / R        : {official['precision']:.4f} / {official['recall']:.4f}")
        speed = yolo["speed"]
        print(f"  YOLO 单图耗时     : 中位 {speed['median_ms']:.1f} ms，约 {speed['fps_from_median']:.1f} FPS（CPU）")
    if ocr:
        full = ocr["full_image"]
        region = ocr["region_pipeline"]
        cmp = ocr["comparison"]
        print(f"  OCR CER 整图      : {full['cer_overall']:.4f}")
        print(f"  OCR CER 分区域    : {region['cer_overall']:.4f}（相对下降 {cmp['cer_relative_drop_pct']:.1f}%）")
        print(f"  OCR 整图耗时      : {full['avg_time_ms']:.0f} ms/图")
        print(f"  分区域 OCR 耗时   : {region['avg_ocr_time_ms']:.0f} ms/图 + 检测 {region['avg_detection_ms'] or 0:.0f} ms")
    if ext:
        print(f"  字段抽取准确率    : {ext['headline_field_accuracy']:.4f} "
              f"({ext['headline_micro']['correct']}/{ext['headline_micro']['total']})")
    print("=" * 72)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="法律文书扫描件智能处理 demo 一键运行")
    parser.add_argument("--skip-train", action="store_true", help="复用已训练的 models/best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=32)
    parser.add_argument("--num-train", type=int, default=24)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--regenerate-data", action="store_true", help="强制重新生成数据集")
    parser.add_argument("--paddle-python", default=None, help=".venv-paddle 解释器路径（自动探测）")
    parser.add_argument("--det-model", default="PP-OCRv5_mobile_det")
    parser.add_argument("--rec-model", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--enable-mkldnn", action="store_true", help="PaddleOCR 默认关闭 MKLDNN（兼容性）")
    parser.add_argument("--split", default="val")
    parser.add_argument("--speed-runs", type=int, default=15)
    parser.add_argument("--save-meta-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.perf_counter()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paddle_python = find_paddle_python(args.paddle_python)

    banner("0/5 环境检查")
    print(f"  项目目录    : {PROJECT_ROOT}")
    print(f"  vision 解释器: {VISION_PYTHON}")
    print(f"  paddle 解释器: {paddle_python}")
    if args.skip_train and not (PROJECT_ROOT / "models" / "best.pt").is_file():
        raise FileNotFoundError(
            "指定了 --skip-train 但 models/best.pt 不存在。请先不带 --skip-train 跑一次训练。"
        )

    steps: list[dict[str, Any]] = []

    banner("1/5 生成合成数据集（精确 bbox + 全文 + 字段 GT）")
    annotations_path = DATA_DIR / "annotations.json"
    if args.regenerate_data or not annotations_path.is_file():
        run_step(
            "make_dataset",
            [
                VISION_PYTHON,
                PROJECT_ROOT / "scripts" / "make_dataset.py",
                "--out",
                DATA_DIR,
                "--num",
                args.num_images,
                "--num-train",
                args.num_train,
                "--seed",
                args.dataset_seed,
            ],
            cwd=PROJECT_ROOT,
            steps=steps,
        )
    else:
        print(f"[run] 复用已有数据集 {annotations_path}（--regenerate-data 可强制重建）")

    banner("2/5 YOLO 版面检测：训练 / 指标 / 推理")
    yolo_cmd: list[Any] = [
        VISION_PYTHON,
        PROJECT_ROOT / "src" / "yolo_detect.py",
        "--stage",
        "all",
        "--data-dir",
        DATA_DIR,
        "--out",
        OUT_DIR,
        "--epochs",
        args.epochs,
        "--imgsz",
        args.imgsz,
        "--batch",
        args.batch,
        "--seed",
        args.seed,
        "--split",
        args.split,
        "--speed-runs",
        args.speed_runs,
    ]
    if args.skip_train:
        yolo_cmd.append("--skip-train")
    run_step("yolo_detect", yolo_cmd, cwd=PROJECT_ROOT, steps=steps)

    banner("3/5 PaddleOCR：整图 vs 区域裁剪，CER 评测（paddle 环境）")
    ocr_cmd: list[Any] = [
        paddle_python,
        PROJECT_ROOT / "src" / "ocr_paddle.py",
        "--data-dir",
        DATA_DIR,
        "--out",
        OUT_DIR,
        "--split",
        args.split,
        "--det-model",
        args.det_model,
        "--rec-model",
        args.rec_model,
    ]
    if args.enable_mkldnn:
        ocr_cmd.append("--enable-mkldnn")
    run_step("ocr_paddle", ocr_cmd, cwd=PROJECT_ROOT, steps=steps)

    banner("4/5 结构化抽取（规则）：OCR 文本 → JSON 字段")
    run_step(
        "extract",
        [
            VISION_PYTHON,
            PROJECT_ROOT / "src" / "extract.py",
            "--data-dir",
            DATA_DIR,
            "--out",
            OUT_DIR,
            "--split",
            args.split,
        ],
        cwd=PROJECT_ROOT,
        steps=steps,
    )

    banner("5/5 汇总生成 outputs/report.md")
    # 报告会读取 run_meta.json 里的运行信息，所以先写一份（报告步骤本身的耗时稍后补上）
    meta: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "command": " ".join([str(VISION_PYTHON), *sys.argv]),
        "steps": list(steps),
        "python": {"vision": str(VISION_PYTHON), "paddle": str(paddle_python)},
        "platform": platform.platform(),
        "hardware_note": "CPU only（无 GPU）",
        "skip_train": bool(args.skip_train),
        "note": "报告生成步骤的耗时在报告生成后追加。",
    }
    (OUT_DIR / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_step(
        "make_report",
        [
            VISION_PYTHON,
            PROJECT_ROOT / "scripts" / "make_report.py",
            "--data-dir",
            DATA_DIR,
            "--out",
            OUT_DIR,
        ],
        cwd=PROJECT_ROOT,
        steps=steps,
    )

    total_seconds = time.perf_counter() - t_start
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_seconds"] = round(total_seconds, 1)
    meta["steps"] = steps
    meta.pop("note", None)
    (OUT_DIR / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_final_summary()
    print(f"\n[run] 全部完成，总耗时 {total_seconds:.1f}s")
    print(f"[run] 报告: {OUT_DIR / 'report.md'}")
    print(f"[run] 运行元数据: {OUT_DIR / 'run_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
