"""结构化抽取：把 OCR 文本 + YOLO 区域类型 → 结构化 JSON 字段。

这是「视觉 → 文本 → 结构化」链路的最后一步：用规则从 OCR 文本里抽取
劳动合同关键信息（合同期限、月工资、试用期、甲方/乙方、签订日期等），
并与合成数据集自带的字段 ground truth 对比，计算字段级准确率。

用法（在 03-vision-ocr-demo 目录下，普通解释器即可，不依赖 paddle/torch）::

    python src/extract.py --data-dir data --out outputs

输入：
    outputs/ocr_texts.json   区域 OCR 结果（含区域类别，来自 src/ocr_paddle.py）
    data/annotations.json    字段 ground truth（仅用于评测，不参与抽取）

输出：
    outputs/extracted.json       每张图抽取出的结构化字段 + 来源区域
    outputs/metrics_extract.json 字段级准确率

为什么用规则而不是直接上 LLM（面试可讲）：
- 字段格式固定、可解释、零延迟、零成本，适合先做 baseline；
- 规则抽取的失败样例可以直接暴露 OCR 错误，便于定位问题；
- 后续可把规则结果作为 few-shot 示例或校验器，交给 LLM 做兜底。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 主要评测字段（与 make_dataset.py 的 fields 对齐）
HEADLINE_FIELDS: list[str] = [
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
#: 附加字段（也抽，也评测，但不算进主要字段准确率）
EXTRA_FIELDS: list[str] = ["probation_salary", "stamp_company"]

DATE_RE = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
#: 兼容 2024-03-15 / 2024/3/5 这类已规范化的日期写法
ISO_DATE_RE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_COMPACT_WS = re.compile(r"[\s\u3000]+")


def compact(line: str) -> str:
    """去掉空白，便于对 OCR 可能插入的空格不敏感地做正则匹配。"""
    return _COMPACT_WS.sub("", line)


def normalize_date(value: str | None) -> str | None:
    """把日期统一成 ``YYYY-MM-DD``。

    支持「2024年03月15日」「2024年3月5日」「2024-03-15」「2024/3/5」；
    无法解析时返回 None。
    """
    if not value:
        return None
    text = compact(value)
    match = re.search(DATE_RE, text)
    if match:
        year, month, day = (int(g) for g in match.groups())
    else:
        match = ISO_DATE_RE.search(text)
        if not match:
            return None
        year, month, day = (int(g) for g in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_text_field(value: str | None) -> str | None:
    """文本字段比对前的规范化：去空白、去首尾常见标点。"""
    if value is None:
        return None
    cleaned = compact(str(value)).strip("。，,；;:：、")
    return cleaned or None


def normalize_int(value: Any) -> int | None:
    """把 "18,000" / "18000元" / 18000 统一成 int；失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    cleaned = re.sub(r"[,\s]", "", str(value))
    match = re.search(r"\d+", cleaned)
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# 规则抽取
# ---------------------------------------------------------------------------


def _iter_lines(regions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """把区域 OCR 结果摊平成「行」列表，保留来源区域信息。"""
    rows: list[dict[str, Any]] = []
    for region in regions:
        for line in region.get("texts", []):
            raw = str(line)
            if not raw.strip():
                continue
            rows.append(
                {
                    "text": raw,
                    "compact": compact(raw),
                    "class_name": region.get("class_name"),
                    "rank": region.get("rank"),
                    "conf": region.get("conf"),
                }
            )
    return rows


def _search(
    rows: Sequence[dict[str, Any]],
    pattern: str,
    *,
    exclude: str | None = None,
) -> tuple[re.Match[str], dict[str, Any]] | None:
    """在行列表中按顺序找第一个匹配，可用 ``exclude`` 排掉某些行。"""
    regex = re.compile(pattern)
    for row in rows:
        if exclude and exclude in row["compact"]:
            continue
        match = regex.search(row["compact"])
        if match:
            return match, row
    return None


def _joined_region_text(region: dict[str, Any]) -> str:
    """把一个区域内所有 OCR 行无空白地拼起来（用于跨行字段，如被换行拆开的日期区间）。"""
    return "".join(compact(line) for line in region.get("texts", []))


def _search_joined(
    regions: Sequence[dict[str, Any]], pattern: str
) -> tuple[re.Match[str], dict[str, Any]] | None:
    """在整个区域的拼接文本里搜索（解决 OCR 把一句话拆成多行导致匹配失败的问题）。"""
    regex = re.compile(pattern)
    for region in regions:
        joined = _joined_region_text(region)
        match = regex.search(joined)
        if match:
            return match, region
    return None


def _source_of_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "class_name": region.get("class_name"),
        "rank": region.get("rank"),
        "conf": region.get("conf"),
        "line": "\n".join(region.get("texts", [])),
    }


def _first_capture(rows: Sequence[dict[str, Any]], pattern: str, *, exclude: str | None = None) -> dict[str, Any] | None:
    found = _search(rows, pattern, exclude=exclude)
    if not found:
        return None
    match, row = found
    return {
        "value": match.group(1) if match.groups() else match.group(0),
        "source": {
            "class_name": row["class_name"],
            "rank": row["rank"],
            "conf": row["conf"],
            "line": row["text"],
        },
    }


def extract_fields(regions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """从区域 OCR 结果中抽取字段。

    Args:
        regions: ``ocr_texts.json`` 中单张图的 ``region_pipeline`` 列表，
            每项含 class_name / texts / rank / conf。

    Returns:
        {字段名: {"value": 值或 None, "source": 来源行/区域}}，缺失字段 value=None。
    """
    rows = _iter_lines(regions)
    fields: dict[str, dict[str, Any]] = {}

    def put(name: str, found: dict[str, Any] | None, *, normalize: str = "text") -> None:
        value = found["value"] if found else None
        if value is not None:
            if normalize == "text":
                value = normalize_text_field(value)
            elif normalize == "int":
                value = normalize_int(value)
            elif normalize == "date":
                value = normalize_date(value)
        fields[name] = {"value": value, "source": found["source"] if found else None}

    put("party_a", _first_capture(rows, r"甲方(?:（用人单位）|\(用人单位\)|单位)?[：:]?(.+?)(?:住所|法定代表人|$)"))
    put("party_b", _first_capture(rows, r"乙方(?:（劳动者）|\(劳动者\)|劳动者)?[：:]?(.+?)(?:身份证|住址|住所|$)"))
    put("contract_years", _first_capture(rows, r"合同期限为(\d+)年"), normalize="int")
    put("monthly_salary", _first_capture(rows, r"月工资为人民币(\d+)元", exclude="试用期"), normalize="int")
    put("probation_months", _first_capture(rows, r"试用期为(\d+)个月"), normalize="int")
    put("probation_salary", _first_capture(rows, r"试用期月工资为人民币(\d+)元"), normalize="int")
    put("job", _first_capture(rows, r"从事(.+?)岗位"))

    # 签订日期：先按行找；OCR 可能把「日期」和「签订日期：」拆到两行，失败再按区域拼接口径找
    sign_date_found = _first_capture(rows, r"签订日期[：:]?(.+)$")
    if sign_date_found is None:
        joined = _search_joined(regions, r"签订日期[：:]?(.+)$")
        if joined is not None:
            match, region = joined
            sign_date_found = {
                "value": match.group(1),
                "source": _source_of_region(region),
            }
    put("sign_date", sign_date_found, normalize="date")

    # 合同起止日期：OCR 常把长条款折行（如「…起至2023年02」+「月19日止」），
    # 因此在该区域的拼接文本上匹配；起始日和结束日分别匹配，避免一端识别错误连累另一端。
    start_found = _search_joined(regions, rf"自{DATE_RE}")
    end_found = _search_joined(regions, rf"至{DATE_RE}")
    fields["contract_start"] = {
        "value": normalize_date(start_found[0].group(0)) if start_found else None,
        "source": _source_of_region(start_found[1]) if start_found else None,
    }
    fields["contract_end"] = {
        "value": normalize_date(end_found[0].group(0)) if end_found else None,
        "source": _source_of_region(end_found[1]) if end_found else None,
    }

    # 印章区域文本第一行作为公司简称
    stamp_line = next(
        (row for row in rows if row["class_name"] == "stamp" and row["compact"]),
        None,
    )
    fields["stamp_company"] = {
        "value": normalize_text_field(stamp_line["text"]) if stamp_line else None,
        "source": (
            {
                "class_name": stamp_line["class_name"],
                "rank": stamp_line["rank"],
                "conf": stamp_line["conf"],
                "line": stamp_line["text"],
            }
            if stamp_line
            else None
        ),
    }

    # 文档标题（来自 title 区域）
    title_line = next(
        (row for row in rows if row["class_name"] == "title" and row["compact"]),
        None,
    )
    fields["document_type"] = {
        "value": normalize_text_field(title_line["text"]) if title_line else None,
        "source": (
            {
                "class_name": title_line["class_name"],
                "rank": title_line["rank"],
                "conf": title_line["conf"],
                "line": title_line["text"],
            }
            if title_line
            else None
        ),
    }
    return fields


# ---------------------------------------------------------------------------
# 与 ground truth 对比
# ---------------------------------------------------------------------------


def compare_value(field: str, extracted: Any, truth: Any) -> bool:
    """字段级比较：按字段类型做规范化后比较。"""
    if extracted is None or truth is None:
        return False
    if field in ("contract_years", "monthly_salary", "probation_months", "probation_salary"):
        return normalize_int(extracted) == normalize_int(truth)
    if field in ("contract_start", "contract_end", "sign_date"):
        return normalize_date(str(extracted)) == normalize_date(str(truth))
    return normalize_text_field(str(extracted)) == normalize_text_field(str(truth))


def evaluate_extraction(
    extracted_images: Sequence[dict[str, Any]],
    annotations: dict[str, Any],
) -> dict[str, Any]:
    """计算字段级准确率（按字段类型规范化后比较）。"""
    truth_by_id = {img["id"]: img["fields"] for img in annotations["images"]}
    field_hits = {name: 0 for name in HEADLINE_FIELDS + EXTRA_FIELDS}
    field_total = {name: 0 for name in HEADLINE_FIELDS + EXTRA_FIELDS}
    per_image: list[dict[str, Any]] = []

    for item in extracted_images:
        truth = truth_by_id[item["id"]]
        checks: dict[str, Any] = {}
        headline_ok = 0
        for name in HEADLINE_FIELDS + EXTRA_FIELDS:
            if name not in truth:
                continue
            extracted_value = item["fields"].get(name, {}).get("value")
            ok = compare_value(name, extracted_value, truth[name])
            checks[name] = {
                "extracted": extracted_value,
                "truth": truth[name],
                "correct": ok,
            }
            field_total[name] += 1
            if ok:
                field_hits[name] += 1
                if name in HEADLINE_FIELDS:
                    headline_ok += 1
        per_image.append(
            {
                "id": item["id"],
                "hard_case": item["hard_case"],
                "num_correct": headline_ok,
                "num_fields": len(HEADLINE_FIELDS),
                "checks": checks,
            }
        )

    def accuracy(hits: int, total: int) -> float:
        return round(hits / total, 4) if total else 0.0

    return {
        "fields": {
            name: {
                "accuracy": accuracy(field_hits[name], field_total[name]),
                "correct": field_hits[name],
                "total": field_total[name],
                "in_headline_set": name in HEADLINE_FIELDS,
            }
            for name in HEADLINE_FIELDS + EXTRA_FIELDS
        },
        "headline_field_accuracy": accuracy(
            sum(field_hits[n] for n in HEADLINE_FIELDS),
            sum(field_total[n] for n in HEADLINE_FIELDS),
        ),
        "headline_micro": {
            "correct": sum(field_hits[n] for n in HEADLINE_FIELDS),
            "total": sum(field_total[n] for n in HEADLINE_FIELDS),
        },
        "image_all_fields_correct": accuracy(
            sum(1 for item in per_image if item["num_correct"] == item["num_fields"]),
            len(per_image),
        ),
        "num_images": len(per_image),
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(data_dir: Path, out_dir: Path, texts_path: Path, split: str | None = None) -> dict[str, Any]:
    annotations = json.loads((data_dir / "annotations.json").read_text(encoding="utf-8"))
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    ocr_by_id = {item["id"]: item for item in texts["images"]}

    print(f"[extract] 对 {len(ocr_by_id)} 张图做规则抽取 ...")
    extracted_images: list[dict[str, Any]] = []
    for img in annotations["images"]:
        if split and img["split"] != split:
            continue
        ocr_item = ocr_by_id.get(img["id"])
        if ocr_item is None:
            continue
        fields = extract_fields(ocr_item["region_pipeline"])
        extracted_images.append(
            {
                "id": img["id"],
                "split": img["split"],
                "hard_case": img["hard_case"],
                "fields": fields,
                "ocr_text_preview": ocr_item["region_pipeline_text"][:300],
            }
        )

    metrics = evaluate_extraction(extracted_images, annotations)
    metrics["config"] = {
        "source_texts": str(texts_path),
        "split": split or "all",
        "pipeline": "YOLO 区域裁剪 → PaddleOCR → 规则抽取",
        "method": "正则规则（不依赖 LLM / 第三方 NLP 库）",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extracted.json").write_text(
        json.dumps({"config": metrics["config"], "images": extracted_images}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics_extract.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[extract] 主要 9 字段整体准确率: {metrics['headline_field_accuracy']:.4f} "
          f"({metrics['headline_micro']['correct']}/{metrics['headline_micro']['total']})")
    for name in HEADLINE_FIELDS:
        info = metrics["fields"][name]
        print(f"          {name:18s} {info['accuracy']:.4f} ({info['correct']}/{info['total']})")
    print(f"[extract] 输出: {out_dir / 'extracted.json'}, {out_dir / 'metrics_extract.json'}")
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR 文本 → 劳动合同结构化字段抽取")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--texts", type=Path, default=None, help="默认 outputs/ocr_texts.json")
    parser.add_argument("--split", default="val", help="只对 val 集的图做抽取（与 OCR 评测一致）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    texts_path = args.texts or (args.out / "ocr_texts.json")
    run(args.data_dir, args.out, texts_path, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
