"""校验并生成评测基准文件。

输入：data/benchmark_seed.py（人工编写的 50 条场景）
输出：
    data/benchmark.jsonl   每行一个评测样例（供 evaluate.py 使用）
    data/benchmark_stats.json  分布统计（README 里引用）

校验内容：
- id 唯一、条号真实存在、relevant 非空、hard_negatives 与 relevant 不重叠；
- 统计每题 gold 条数（多意图题 gold 更多）、topic 分布、平均问题长度。

用法：
    python data/build_benchmark.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_JSONL = Path(__file__).resolve().parent / "benchmark.jsonl"
OUT_STATS = Path(__file__).resolve().parent / "benchmark_stats.json"


def main() -> int:
    from benchmark_seed import CASES, validate

    problems = validate()
    if problems:
        print("基准集校验失败：")
        for p in problems:
            print("  -", p)
        return 1

    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for case in CASES:
            row = {
                "id": case["id"],
                "question": case["question"],
                "topic": case["topic"],
                "relevant": case["relevant"],
                "hard_negatives": case["hard_negatives"],
                "note": case.get("note", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gold_counts = [len(c["relevant"]) for c in CASES]
    topics = Counter(c["topic"] for c in CASES)
    stats = {
        "cases": len(CASES),
        "topics": dict(sorted(topics.items(), key=lambda kv: -kv[1])),
        "gold_articles_total": sum(gold_counts),
        "gold_per_case": {
            "min": min(gold_counts), "max": max(gold_counts),
            "avg": round(sum(gold_counts) / len(gold_counts), 2),
        },
        "multi_gold_cases": sum(1 for n in gold_counts if n > 1),
        "hard_negatives_total": sum(len(c["hard_negatives"]) for c in CASES),
        "avg_question_chars": round(sum(len(c["question"]) for c in CASES) / len(CASES), 1),
        "source": "data/benchmark_seed.py（人工编写，条号按 shared/corpus 四部法律全文核对）",
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"已生成 {OUT_JSONL}（{len(CASES)} 条）与 {OUT_STATS.name}")
    print(f"  gold 法条共 {stats['gold_articles_total']} 条，"
          f"平均每题 {stats['gold_per_case']['avg']} 条；多 gold 题 {stats['multi_gold_cases']} 道")
    print(f"  hard negative 共 {stats['hard_negatives_total']} 条")
    print("  topic 分布：", dict(topics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
