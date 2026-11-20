"""检索与答案评测指标（纯函数，便于单元测试）。

检索指标按 **法条（article_id）** 计算，而不是按 chunk：
一条法条被切成多个 chunk 时，只要任一 chunk 命中就算该法条命中；RRF/重排后同一法条的
多个 chunk 只保留排名最高的一次，避免"重复召回同一条法条"刷高 Hit/Recall。

- MRR@k      ：第一个相关法条排名的倒数均值（衡量"第一名对不对"）
- Hit@k      ：top-k 内至少命中一条相关法条的题目占比（衡量"有没有找到"）
- Recall@k   ：top-k 命中 related id 的比例均值（多意图题更敏感，衡量"找全没"）
- nDCG@k     ：考虑多级相关的排序质量（这里 relevance=1，等价于 DCG/IDCG）
"""

from __future__ import annotations

import math


def dedupe_by_article(article_ids: list[str]) -> list[str]:
    """按出现顺序去重（同一法条多个 chunk 只保留首次出现的排名）。"""
    seen: set[str] = set()
    out: list[str] = []
    for aid in article_ids:
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int | None = None) -> float:
    if not ranked or not relevant:
        return 0.0
    ranked = dedupe_by_article(ranked)
    if k is not None:
        ranked = ranked[:k]
    for rank, aid in enumerate(ranked, start=1):
        if aid in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(dedupe_by_article(ranked)[:k]) & relevant else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    got = set(dedupe_by_article(ranked)[:k]) & relevant
    return len(got) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = dedupe_by_article(ranked)[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    ranked = dedupe_by_article(ranked)[:k]
    if not relevant or not ranked:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, aid in enumerate(ranked, start=1) if aid in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_rankings(
    cases: list[dict],
    rankings: list[list[str]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """对一批题目的排名结果计算聚合指标。

    cases: [{"id", "relevant": [article_id...]}, ...]
    rankings: 与 cases 对齐的 article_id 有序列表
    """
    if len(cases) != len(rankings):
        raise ValueError(f"cases({len(cases)}) 与 rankings({len(rankings)}) 数量不一致")
    n = len(cases)
    if n == 0:
        return {}
    result: dict = {"n": n, "per_case": []}
    for k in ks:
        result[f"mrr@{k}"] = round(
            sum(reciprocal_rank(r, set(c["relevant"]), k) for c, r in zip(cases, rankings)) / n, 4)
        result[f"hit@{k}"] = round(
            sum(hit_at_k(r, set(c["relevant"]), k) for c, r in zip(cases, rankings)) / n, 4)
        result[f"recall@{k}"] = round(
            sum(recall_at_k(r, set(c["relevant"]), k) for c, r in zip(cases, rankings)) / n, 4)
        result[f"ndcg@{k}"] = round(
            sum(ndcg_at_k(r, set(c["relevant"]), k) for c, r in zip(cases, rankings)) / n, 4)
    # 不使用截断的 MRR（等价 mrr@all）
    result["mrr"] = round(
        sum(reciprocal_rank(r, set(c["relevant"])) for c, r in zip(cases, rankings)) / n, 4)
    # 每条题目明细，便于错误分析
    for case, ranked in zip(cases, rankings):
        rel = set(case["relevant"])
        first = next((i for i, a in enumerate(dedupe_by_article(ranked), 1) if a in rel), None)
        result["per_case"].append({
            "id": case["id"],
            "question": case.get("question", ""),
            "first_relevant_rank": first,
            "hit@5": hit_at_k(ranked, rel, 5),
            "recall@5": recall_at_k(ranked, rel, 5),
            "ranked_top10": dedupe_by_article(ranked)[:10],
            "relevant": case["relevant"],
        })
    return result


def format_metrics_row(name: str, m: dict, k: int = 5) -> dict:
    """把一次评测结果压成一行，方便写 Markdown 表格。"""
    return {
        "system": name,
        "MRR": m.get("mrr", 0.0),
        f"MRR@{k}": m.get(f"mrr@{k}", 0.0),
        f"Hit@{k}": m.get(f"hit@{k}", 0.0),
        f"Recall@{k}": m.get(f"recall@{k}", 0.0),
        f"nDCG@{k}": m.get(f"ndcg@{k}", 0.0),
    }
