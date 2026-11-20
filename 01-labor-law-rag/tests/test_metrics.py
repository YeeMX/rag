"""评测指标单元测试（手算验证，防止指标实现悄悄出错）。"""

from __future__ import annotations

import math

import pytest

from lrs.metrics import (
    dedupe_by_article,
    evaluate_rankings,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

REL = {"a", "b", "c"}


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "x", "y"], REL) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["x", "y", "a"], REL) == pytest.approx(1 / 3)


def test_reciprocal_rank_none():
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], REL, k=1) == 0.0
    assert hit_at_k(["x", "a"], REL, k=2) == 1.0


def test_recall_at_k_partial():
    # top-2 只命中 a，b 未命中 => 1/3
    assert recall_at_k(["a", "x"], REL, k=2) == pytest.approx(1 / 3)


def test_dedupe_keeps_first_rank():
    assert dedupe_by_article(["lcl:1", "lcl:1", "lcl:2"]) == ["lcl:1", "lcl:2"]


def test_duplicate_chunks_do_not_inflate_recall():
    # 同一法条两个 chunk 只算一次
    ranked = ["lcl:1#0", "lcl:1#1"]
    # 注意：这里的 id 不是真实 article_id，仅验证去重逻辑
    assert recall_at_k(["lcl:1", "lcl:1"], {"lcl:1", "lcl:2"}, k=5) == 0.5


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], REL, k=3) == pytest.approx(1.0)


def test_ndcg_accounts_for_position():
    # 相关集只含 a：完美排第一为 1.0，排到第三则 DCG 折损
    assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "a"], {"a"}, k=3) == pytest.approx(1 / math.log2(4))


def test_evaluate_rankings_aggregate():
    cases = [
        {"id": "q1", "question": "q1", "relevant": ["a"]},
        {"id": "q2", "question": "q2", "relevant": ["b", "c"]},
    ]
    rankings = [["a", "x"], ["x", "b"]]
    m = evaluate_rankings(cases, rankings)
    assert m["n"] == 2
    assert m["mrr"] == pytest.approx(0.75)       # 两题第一命中分别在第1、第2位 => (1 + 0.5)/2
    assert m["hit@1"] == pytest.approx(0.5)
    assert m["recall@5"] == pytest.approx((1.0 + 0.5) / 2)
    assert len(m["per_case"]) == 2


def test_evaluate_rankings_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_rankings([{"id": "q", "relevant": ["a"]}], [[], []])
