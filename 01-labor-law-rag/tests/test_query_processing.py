"""查询理解：分解与扩展的单元测试。"""

from __future__ import annotations

from lrs.query_processing import plan_query


def test_single_intent_no_decomposition():
    plan = plan_query("经济补偿怎么计算？")
    assert plan.is_multi_intent is False
    assert plan.sub_queries == []


def test_multi_intent_decomposed():
    q = ("入职8个月没签劳动合同，现在被辞退，"
         "能同时要双倍工资和赔偿金吗？")
    plan = plan_query(q)
    assert plan.is_multi_intent is True
    assert len(plan.sub_queries) >= 2
    # 子查询必须保留关键事实
    joined = "".join(plan.sub_queries)
    assert "8个月" in joined or "双倍" in joined


def test_connector_splits_three_intents():
    q = "试用期6个月、工资打7折、还说试用期不缴社保，这三件事公司都违法吗？"
    plan = plan_query(q)
    assert plan.is_multi_intent
    assert len(plan.sub_queries) >= 2


def test_expansion_maps_colloquial_to_legal_terms():
    plan = plan_query("公司把我开除了，有赔偿吗？")
    assert any("解除劳动合同" in t for t in plan.expansions)
    assert any("经济补偿" in t or "赔偿金" in t for t in plan.expansions)


def test_decomposition_can_be_disabled():
    q = "没签合同还被辞退，能要双倍工资吗？"
    plan = plan_query(q, decompose=False, expand=False)
    assert plan.is_multi_intent is False
    assert plan.sub_queries == []
    assert plan.expansions == []


def test_empty_question_raises():
    import pytest

    with pytest.raises(ValueError):
        plan_query("   ")
