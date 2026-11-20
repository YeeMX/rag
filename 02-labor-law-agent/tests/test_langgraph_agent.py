"""方案 B（LangGraph StateGraph）测试：图结构、mock 链路、reflect/轮数上限、与方案 A 的一致性。"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from labor_agent.function_calling import run_function_calling  # noqa: E402
from labor_agent.langgraph_agent import (  # noqa: E402
    AVAILABLE_NODES,
    build_graph,
    graph_mermaid,
    is_langgraph_available,
    run_langgraph,
)
from labor_agent.llm import MockLLM  # noqa: E402
from support import AlwaysToolLLM  # noqa: E402

QUESTION = "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？依据哪一条？"


def test_langgraph_is_available_in_this_env() -> None:
    assert is_langgraph_available() is True


def test_graph_has_expected_nodes() -> None:
    graph = build_graph(MockLLM())
    mermaid = graph_mermaid()
    for node in AVAILABLE_NODES:
        assert node in mermaid
    assert "graph TD" in mermaid
    assert graph is not None


def test_run_langgraph_with_mock() -> None:
    result = run_langgraph(QUESTION, MockLLM())
    assert result.framework == "langgraph"
    assert result.mock is True
    assert result.error is None
    assert result.used_tools == ["calc_severance", "search_labor_law"]
    assert "96000.00" in result.answer
    trace_types = [event["type"] for event in result.trace]
    assert trace_types[0] == "start"
    assert "planner" in trace_types
    assert "tool" in trace_types
    assert trace_types[-1] == "answer"
    planner_events = [event for event in result.trace if event["type"] == "planner"]
    assert all(event["detail"]["node"] == "planner" for event in planner_events)


def test_langgraph_budget_exhaustion_hits_reflect_node() -> None:
    result = run_langgraph("测试 reflect 分支", AlwaysToolLLM(), max_steps=2)
    assert result.iterations == 2
    reflect_events = [event for event in result.trace if event["type"] == "reflect"]
    assert len(reflect_events) == 1
    assert reflect_events[0]["detail"]["reflections"] == 1
    assert result.forced is True
    assert "系统兜底" in result.answer


def test_langgraph_and_function_calling_agree_on_amounts() -> None:
    question = (
        "公司经济性裁员，我月薪30000，工作15年，当地社平工资9000，经济补偿怎么算？"
    )
    fc = run_function_calling(question, MockLLM())
    lg = run_langgraph(question, MockLLM())
    fc_calc = next(p for p in fc.tool_results if p["tool"] == "calc_severance")
    lg_calc = next(p for p in lg.tool_results if p["tool"] == "calc_severance")
    assert fc_calc["data"]["total"] == lg_calc["data"]["total"] == 324000.0
    assert fc_calc["data"]["steps"] == lg_calc["data"]["steps"]


def test_langgraph_invalid_question_raises() -> None:
    with pytest.raises(ValueError):
        run_langgraph("  ", MockLLM())
