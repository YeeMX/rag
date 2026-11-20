"""方案 A（原生 function calling 循环）测试：正常链路、未知工具、坏 JSON、轮数上限、模型异常。"""

from __future__ import annotations

import pytest

from labor_agent.function_calling import run_function_calling
from labor_agent.llm import LLMError, LLMResponse, MockLLM
from support import AlwaysToolLLM, ScriptedLLM, tool_call_response

QUESTION = "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？依据哪一条？"


def test_mock_end_to_end_calls_two_tools() -> None:
    result = run_function_calling(QUESTION, MockLLM())
    assert result.mock is True
    assert result.error is None
    assert result.used_tools == ["calc_severance", "search_labor_law"]
    assert result.iterations == 2
    assert "96000.00" in result.answer
    assert "MOCK" in result.answer
    # trace 必须可回放：start → planner → tool ×2 → planner → answer
    assert [event["type"] for event in result.trace] == [
        "start",
        "planner",
        "tool",
        "tool",
        "planner",
        "answer",
    ]
    tool_event = result.trace[2]
    assert tool_event["detail"]["name"] == "calc_severance"
    assert tool_event["detail"]["ok"] is True
    assert tool_event["detail"]["result"]["total"] == 96000.0


def test_trace_contains_arguments_for_interview_demo() -> None:
    result = run_function_calling(QUESTION, MockLLM())
    planner_event = result.trace[1]
    tool_calls = planner_event["detail"]["tool_calls"]
    assert tool_calls[0]["name"] == "calc_severance"
    assert tool_calls[0]["arguments"]["scenario"] == "unlawful_termination"
    assert tool_calls[0]["arguments"]["monthly_salary"] == 12000.0


def test_unknown_tool_is_reported_to_model_and_loop_continues() -> None:
    llm = ScriptedLLM(
        [
            tool_call_response("delete_database", "{}"),
            LLMResponse(content="最终答案：未知工具已忽略。", mock=True, provider="scripted"),
        ]
    )
    result = run_function_calling("测试未知工具", llm, max_steps=4)
    assert result.error is None
    assert result.tool_results[0]["ok"] is False
    assert result.tool_results[0]["error"]["code"] == "unknown_tool"
    assert result.answer == "最终答案：未知工具已忽略。"
    # 错误通过 tool 消息回传给模型，而不是抛异常/中断
    tool_message = next(m for m in result.messages if m["role"] == "tool")
    assert "unknown_tool" in tool_message["content"]


def test_bad_json_arguments_are_reported_as_structured_error() -> None:
    llm = ScriptedLLM(
        [
            tool_call_response("calc_severance", '{"scenario": "overtime", 坏'),
            LLMResponse(content="参数有误，已记录。", mock=True, provider="scripted"),
        ]
    )
    result = run_function_calling("测试坏 JSON", llm, max_steps=4)
    assert result.tool_results[0]["ok"] is False
    assert result.tool_results[0]["error"]["code"] == "invalid_arguments_json"
    assert result.answer == "参数有误，已记录。"


def test_max_steps_exhaustion_uses_deterministic_fallback() -> None:
    llm = AlwaysToolLLM()
    result = run_function_calling("测试轮数上限", llm, max_steps=2)
    assert result.iterations == 2
    assert len(result.tool_calls) == 2
    assert result.forced is True
    assert "系统兜底" in result.answer
    assert result.error == "max_steps_exceeded"
    error_events = [event for event in result.trace if event["type"] == "error"]
    assert any(event["detail"].get("code") == "max_steps_exceeded" for event in error_events)


def test_model_error_is_handled_with_fallback_answer() -> None:
    llm = ScriptedLLM([LLMError("boom")])
    result = run_function_calling("测试模型异常", llm, max_steps=2)
    assert result.error is not None and result.error.startswith("model_error")
    assert result.forced is True
    assert "系统兜底" in result.answer
    assert any(event["type"] == "error" for event in result.trace)


def test_empty_question_raises_value_error() -> None:
    with pytest.raises(ValueError):
        run_function_calling("   ", MockLLM())


def test_invalid_max_steps_raises_value_error() -> None:
    with pytest.raises(ValueError):
        run_function_calling("问题", MockLLM(), max_steps=0)


def test_mock_answer_uses_real_tool_numbers_not_hallucination() -> None:
    result = run_function_calling(
        "我工作日加班10小时、休息日加班8小时、法定节假日加班4小时，月薪12000，加班费多少？",
        MockLLM(),
    )
    assert "2965.52" in result.answer
    assert "MOCK" in result.answer
