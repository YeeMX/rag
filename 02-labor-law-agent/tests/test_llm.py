"""LLM 抽象层测试：Mock 的确定性/诚实标注、真实模型响应转换、工厂回退。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from labor_agent import tools
from labor_agent.llm import (
    MOCK_BANNER,
    BaseLLM,
    LLMError,
    LLMResponse,
    LLMUnavailableError,
    MockLLM,
    OpenAICompatLLM,
    ToolCall,
    build_llm,
    has_openai_credentials,
)

QUESTION = "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？依据哪一条？"


def _messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": "你是劳动法助手"},
        {"role": "user", "content": question},
    ]


def test_mock_llm_is_honest_about_being_mock() -> None:
    llm = MockLLM()
    assert isinstance(llm, BaseLLM)
    assert llm.is_mock is True
    response = llm.complete(_messages(QUESTION), tools.TOOL_SCHEMAS)
    assert response.mock is True
    assert response.provider == "mock-rules"
    assert response.content and "MOCK" in response.content
    assert "mock" in llm.describe().lower()


def test_mock_llm_plans_calc_and_legal_search_in_one_round() -> None:
    response = MockLLM().complete(_messages(QUESTION), tools.TOOL_SCHEMAS)
    assert [call.name for call in response.tool_calls] == [
        "calc_severance",
        "search_labor_law",
    ]
    calc_args = json.loads(response.tool_calls[0].arguments_json)
    assert calc_args == {
        "scenario": "unlawful_termination",
        "monthly_salary": 12000.0,
        "years": 3,
        "months": 7,
    }
    search_args = json.loads(response.tool_calls[1].arguments_json)
    assert "违法解除" in search_args["query"]


def test_mock_llm_extracts_scenario_fields() -> None:
    cases = [
        (
            "入职14个月一直没签书面劳动合同，月薪15000，能主张多少二倍工资差额？",
            "double_wage",
            {"months_without_contract": 14},
        ),
        (
            "我工作日加班10小时、休息日加班8小时、法定节假日加班4小时，月薪12000，加班费多少？",
            "overtime",
            {"overtime_hours": {"workday": 10.0, "rest_day": 8.0, "holiday": 4.0}},
        ),
        (
            "公司经济性裁员，我月薪30000，工作15年，当地社平工资9000，经济补偿怎么算？",
            "severance",
            {"city_avg_salary": 9000.0, "years": 15, "months": 0},
        ),
    ]
    for question, scenario, expected in cases:
        response = MockLLM().complete(_messages(question), tools.TOOL_SCHEMAS)
        assert response.tool_calls, question
        assert response.tool_calls[0].name == "calc_severance"
        args = json.loads(response.tool_calls[0].arguments_json)
        assert args["scenario"] == scenario
        for key, value in expected.items():
            assert args[key] == value, (question, key)


def test_mock_llm_plans_verify_citation() -> None:
    question = (
        "帮我核验这条引用是否准确："
        "《劳动合同法》第八十二条说未签书面合同最多支付11个月二倍工资？"
    )
    response = MockLLM().complete(_messages(question), tools.TOOL_SCHEMAS)
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "verify_citation"
    args = json.loads(call.arguments_json)
    assert args["law_id"] == "lcl"
    assert args["article_no"] == "第八十二条"
    assert "11个月" in args["claim"]


def test_mock_llm_falls_back_to_search_when_info_missing() -> None:
    response = MockLLM().complete(_messages("我被辞退了有补偿吗？"), tools.TOOL_SCHEMAS)
    assert [call.name for call in response.tool_calls] == ["search_labor_law"]
    assert "缺少必要信息" in (response.content or "")


def test_mock_llm_handles_small_talk_without_tools() -> None:
    response = MockLLM().complete(_messages("你好"), tools.TOOL_SCHEMAS)
    assert response.tool_calls == []
    assert "离线规则 Mock" in (response.content or "")


def test_mock_llm_composes_answer_from_tool_results() -> None:
    payload = tools.calc_severance(
        {
            "scenario": "unlawful_termination",
            "monthly_salary": 12000,
            "years": 3,
            "months": 7,
        }
    )
    messages = _messages(QUESTION) + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1_1",
                    "type": "function",
                    "function": {"name": "calc_severance", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1_1",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]
    response = MockLLM().complete(messages, tools.TOOL_SCHEMAS)
    assert response.tool_calls == []
    assert response.content and response.content.startswith(MOCK_BANNER)
    assert "96000.00" in response.content
    assert "不是真实大模型的推理输出" in response.content


def test_build_llm_mock_and_auto_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(build_llm("mock"), MockLLM)
    assert isinstance(build_llm(), MockLLM)
    assert isinstance(build_llm("auto"), MockLLM)
    assert has_openai_credentials() is False


def test_build_llm_openai_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMUnavailableError):
        build_llm("openai")


def test_build_llm_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_llm("magic")


def test_build_llm_auto_prefers_real_model_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-only-for-construction")
    llm = build_llm("auto")
    assert llm.is_mock is False
    assert llm.name.startswith("openai-compatible:")


def test_openai_compat_converts_tool_calls_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    llm = OpenAICompatLLM(api_key="sk-test", model="gpt-test")
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_abc",
                        function=SimpleNamespace(
                            name="search_labor_law",
                            arguments='{"query": "经济补偿"}',
                        ),
                    )
                ],
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

    llm._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    response = llm.complete(_messages("问题"), tools.TOOL_SCHEMAS)
    assert captured["model"] == "gpt-test"
    assert captured["tools"] == tools.TOOL_SCHEMAS
    assert captured["tool_choice"] == "auto"
    assert response.mock is False
    assert response.content is None
    assert response.tool_calls == [
        ToolCall(id="call_abc", name="search_labor_law", arguments_json='{"query": "经济补偿"}')
    ]
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_openai_compat_wraps_exceptions_as_llm_error() -> None:
    pytest.importorskip("openai")
    llm = OpenAICompatLLM(api_key="sk-test")

    class FailingCompletions:
        def create(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    llm._client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    with pytest.raises(LLMError, match="network down"):
        llm.complete(_messages("问题"), tools.TOOL_SCHEMAS)


def test_llm_response_wants_tools_property() -> None:
    assert LLMResponse(content="hi").wants_tools is False
    assert LLMResponse(content=None, tool_calls=[ToolCall("1", "x", "{}")]).wants_tools is True
