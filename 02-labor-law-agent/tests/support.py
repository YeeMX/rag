"""测试替身：用于驱动主循环异常分支的「脚本化 LLM」。

这些类不是生产代码，只放在 tests/ 下，避免污染 labor_agent 包。
"""

from __future__ import annotations

from typing import Any, Sequence

from labor_agent.llm import BaseLLM, LLMResponse, ToolCall


class ScriptedLLM(BaseLLM):
    """按脚本顺序返回预设回复；脚本耗尽后返回一句兜底文本。"""

    name = "scripted"

    def __init__(self, responses: Sequence[Any], *, mock: bool = True) -> None:
        self.responses = list(responses)
        self._mock = mock
        self.calls: list[dict[str, Any]] = []

    @property
    def is_mock(self) -> bool:
        return self._mock

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return LLMResponse(content="（脚本结束）默认最终回答", mock=self._mock, provider=self.name)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AlwaysToolLLM(BaseLLM):
    """永远要求调用工具：用于测试轮数上限 / reflect 分支 / 兜底逻辑。"""

    name = "always-tool"

    def __init__(self, tool_name: str = "search_labor_law", arguments_json: str = '{"query": "经济补偿"}') -> None:
        self.tool_name = tool_name
        self.arguments_json = arguments_json
        self.calls = 0

    @property
    def is_mock(self) -> bool:
        return True

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="我还想再查一次。",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.calls}",
                    name=self.tool_name,
                    arguments_json=self.arguments_json,
                )
            ],
            mock=True,
            provider=self.name,
        )


def tool_call_response(
    name: str,
    arguments_json: str,
    *,
    call_id: str = "call_1",
    content: str | None = None,
) -> LLMResponse:
    """构造一个只包含一次工具调用的 Mock 回复。"""
    return LLMResponse(
        content=content,
        tool_calls=[ToolCall(id=call_id, name=name, arguments_json=arguments_json)],
        mock=True,
        provider="scripted",
    )
