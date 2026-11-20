"""方案 A：原生 OpenAI function calling 循环（不依赖任何 Agent 框架）。

控制流（面试可讲）：就是一个 ``while`` 循环 + 一个 ``if``：

1. 把 ``messages + tools`` 发给模型；
2. 模型返回 ``tool_calls`` → 执行工具 → 把 tool 结果写回 ``messages`` → 回到第 1 步；
3. 模型返回自然语言 → 结束；
4. 达到 ``max_steps`` 仍没有最终答案 → 强制收尾（禁止再调工具）→ 仍失败则用确定性兜底。

Mock 模式走的是完全相同的循环：``MockLLM`` 也会返回 ``tool_calls``，因此没有 API key
也能演示完整链路。未知工具、坏 JSON 参数等错误由 ``tools.execute_tool_call`` 转成
结构化 tool 结果，模型会看到错误并据此继续。
"""

from __future__ import annotations

import json
from typing import Any

from . import tools
from .llm import BaseLLM, LLMError, build_llm
from .state import (
    DEFAULT_MAX_STEPS,
    AgentResult,
    add_trace,
    assistant_message,
    execute_pending_calls,
    finish,
    new_state,
    pending_calls_from_response,
    request_forced_answer,
    validate_question,
)


def _trace_arguments(arguments_json: str) -> Any:
    try:
        return json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return arguments_json


def run_function_calling(
    question: str,
    llm: BaseLLM | None = None,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """用原生 function calling 循环回答一个问题。

    Args:
        question: 用户问题。
        llm: 模型（默认 :func:`labor_agent.llm.build_llm`，无 key 时为 MockLLM）。
        max_steps: 模型决策轮数上限（每轮可含多个工具调用）。

    Returns:
        统一的 :class:`~labor_agent.state.AgentResult`，``trace`` 可完整回放。
    """
    question = validate_question(question)
    if max_steps < 1:
        raise ValueError("max_steps 必须 >= 1")
    llm = llm or build_llm("mock")
    state = new_state(question, "function_calling")
    state["mock"] = llm.is_mock
    add_trace(
        state,
        "start",
        "开始 function calling 循环",
        framework="function_calling",
        llm=llm.describe(),
        max_steps=max_steps,
    )

    answer: str | None = None
    for _ in range(max_steps):
        try:
            response = llm.complete(state["messages"], tools.TOOL_SCHEMAS)
        except LLMError as exc:
            state["error"] = f"model_error: {exc}"
            add_trace(state, "error", "模型调用失败", code="model_error", message=str(exc))
            return finish(state, None, forced=True)

        state["iterations"] = int(state.get("iterations", 0)) + 1
        add_trace(
            state,
            "planner",
            f"第 {state['iterations']} 轮模型决策",
            mock=response.mock,
            content=response.content,
            tool_calls=[
                {"name": call.name, "arguments": _trace_arguments(call.arguments_json)}
                for call in response.tool_calls
            ],
        )
        state["messages"].append(assistant_message(response))

        if not response.tool_calls:
            answer = response.content
            break

        state["pending_calls"] = pending_calls_from_response(response)
        execute_pending_calls(state)
    else:
        # 循环跑满 max_steps 且最后一轮仍在调工具：强制收尾，避免无限循环。
        add_trace(
            state,
            "error",
            "达到最大轮数，尝试强制收尾",
            code="max_steps_exceeded",
            max_steps=max_steps,
        )
        answer = request_forced_answer(llm, state)
        if answer is None:
            state["error"] = "max_steps_exceeded"
            add_trace(
                state,
                "error",
                "模型在预算内仍未给出最终答案，使用确定性兜底",
                code="max_steps_exceeded",
            )
            return finish(state, None, forced=True)

    return finish(state, answer)
