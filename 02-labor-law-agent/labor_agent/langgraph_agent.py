"""方案 B：用 LangGraph ``StateGraph`` 实现 planner → tools → reflect → answer。

与方案 A 的关键差异（面试可讲）：

- 方案 A 用 ``while`` 循环手写控制流；方案 B 把控制流显式建模成**图**：
  节点（planner / tools / reflect / answer）+ 条件边（路由函数）+ 轮数上限。
- 状态结构与工具执行逻辑（``state.py``、``tools.py``）与方案 A **完全复用**，
  所以两种实现回答同一个问题时应得到一致的金额与法条。
- ``reflect`` 节点是 LangGraph 特有的显式分支：当 planner 还想调工具但轮数预算耗尽时，
  先进入 reflect（记录并注入「禁止再调工具」的提示），再走 answer，避免死循环。

``langgraph`` 采用**延迟导入**：未安装时本模块仍可 import，只有真正运行图时才抛
:class:`LangGraphUnavailable` 并给出可操作的提示，不影响 CLI 的其他功能。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from . import tools
from .llm import BaseLLM, LLMError, MockLLM, build_llm
from .state import (
    FORCE_ANSWER_PROMPT,
    AgentResult,
    AgentState,
    add_trace,
    assistant_message,
    execute_pending_calls,
    finish,
    new_state,
    pending_calls_from_response,
    request_forced_answer,
    validate_question,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 4
"""LangGraph 方案默认轮数（图每轮 = planner + tools 两次节点跳转，比方案 A 略保守）。"""

AVAILABLE_NODES: tuple[str, ...] = ("planner", "tools", "reflect", "answer")


class LangGraphUnavailable(RuntimeError):
    """langgraph 未安装或导入失败。"""


def _import_langgraph() -> tuple[Any, Any, Any]:
    """延迟导入 LangGraph；失败时抛出带操作指引的 :class:`LangGraphUnavailable`。"""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise LangGraphUnavailable(
            "未安装 langgraph，无法运行方案 B。\n"
            "解决方式：使用 `--framework function_calling` 运行方案 A（功能等价）；"
            "或安装 langgraph 后重试。\n"
            f"原始错误：{exc}"
        ) from exc
    return StateGraph, START, END


def is_langgraph_available() -> bool:
    """langgraph 是否可导入（CLI / demo 用来自动降级）。"""
    try:
        _import_langgraph()
        return True
    except LangGraphUnavailable:
        return False


def _trace_arguments(arguments_json: str) -> Any:
    try:
        return json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return arguments_json


def build_graph(
    llm: BaseLLM,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    on_node: Callable[[str, AgentState], None] | None = None,
):
    """构建并编译 LangGraph 图。

    节点：
    - ``planner``：调用 LLM，拿到 tool_calls 或自然语言草稿；
    - ``tools``：执行 pending tool calls（复用 ``state.execute_pending_calls``）；
    - ``reflect``：轮数预算耗尽时的反思/止损节点；
    - ``answer``：收尾，写入最终回答。

    条件边：
    - planner → tools（还有预算）/ reflect（预算耗尽但仍要调工具）/ answer（已可回答）；
    - tools → planner（继续）/ reflect（预算耗尽）；
    - reflect → answer。

    Args:
        llm: 模型实例（Mock 或真实模型）。
        max_steps: planner 被调用的最大轮数。
        on_node: 可选的节点回调（测试/调试时观察图执行路径）。
    """
    if max_steps < 1:
        raise ValueError("max_steps 必须 >= 1")
    StateGraph, START, END = _import_langgraph()

    def _notify(node: str, state: AgentState) -> None:
        if on_node is not None:
            on_node(node, state)

    def planner_node(state: AgentState) -> dict[str, Any]:
        try:
            response = llm.complete(state["messages"], tools.TOOL_SCHEMAS)
        except LLMError as exc:
            state["error"] = f"model_error: {exc}"
            state["draft_answer"] = None
            state["pending_calls"] = []
            add_trace(
                state, "error", "planner 节点模型调用失败", node="planner", message=str(exc)
            )
            _notify("planner", state)
            return dict(state)

        iterations = int(state.get("iterations", 0)) + 1
        state["iterations"] = iterations
        state["mock"] = bool(state.get("mock") or response.mock)
        add_trace(
            state,
            "planner",
            f"planner 节点第 {iterations} 轮决策",
            node="planner",
            mock=response.mock,
            content=response.content,
            tool_calls=[
                {"name": call.name, "arguments": _trace_arguments(call.arguments_json)}
                for call in response.tool_calls
            ],
        )
        state["messages"].append(assistant_message(response))
        if response.tool_calls:
            state["pending_calls"] = pending_calls_from_response(response)
            state["draft_answer"] = None
        else:
            state["pending_calls"] = []
            state["draft_answer"] = response.content
        _notify("planner", state)
        return dict(state)

    def tools_node(state: AgentState) -> dict[str, Any]:
        execute_pending_calls(state)
        _notify("tools", state)
        return dict(state)

    def reflect_node(state: AgentState) -> dict[str, Any]:
        state["reflections"] = int(state.get("reflections", 0)) + 1
        state["force_answer"] = True
        add_trace(
            state,
            "reflect",
            "reflect 节点：工具轮数预算耗尽，注入「禁止再调工具」提示后直接作答",
            node="reflect",
            reflections=state["reflections"],
            max_steps=max_steps,
        )
        state["messages"].append({"role": "system", "content": FORCE_ANSWER_PROMPT})
        _notify("reflect", state)
        return dict(state)

    def answer_node(state: AgentState) -> dict[str, Any]:
        answer = state.get("draft_answer")
        forced = bool(state.get("force_answer"))
        if not answer:
            forced = True
            if state.get("error"):
                answer = None  # 模型调用失败：不再重复请求，直接确定性兜底
            else:
                answer = request_forced_answer(llm, state)
            if answer is None:
                add_trace(
                    state,
                    "error",
                    "预算耗尽且模型仍未直接作答，使用确定性兜底",
                    node="answer",
                    code="max_steps_exceeded",
                )
                state["error"] = state.get("error") or "max_steps_exceeded"
        finish(state, answer, forced=forced)
        _notify("answer", state)
        return dict(state)

    def route_after_planner(state: AgentState) -> str:
        if state.get("pending_calls"):
            if int(state.get("iterations", 0)) >= max_steps:
                return "reflect"
            return "tools"
        return "answer"

    def route_after_tools(state: AgentState) -> str:
        if int(state.get("iterations", 0)) >= max_steps:
            return "reflect"
        return "planner"

    def route_after_reflect(state: AgentState) -> str:
        return "answer"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tools", tools_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "reflect": "reflect", "answer": "answer"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"planner": "planner", "reflect": "reflect"},
    )
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"answer": "answer"},
    )
    graph.add_edge("answer", END)
    return graph.compile()


def run_langgraph(
    question: str,
    llm: BaseLLM | None = None,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """用 LangGraph 状态图回答一个问题，返回与方案 A 相同结构的 :class:`AgentResult`。"""
    question = validate_question(question)
    llm = llm or build_llm("mock")
    graph = build_graph(llm, max_steps=max_steps)
    state = new_state(question, "langgraph")
    state["mock"] = llm.is_mock
    add_trace(
        state,
        "start",
        "开始 LangGraph 图执行",
        framework="langgraph",
        llm=llm.describe(),
        max_steps=max_steps,
        nodes=list(AVAILABLE_NODES),
    )
    # 每次循环最多 planner→tools 两次跳转，recursion_limit 留足余量避免 LangGraph 报错。
    recursion_limit = max_steps * 3 + 10
    try:
        final_state = graph.invoke(state, config={"recursion_limit": recursion_limit})
    except LangGraphUnavailable:
        raise
    except Exception as exc:  # 图执行异常也要给出可读结果，而不是直接崩
        _LOGGER.exception("LangGraph 图执行失败")
        state["error"] = f"graph_error: {exc}"
        add_trace(state, "error", "LangGraph 图执行失败", code="graph_error", message=str(exc))
        return finish(state, None, forced=True)
    if not isinstance(final_state, dict):
        final_state = state
    return AgentResult.from_state(final_state)


def graph_mermaid(max_steps: int = DEFAULT_MAX_STEPS) -> str:
    """输出 LangGraph 编译后图的 mermaid 定义（README / CLI ``graph`` 子命令使用）。"""
    graph = build_graph(MockLLM(), max_steps=max_steps)
    return graph.get_graph().draw_mermaid()
