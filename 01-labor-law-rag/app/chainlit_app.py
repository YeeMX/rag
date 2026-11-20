"""Chainlit 前端：法律咨询对话界面。

启动：
    chainlit run app/chainlit_app.py -w
（需要在项目根目录执行，并确保 .venv-rag 的 Scripts 在 PATH 中）

设计说明：
- 前端只负责交互与展示，检索/生成全部复用 `lrs.service.RAGService`，保证"UI 与评测同源"；
- 展示检索轨迹（每阶段分数、来源通道、是否被过滤），让系统可解释——
  这是法律场景的硬需求：用户必须能看到"结论来自哪一条法条"；
- 会话级缓存 service（Chainlit 的 user_session），避免每次提问都重载模型。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chainlit as cl  # noqa: E402

logger = logging.getLogger(__name__)

EXAMPLES = [
    "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？",
    "入职8个月没签劳动合同，现在被辞退，能同时要双倍工资和赔偿金吗？",
    "我怀孕5个月，公司说业绩不好要裁员，可以解除我的合同吗？",
    "平时加班、周末加班、国庆加班，加班费分别按几倍算？",
    "公司一直不缴社保，我能要求补缴并以此辞职要补偿吗？",
]


def _service():
    svc = cl.user_session.get("rag_service")
    if svc is None:
        from lrs.service import RAGService

        svc = RAGService.load_or_build()
        cl.user_session.set("rag_service", svc)
    return svc


@cl.on_chat_start
async def on_chat_start() -> None:
    svc = _service()
    await cl.Message(
        content=(
            "## ⚖️ 劳动法智能咨询（RAG + 多路召回）\n\n"
            f"知识库：4 部劳动法系法律、357 条法条　|　"
            f"检索：BGE 稠密 + BM25 稀疏 + RRF 融合 + cross-encoder 重排　|　"
            f"生成：`{svc.llm.mode}` 模式\n\n"
            "直接输入你的问题，我会给出**带法条引用**的回答，并展示检索过程。\n\n"
            "**示例问题：**\n" + "\n".join(f"- {q}" for q in EXAMPLES) + "\n\n"
            "> 本系统仅用于技术演示，不构成正式法律意见。"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    svc = _service()
    question = message.content.strip()
    if not question:
        return

    async with cl.Step(name="检索 + 生成", type="run") as step:
        step.output = "查询理解 → 混合召回 → RRF 融合 → 重排 → 零样本过滤 → 答案生成"
        result = svc.answer(question)

    # 1) 回答主体
    answer = result.answer
    if result.citations:
        answer += "\n\n---\n**引用法条**\n" + "\n".join(
            f"- {c.title}：{c.quote}" for c in result.citations)
    if result.llm_mode == "mock":
        answer += "\n\n> ⚠️ 当前为离线 Mock 模式（未配置 `OPENAI_API_KEY`），回答为抽取式拼接，非大模型生成。"

    await cl.Message(content=answer).send()

    # 2) 检索可解释性面板
    plan_lines = []
    if result.plan.is_multi_intent:
        plan_lines.append(f"**多意图分解**（{len(result.plan.sub_queries)} 个子查询）：")
        plan_lines.extend(f"- {q}" for q in result.plan.sub_queries)
    if result.plan.expansions:
        plan_lines.append("**术语扩展**：" + "、".join(result.plan.expansions))
    lines = ["### 🔍 检索轨迹", ""]
    if plan_lines:
        lines.extend(plan_lines)
        lines.append("")
    lines.append("| # | 法条 | dense | BM25 | RRF | rerank | 来源 | 保留 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in result.retrieved:
        lines.append(
            f"| {r.rank} | {r.title} | {r.scores.get('dense', 0):.3f} | "
            f"{r.scores.get('bm25', 0):.2f} | {r.scores.get('rrf', 0):.3f} | "
            f"{r.scores.get('rerank', 0):.3f} | {r.source} | {'✅' if r.kept else '❌'} |"
        )
    lines.append("")
    lines.append(
        f"⏱ 耗时：plan {result.latency_ms['plan']}ms · retrieve "
        f"{result.latency_ms['retrieve']}ms · rerank {result.latency_ms['rerank']}ms · "
        f"filter {result.latency_ms['filter']}ms · generate {result.latency_ms['generate']}ms"
    )
    await cl.Message(content="\n".join(lines)).send()
