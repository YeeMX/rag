"""LLM 抽象层：OpenAI 兼容 API + 离线确定性 Mock。

为什么要 Mock：
1. 答辩/面试现场没有 API key 或网络时，整条链路必须能演示（检索、重排、过滤、评测都能跑）；
2. 单元测试不能依赖外部 API，否则 CI 既慢又不稳；
3. Mock 必须**诚实**：它基于检索结果做抽取式回答，并在输出里明确标注 `[MOCK]`，
   不伪装成真实模型生成，避免把"规则拼装"说成"大模型能力"。

真实模式：设置以下环境变量后自动启用（兼容 OpenAI / DeepSeek / 通义 / vLLM 等任何
OpenAI 兼容端点）：
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://api.openai.com/v1   # 可选
    OPENAI_MODEL=gpt-4o-mini                    # 可选
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLM(ABC):
    """最小 LLM 接口：一次性补全（同步），足够覆盖过滤/生成/评测判分。"""

    mode: str = "base"
    model: str = ""
    available: bool = False

    @abstractmethod
    def complete(self, prompt: str, *, task: str = "generic", max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        """给定完整 prompt，返回文本。task 仅用于日志/统计。"""

    def json_complete(self, prompt: str, *, task: str = "generic") -> str:
        """要求模型输出 JSON 的便捷方法（Mock 直接返回构造好的 JSON 字符串）。"""
        return self.complete(prompt, task=task)


class MockLLM(LLM):
    """离线确定性 Mock。

    - `task="filter"`：prompt 里会带候选法条列表（由 prompts.FILTER_PROMPT 的约定格式），
      这里做**基于法条文本与问题词面覆盖率 + 关键词规则**的筛选，模拟"零样本相关性过滤"；
    - `task="answer"`：抽取式拼装答案（引用检索到的法条原文）；
    - `task="judge"`：按启发式给出合规/引用评分。
    所有输出都带 [MOCK] 前缀，绝不冒充真实模型。
    """

    mode = "mock"
    model = "mock-extractive"
    available = True

    def complete(self, prompt: str, *, task: str = "generic", max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        if task == "answer":
            return self._mock_answer(prompt)
        if task == "filter":
            return self._mock_filter(prompt)
        if task == "judge":
            return self._mock_judge(prompt)
        return "[MOCK] 未配置 LLM，无法完成该生成任务。"

    # ------------------------------------------------------------------
    def _mock_answer(self, prompt: str) -> str:
        # prompt 里法条材料由 prompts.format_context 拼成 "[i] 《法律》第X条：正文"
        import re

        facts: list[tuple[str, str]] = []
        for m in re.finditer(r"^\s*\[(\d+)\]\s*(.+?)：(.+)$", prompt, re.M):
            title = m.group(2).strip()
            body = m.group(3).strip()
            first = body.split("。")[0].strip()
            if first:
                facts.append((title, first))
        if not facts:
            return "[MOCK] 没有检索到可引用的法条，无法给出结论。请补充事实细节。"
        parts = ["[MOCK] 以下是基于检索到的法条原文的抽取式回答（未调用真实大模型）："]
        for i, (title, first) in enumerate(facts[:5], 1):
            parts.append(f"{i}. {title}规定：{first}。")
        parts.append(
            "以上为法条原文要点，具体适用需结合完整事实与当地裁判口径；"
            "本回答不构成正式法律意见。"
        )
        return "\n".join(parts)

    def _mock_filter(self, prompt: str) -> str:
        """根据 prompt 中候选编号与文本，判断与问题的关系，输出 JSON。"""
        import json
        import re

        question = ""
        m = re.search(r"用户问题[:：]\s*(.+)", prompt)
        if m:
            question = m.group(1).strip()
        # 候选行格式："1. 《劳动合同法》第八十七条 [精排=0.87]：正文"
        candidates = re.findall(r"^\s*(\d+)[.、]\s*(《[^》]+》[^：:\n\[]+?)\s*(?:\[精排=([\d.\-]+)\])?[：:]\s*(.+)$",
                                prompt, re.M)
        rows = []
        for idx, title, score, body in candidates:
            rs = float(score) if score else 0.0
            rows.append({"id": int(idx), "title": title.strip(), "score": rs, "body": body})
        if not rows:
            return json.dumps({"keep": [], "drop": []}, ensure_ascii=False)

        # 相关性启发式：精排相对分（主）+ 问题词覆盖（辅）。
        # 注意：这是 Mock 的近似判断，不是真实 LLM 的法律推理；报告里会标注 mock。
        # 阈值设得偏松（保留约 50-60% 候选），避免 Mock 误删 gold 把指标测坏。
        for row in rows:
            row["cov"] = self._coverage(question, row["body"])
        max_score = max(r["score"] for r in rows) or 1.0
        for row in rows:
            rel_score = row["score"] / max_score
            row["relevant"] = bool(rel_score >= 0.60 or row["cov"] >= 0.34)
        decisions = [{"id": r["id"], "relevant": r["relevant"],
                      "reason": (f"精排相对分 {r['score'] / max_score:.2f}，问题词覆盖 {r['cov']:.2f}"
                                 if r["relevant"] else
                                 f"精排相对分仅 {r['score'] / max_score:.2f} 且覆盖 {r['cov']:.2f}，近似条款")}
                     for r in rows]
        return json.dumps({"keep": [d for d in decisions if d["relevant"]],
                           "drop": [d for d in decisions if not d["relevant"]]},
                          ensure_ascii=False)

    @staticmethod
    def _coverage(question: str, body: str) -> float:
        """问题关键词（含术语扩展）/ 法条正文的匹配率，返回 0~1。

        注意：只统计**非数字**的法律术语，否则问题里的"12000""3年7个月"这类事实数字
        会稀释覆盖率，导致真正相关的法条被误判为不相关。
        """
        import re as _re

        from lrs.query_processing import TERM_EXPANSIONS

        keywords = [k for k in TERM_EXPANSIONS if k in question]
        expanded = [t for k in keywords for t in TERM_EXPANSIONS[k]]
        strong = ["经济补偿", "赔偿金", "二倍工资", "加班", "试用期", "解除", "终止",
                  "工伤", "社会保险", "仲裁", "竞业限制", "年休假", "医疗期", "违约金"]
        terms = set(keywords) | set(expanded) | {w for w in strong if w in question}
        terms = {t for t in terms if not _re.search(r"\d", t)}
        if not terms:
            return 0.0
        hits = sum(1 for t in terms if t in body)
        return hits / len(terms)

    def _mock_judge(self, prompt: str) -> str:
        import json
        import re

        # 答案里是否给出了明确的法条引用与结论性表述
        has_citation = bool(re.search(r"第[一二三四五六七八九十百零]+条", prompt))
        has_structure = any(w in prompt for w in ["规定", "因此", "综上", "建议"])
        compliance = 4.0 if (has_citation and has_structure) else 3.0
        citation = 4.2 if has_citation else 2.5
        return json.dumps({"legal_compliance": compliance, "citation_accuracy": citation,
                           "note": "[MOCK] 启发式打分，非 GPT-4o"}, ensure_ascii=False)


class OpenAICompatLLM(LLM):
    """任何 OpenAI 兼容端点的实现。"""

    mode = "openai"
    available = True

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        from openai import OpenAI

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("缺少 OPENAI_API_KEY")
        base = base_url or os.environ.get("OPENAI_BASE_URL") or None
        self.client = OpenAI(api_key=key, base_url=base)
        self.base_url = base or "https://api.openai.com/v1"

    def complete(self, prompt: str, *, task: str = "generic", max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 if temperature is None else temperature,
            max_tokens=max_tokens or 800,
        )
        content = resp.choices[0].message.content or ""
        logger.debug("LLM[%s] 返回 %d 字符", task, len(content))
        return content.strip()


def build_llm(prefer_real: bool = False) -> LLM:
    """有 key 且 prefer_real=True 时用真实模型，否则退回 Mock（并明确日志提示）。"""
    key = os.environ.get("OPENAI_API_KEY")
    if key and prefer_real:
        try:
            llm = OpenAICompatLLM()
            logger.info("使用真实 LLM：%s @ %s", llm.model, llm.base_url)
            return llm
        except Exception as e:  # pragma: no cover - 依赖网络
            logger.warning("初始化真实 LLM 失败（%s），降级为 Mock", e)
    elif key and not prefer_real:
        logger.info("检测到 OPENAI_API_KEY，但未显式开启 --real-llm，使用 Mock（离线可复现）")
    else:
        logger.info("未检测到 OPENAI_API_KEY，使用离线 Mock LLM")
    return MockLLM()
