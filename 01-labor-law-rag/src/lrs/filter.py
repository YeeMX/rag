"""零样本相关性过滤（LLM-as-filter）。

要解决的问题：混合检索 + 重排之后，Top-5 里仍会混入"语义相近但法律上不适用"的法条。
例如问"公司未提前30天通知就辞退我，能要多少赔偿"，检索器很可能召回：
- 第40条（无过失性辞退，含"提前三十日以书面形式通知"）—— 语义高度相似；
- 第87条（违法解除的赔偿金）—— 真正该引用的条款。
两者词面/语义都很近，双塔甚至 cross-encoder 都可能排错，但"是否适用"需要**法律推理**，
这正是 LLM 零样本判断擅长的部分。

实现要点：
1. 只对候选集做"保留/剔除"的二元决策，不重新排序（避免引入不可控的排序噪声）；
2. 强制 JSON 输出 + 鲁棒解析（真实模型经常带 markdown 代码块、多说话）；
3. 任何异常都退回"保留全部候选"，保证链路绝不因过滤失败而中断；
4. Mock LLM 走同一套 Prompt 格式，保证两种模式下代码路径一致。
"""

from __future__ import annotations

import json
import logging
import re

from lrs.llm import LLM
from lrs.prompts import FILTER_PROMPT, format_context
from lrs.schemas import Retrieved

logger = logging.getLogger(__name__)


def _parse_json_loose(text: str) -> dict | None:
    """从 LLM 输出里尽量抠出 JSON（兼容 ```json 包裹、前后有解释文字）。"""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # 常见问题：尾逗号
        cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def filter_candidates(question: str, candidates: list[Retrieved], llm: LLM,
                      *, min_keep: int = 2, top_guaranteed: int = 3,
                      enabled: bool = True) -> list[Retrieved]:
    """用 LLM 做零样本适用性过滤，返回保留的候选（保持原相对顺序）。

    - min_keep：至少保留几条，防止模型过度剔除导致没有材料可答；
    - top_guaranteed：精排 top-N 无条件保留（过滤只剔除边缘近似条款，不推翻精排头部）；
    - 失败/超时/解析失败：全部保留（fail-open），并记录日志。
    """
    if not enabled or len(candidates) <= min_keep:
        return candidates
    prompt = FILTER_PROMPT.format(question=question,
                                  candidates=format_context(candidates, with_scores=True))
    try:
        raw = llm.complete(prompt, task="filter")
    except Exception as e:  # pragma: no cover - 网络异常
        logger.warning("过滤调用失败，保留全部候选: %s", e)
        return candidates

    parsed = _parse_json_loose(raw)
    if not parsed or "keep" not in parsed:
        logger.warning("过滤输出无法解析，保留全部候选。原始输出前 200 字: %r", (raw or "")[:200])
        return candidates

    keep_ids: set[int] = set()
    reasons: dict[int, str] = {}
    for item in parsed.get("keep", []) or []:
        try:
            idx = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        keep_ids.add(idx)
        reasons[idx] = str(item.get("reason", ""))[:120]
    drop_reasons: dict[int, str] = {}
    for item in parsed.get("drop", []) or []:
        try:
            idx = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        drop_reasons[idx] = str(item.get("reason", ""))[:120]

    if len(keep_ids) < min_keep:
        # 模型过度剔除时，按原顺序补足到 min_keep（而不是整体放弃过滤）
        need = min_keep - len(keep_ids)
        for i in range(1, len(candidates) + 1):
            if need <= 0:
                break
            if i not in keep_ids:
                keep_ids.add(i)
                need -= 1
        logger.warning("过滤后仅剩 %d 条（<min_keep），已按原顺序补足到 %d 条",
                       len(keep_ids) - need, len(keep_ids))

    # 安全网：无论 LLM 怎么判，精排分数最高的 top_guaranteed 条必须保留。
    # 理由：过滤只应剔除"边缘的近似条款"，不应推翻精排的头部判断；这也符合
    # "过滤是精排的补充而非替代"的设计定位，避免模型一次误判就丢掉 gold。
    ranked_by_rerank = sorted(
        range(1, len(candidates) + 1),
        key=lambda i: float(candidates[i - 1].scores.get("rerank", 0.0) or 0.0),
        reverse=True,
    )
    for i in ranked_by_rerank[: min(top_guaranteed, len(candidates))]:
        keep_ids.add(i)

    kept: list[Retrieved] = []
    for i, cand in enumerate(candidates, start=1):
        if i in keep_ids:
            cand.kept = True
            cand.scores["filter"] = 1.0
            kept.append(cand)
        else:
            cand.kept = False
            cand.scores["filter"] = 0.0
            cand.scores["drop_reason"] = drop_reasons.get(i, "")  # type: ignore[assignment]
    logger.info("零样本过滤：%d -> %d 条", len(candidates), len(kept))
    return kept
