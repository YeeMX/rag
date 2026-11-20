"""查询理解：多意图分解（decomposition）+ 法律术语扩展（expansion）。

毕设里的第一个创新点就是「上游查询分解与扩展，减少检索前的语义漂移」。
这里实现两个可独立开关、可单独评测的步骤：

1. **分解**：一个真实法律咨询往往同时问多件事（"没签合同 + 被辞退 + 能要多少赔偿"）。
   如果整句丢进检索器，向量会被多主题平均掉，导致"每件事都检不准"。做法是先按
   连接词/标点/分句切出子意图，每个子意图单独检索，最后再融合。
2. **扩展**：口语用词与法条术语存在系统性错位（"被开除" vs "解除劳动合同"），
   用一张人工维护的「口语→法条术语」映射做查询扩展，提升稀疏/稠密两路召回。
   可选叠加 LLM 改写（`use_llm=True` 时），但默认走确定性规则，保证离线可复现。

设计取舍：规则版可解释、零成本、可测；LLM 版更灵活但有延迟和不确定性。
生产上通常两级：规则兜底 + LLM 增强，这里把两者做成同一个接口。
"""

from __future__ import annotations

import logging
import re

from lrs.schemas import QueryPlan

logger = logging.getLogger(__name__)

# 编号枚举：只匹配"行首或标点后 + 单个数字 + 枚举符号"，避免把「月薪12000、」里的数字切开
_ENUM_RE = re.compile(r"(?:^|(?<=[，,；;：:、\s]))[0-9一二三四五六七八九]{1,2}[)）.、]\s*")

# 多意图切分：显式连接词、编号、分号/问号、顿号（中文列举）
_SPLIT_RE = re.compile(
    r"[；;]|(?<=[。？?！!])|"          # 分号、句末标点后切
    r"、|"                             # 顿号（"试用期6个月、工资打7折、不缴社保"）
    r"(?:另外|此外|还有|同时|并且|而且|以及|顺便问|再问|还想问)|"  # 连接词
    + _ENUM_RE.pattern                      # 编号 1. 2) (3)
)
_CONNECTOR_WORDS = ["另外", "此外", "还有", "同时", "并且", "而且", "以及", "顺便问", "再问", "还想问"]

# 单意图但包含多个"动作"的典型组合，用来判断是否需要分解
_MULTI_INTENT_HINTS = [
    ("解除", "补偿"), ("辞退", "赔偿"), ("没签", "双倍"), ("未签", "二倍"),
    ("加班", "工资"), ("工伤", "赔偿"), ("社保", "补缴"), ("试用期", "辞退"),
    ("调岗", "降薪"), ("仲裁", "时效"), ("竞业", "补偿"),
]

# 口语 -> 法条术语（人工整理，针对劳动法高频问题）
TERM_EXPANSIONS: dict[str, list[str]] = {
    "被辞退": ["用人单位解除劳动合同", "违法解除"],
    "被开除": ["解除劳动合同", "用人单位单方解除"],
    "开除": ["解除劳动合同", "用人单位单方解除"],
    "辞退": ["解除劳动合同"],
    "炒鱿鱼": ["解除劳动合同"],
    "辞职": ["劳动者解除劳动合同", "提前三十日书面通知"],
    "自离": ["劳动者解除劳动合同", "旷工"],
    "n+1": ["经济补偿", "代通知金"],
    "赔偿": ["经济补偿", "赔偿金", "二倍工资"],
    "赔偿金": ["违法解除劳动合同", "二倍经济补偿"],
    "补偿": ["经济补偿"],
    "工资": ["劳动报酬", "工资支付"],
    "拖欠工资": ["未及时足额支付劳动报酬", "克扣工资"],
    "拖欠": ["未及时足额支付劳动报酬"],
    "被迫离职": ["劳动者解除劳动合同", "未及时足额支付劳动报酬"],
    "被迫解除": ["劳动者解除劳动合同", "经济补偿"],
    "加班": ["延长工作时间", "加班费"],
    "加班费": ["延长工作时间的工资报酬", "百分之一百五十"],
    "没签合同": ["未订立书面劳动合同", "二倍工资"],
    "未签合同": ["未订立书面劳动合同", "二倍工资"],
    "不签": ["未订立书面劳动合同", "二倍工资"],
    "没签": ["未订立书面劳动合同", "二倍工资"],
    "书面劳动合同": ["未订立书面劳动合同", "二倍工资"],
    "合同到期": ["劳动合同期满", "终止劳动合同"],
    "不续签": ["劳动合同期满终止", "经济补偿"],
    "试用期": ["试用期期限", "试用期工资"],
    "转正": ["试用期", "劳动合同期限"],
    "社保": ["社会保险", "社会保险费"],
    "五险一金": ["社会保险", "住房公积金"],
    "公积金": ["住房公积金"],
    "工伤": ["工伤保险", "工伤认定"],
    "竞业": ["竞业限制"],
    "保密": ["保密义务", "竞业限制"],
    "调岗": ["变更劳动合同", "工作岗位"],
    "降薪": ["变更劳动合同", "劳动报酬"],
    "孕妇": ["孕期", "女职工", "不得解除劳动合同"],
    "怀孕": ["孕期", "女职工", "不得依第四十条、第四十一条解除"],
    "孕期": ["孕期女职工", "不得解除劳动合同"],
    "产假": ["女职工生育", "产假"],
    "裁员": ["经济性裁员", "裁减人员"],
    "年假": ["带薪年休假", "休息休假"],
    "病假": ["医疗期", "患病"],
    "仲裁": ["劳动争议仲裁", "仲裁时效"],
    "起诉": ["劳动争议诉讼", "仲裁前置"],
    "赔偿金标准": ["经济补偿标准", "月工资"],
    "n倍": ["经济补偿", "工作年限"],
    "劳动仲裁": ["劳动争议仲裁委员会", "仲裁申请"],
    "辞退赔偿": ["违法解除劳动合同的赔偿金"],
}


def _clean(text: str) -> str:
    """去掉空白与首尾标点，并把悬空的"能/可以/要"等连接词去掉。"""
    text = re.sub(r"\s+", "", text).strip("，。；、 ")
    text = re.sub(r"[能可要否是]$", "", text)
    return text


def _needs_decomposition(text: str) -> bool:
    """判断是否需要多意图分解（宁可少拆，避免把单意图题拆坏）。"""
    if any(w in text for w in _CONNECTOR_WORDS):
        return True
    if _ENUM_RE.search(text):
        return True
    # 顿号列举：出现 2 个以上顿号，且分句足够长，才认为是多件事
    clauses_by_dunhao = [c for c in re.split(r"[、]", text) if len(_clean(c)) >= 4]
    if "、" in text and len(clauses_by_dunhao) >= 3:
        return True
    # 逗号列举：至少 3 个较长分句才认定为多件事（2 个分句往往只是"事实+问题"）
    if "，" in text:
        clauses = [c for c in re.split(r"[，,]", text) if len(_clean(c)) >= 6]
        if len(clauses) >= 3:
            return True
    hits = sum(1 for a, b in _MULTI_INTENT_HINTS if a in text and b in text)
    return hits >= 2


def _split_intents(text: str, max_parts: int) -> list[str]:
    """按连接词/编号/顿号/分句切分出子意图。

    分两档：
    - 强切分（分号、连接词、编号、问号、顿号）：一定能用；
    - 逗号切分：只有在每一段都足够长（>=6 字）时才采用，避免把
      "公司违法辞退我，月薪12000、工作3年7个月，能赔多少" 切成碎片。
    切完统一做"尾巴合并"（"能同时要…吗"这类指代整句的分句并回上一条）。
    """
    normalized = _ENUM_RE.sub("；", text)
    strong = [p for p in _SPLIT_RE.split(normalized) if p and len(_clean(p)) >= 4]
    if len(strong) >= 2:
        return _merge_tail_clauses([_clean(p) for p in strong if _clean(p)])[:max_parts]

    comma_parts = [p for p in re.split(r"[，,]", normalized) if p]
    merged = _merge_commas(comma_parts)
    if len(merged) >= 2 and all(len(p) >= 6 for p in merged):
        # 逗号切分时上一条可能以逗号结尾，做一次清理
        return _merge_tail_clauses([p.strip("，。；、 ") for p in merged])[:max_parts]
    return []


def _merge_commas(parts: list[str]) -> list[str]:
    """合并过短的逗号分句（"现在被辞退" 只有 5 字，与上一句合并）。"""
    out: list[str] = []
    for p in parts:
        p = _clean(p)
        if not p:
            continue
        if out and len(p) < 6:
            out[-1] = f"{out[-1]}，{p}"
        else:
            out.append(p)
    return out


def _merge_tail_clauses(parts: list[str]) -> list[str]:
    """把"能赔多少/怎么办"这类没有独立内容的尾巴合并到上一条。

    注意：如果尾巴本身带了具体的请求内容（"要双倍工资和赔偿金吗"），
    它就是一个独立的子意图，不能合并，否则多意图分解等于没做。
    """
    if not parts:
        return parts
    tail_re = re.compile(r"^(能|可以|是否|需不需要|要不要|怎么|如何|该|这|还|并)")
    content_hint = re.compile(r"(工资|赔偿|补偿|社保|合同|工伤|加班|休假|竞业|证据|仲裁|起诉|"
                              r"补缴|恢复|继续履行|违约金|双倍|二倍)")
    merged: list[str] = []
    for p in parts:
        p = _clean(p)
        is_empty_tail = (
            merged
            and len(p) <= 10
            and tail_re.match(p)
            and not content_hint.search(p)
        )
        if is_empty_tail:
            merged[-1] = _clean(f"{merged[-1]}，{p}")
        else:
            merged.append(p)
    return merged


def _expand_terms(text: str) -> list[str]:
    """按术语表做查询扩展，返回命中的法条术语（去重，保持稳定顺序）。"""
    low = text.lower()
    out: list[str] = []
    for colloquial, legal_terms in TERM_EXPANSIONS.items():
        if colloquial in low:
            for t in legal_terms:
                if t not in out:
                    out.append(t)
    return out


# 只保留"有判别力"的法律术语用于拼复合查询：去掉过于宽泛的词
_GENERIC_TERMS = {"经济补偿", "赔偿金", "二倍工资", "劳动报酬", "工资支付"}
# 提问/口语尾巴，拼复合查询时去掉，保留"核心事实"
_TAIL_RE = re.compile(r"(吗|呢|吧|啊|怎么算|如何计算|能赔多少|能要多少|可以要吗|是否合法|怎么办|"
                      r"能不能|可不可以|有没有|是多少|得多少)[？?。！!]*$")


def _strip_tail(question: str) -> str:
    """提取"核心事实句"：去掉疑问尾巴、纯提问分句和句末标点。

    例：
        "公司拖欠我两个月工资，我可以直接走人并要求经济补偿吗？"
        -> "公司拖欠我两个月工资"
    """
    # 先按逗号/问号分成小句，丢掉纯提问小句
    clauses = [c.strip() for c in re.split(r"[，,；;？?。！!]", question) if c.strip()]
    keep: list[str] = []
    for c in clauses:
        if re.match(r"^(请问|我想问|想问)", c):
            c = re.sub(r"^(请问|我想问|想问)", "", c).strip()
        if not c:
            continue
        # 纯提问/程序性小句：不含具体事实，丢掉
        if re.search(r"(怎么办|怎么算|如何|能不能|可不可以|可以吗|是否|要不要|需要吗|"
                     r"走人|直接走|仲裁|起诉|维权)", c) and not re.search(r"辞退|开除|拖欠|"
                                                                          r"未签|没签|怀孕|加班|"
                                                                          r"受伤|社保", c):
            continue
        keep.append(c)
    core = "，".join(keep[:2]) if keep else question
    return core.rstrip("？?。！!，, ")[:40]


def _build_compound_queries(question: str, terms: list[str], max_parts: int = 3) -> list[str]:
    """把核心事实与法律术语拼成"法条式查询"，作为高权重检索变体。

    为什么不直接把术语单独当检索式：实测发现"解除劳动合同"这类通用术语单独检索时，
    会召回大量泛化条款（第28条、第46条…），再经 RRF 多路投票，反而把真正精准的
    第87条挤出 top-10。而"原问题 + 术语"的完整句又会被口语叙述稀释（例如
    "公司拖欠我两个月工资，我可以直接走人..."）。折中做法：先用**去掉疑问尾巴的
    核心事实**拼接术语，得到与法条语言最接近的查询。

    例：
        "公司拖欠我两个月工资，我可以直接走人并要求经济补偿吗？"
        -> "公司拖欠我两个月工资 未及时足额支付劳动报酬"
        -> dense 能把第38条从 top-10 之外拉进 top-3。
    """
    terms = [t for t in terms if t not in _GENERIC_TERMS]
    if not terms:
        return []
    core = _strip_tail(question)
    sig_terms = [t for t in terms if not t.isdigit()] or terms
    out: list[str] = []
    # 主复合查询：核心事实 + 最重要的术语（术语拼接短语，而不是拆成单独检索式）
    for t in sig_terms[:2]:
        q = f"{core} {t}"
        if q not in out:
            out.append(q)
    # 术语融合查询：只保留法律术语（去掉个人叙事），法条语言最纯，稠密检索最"锐利"
    if len(sig_terms) >= 2:
        q = " ".join(sig_terms[:3])
        if q not in out:
            out.append(q)
    # 术语组合查询：两个术语拼起来（法条里共现概率更高）
    for i, t in enumerate(sig_terms[:3]):
        for u in sig_terms[i + 1:i + 2]:
            q = f"{core} {t} {u}"
            if q not in out:
                out.append(q)
        if len(out) >= max_parts:
            break
    return out[:max_parts]


def plan_query(question: str, *, decompose: bool = True, expand: bool = True,
               max_sub_queries: int = 3) -> QueryPlan:
    """把原始问题转成 QueryPlan（确定性规则版，离线可复现）。"""
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    multi = decompose and _needs_decomposition(question)
    sub_queries: list[str] = []
    if multi:
        sub_queries = _split_intents(question, max_parts=max_sub_queries)

    expansions: list[str] = []
    compound: list[str] = []
    if expand:
        bases = [question, *sub_queries]
        for b in bases:
            for t in _expand_terms(b):
                if t not in expansions:
                    expansions.append(t)
        compound = _build_compound_queries(question, expansions, max_parts=max_sub_queries)

    note = ""
    if multi:
        note = f"检测到多意图，拆分为 {len(sub_queries)} 个子查询"
    elif expansions:
        note = f"单意图，触发 {len(expansions)} 个术语扩展"
    return QueryPlan(
        original=question,
        sub_queries=sub_queries,
        expansions=expansions[:6],
        compound_queries=compound,
        is_multi_intent=multi,
        note=note,
    )


def plan_query_llm(question: str, llm, *, max_sub_queries: int = 3) -> QueryPlan:
    """LLM 版分解（可选）。LLM 不可用/解析失败时回退到规则版，保证链路不中断。"""
    from lrs.prompts import DECOMPOSE_PROMPT

    rule_plan = plan_query(question, decompose=True, expand=True, max_sub_queries=max_sub_queries)
    if llm is None or not getattr(llm, "available", False):
        return rule_plan
    try:
        raw = llm.complete(
            DECOMPOSE_PROMPT.format(question=question, max_parts=max_sub_queries),
            task="decompose",
        )
        subs = [s.strip() for s in re.split(r"[\n；;]+", raw) if s.strip()]
        subs = [s for s in subs if len(s) >= 3][:max_sub_queries]
        if not subs or subs == [question]:
            return rule_plan
        return QueryPlan(
            original=question,
            sub_queries=subs,
            expansions=rule_plan.expansions,
            compound_queries=rule_plan.compound_queries,
            is_multi_intent=len(subs) > 1,
            note=f"LLM 分解为 {len(subs)} 个子查询",
        )
    except Exception as e:  # pragma: no cover - 网络/解析异常
        logger.warning("LLM 分解失败，回退规则版: %s", e)
        return rule_plan
