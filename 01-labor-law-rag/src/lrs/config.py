"""全局配置：模型、路径、检索参数。

设计原则：
1. 默认配置在 **CPU + 无 API Key** 下也能完整跑通（自动降级），方便任何人 clone 后直接复现；
2. 所有可调参数集中在这里，评测脚本按同名参数生成不同配置，避免"评测代码"和"线上代码"两套逻辑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
# 法条语料随仓库一起分发（四部法律全文 + 结构化 JSONL），保证 clone 即可运行
CORPUS_DIR = DATA_DIR / "corpus"
ARTICLES_JSONL = CORPUS_DIR / "labor_law_articles.jsonl"

CHROMA_DIR = DATA_DIR / "chroma"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
RUNS_DIR = DATA_DIR / "runs"

for _d in (DATA_DIR, CHROMA_DIR, INDEX_DIR, EVAL_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """一次检索/生成流程的完整配置。评测时用 dataclasses.replace 派生变体。"""

    # --- 检索 ---
    top_k: int = 30               # 融合后的召回池大小（送进重排的候选）
    final_k: int = 5              # 最终注入 Prompt 的法条数
    dense_enabled: bool = True    # 稠密召回开关（消融用）
    bm25_enabled: bool = True     # 稀疏召回开关（消融用）
    dense_model: str = "BAAI/bge-small-zh-v1.5"
    dense_batch_size: int = 32
    rrf_k: int = 60               # RRF 平滑常数，论文取 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0

    # --- 重排 ---
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True
    rerank_candidates: int = 10   # 送进 cross-encoder 的候选数

    # --- 查询处理 ---
    decompose_enabled: bool = True
    expand_enabled: bool = True
    max_sub_queries: int = 3

    # --- 过滤 / 生成 ---
    filter_enabled: bool = True
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_answer_tokens: int = 800
    use_llm: bool = False         # 由 llm.build_llm() 根据环境变量决定

    # --- 运行 ---
    seed: int = 42
    device: str = "cpu"
    allow_model_download: bool = True
    collection_name: str = "labor_law"


DEFAULT = Settings()


def load_settings(**overrides) -> Settings:
    """从环境变量 + 显式参数构造配置（显式参数优先）。"""
    env_map = {
        "LRS_TOP_K": ("top_k", int),
        "LRS_FINAL_K": ("final_k", int),
        "LRS_DENSE_MODEL": ("dense_model", str),
        "LRS_RERANK_MODEL": ("rerank_model", str),
        "LRS_RERANK": ("rerank_enabled", lambda v: v.lower() not in {"0", "false", "no"}),
        "LRS_LLM_MODEL": ("llm_model", str),
        "LRS_DEVICE": ("device", str),
    }
    env_values = {}
    for env_key, (field_name, caster) in env_map.items():
        if env_key in os.environ:
            env_values[field_name] = caster(os.environ[env_key])
    return replace(DEFAULT, **{**env_values, **overrides})


# 供 README / 评测报告的配置说明
ABLATION_PRESETS: dict[str, dict] = {
    # 只开稠密通道
    "dense_only": dict(decompose_enabled=False, expand_enabled=False, rerank_enabled=False,
                       filter_enabled=False, dense_enabled=True, bm25_enabled=False),
    # 只开 BM25 通道
    "bm25_only": dict(decompose_enabled=False, expand_enabled=False, rerank_enabled=False,
                      filter_enabled=False, dense_enabled=False, bm25_enabled=True),
    # 双路 + RRF（查询理解关闭）
    "hybrid_rrf": dict(decompose_enabled=False, expand_enabled=False, rerank_enabled=False,
                       filter_enabled=False, dense_enabled=True, bm25_enabled=True),
    # 双路 + RRF + cross-encoder 重排
    "hybrid_rerank": dict(decompose_enabled=False, expand_enabled=False, rerank_enabled=True,
                          filter_enabled=False, dense_enabled=True, bm25_enabled=True),
    # 完整链路
    "full": dict(decompose_enabled=True, expand_enabled=True, rerank_enabled=True,
                 filter_enabled=True, dense_enabled=True, bm25_enabled=True),
}
