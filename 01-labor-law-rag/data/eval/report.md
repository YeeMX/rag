# 检索与答案评测报告

> 生成时间：2026-11-21T02:58:22+08:00　|　数据：`data/benchmark.jsonl`（50 题）　|　embedding：`sentence-transformers:BAAI/bge-small-zh-v1.5`　|　rerank：`cross-encoder:BAAI/bge-reranker-base`　|　LLM：`mock`

## 1. 检索消融（按法条 article_id 计算）

| 配置 | MRR | MRR@5 | Hit@5 | Recall@5 | nDCG@5 | 平均耗时(ms) |
|---|---|---|---|---|---|---|
| dense_only | 0.4987 | 0.4987 | 0.7600 | 0.4757 | 0.4095 | 9.8 |
| bm25_only | 0.6123 | 0.6123 | 0.7600 | 0.5050 | 0.4811 | 3.3 |
| hybrid_rrf | 0.5740 | 0.5740 | 0.7600 | 0.5100 | 0.4693 | 13.6 |
| hybrid_rerank | 0.6427 | 0.6427 | 0.8400 | 0.5267 | 0.5063 | 3884.6 |
| full | 0.6267 | 0.6267 | 0.7400 | 0.4383 | 0.4569 | 4185.7 |

指标定义：MRR=第一个相关法条排名倒数均值；Hit@5=前5条至少命中一条的题目占比；Recall@5=前5条覆盖 gold 法条的比例；nDCG@5=考虑排序位置的归一化折损累计增益。

**对比（full vs 单路基线）**：

- 相比 `dense_only`：MRR 0.4987 → 0.6267（+25.7%）；Hit@5 0.7600 → 0.7400（-2.6%）；Recall@5 0.4757 → 0.4383（-7.9%）
- 相比 `bm25_only`：MRR 0.6123 → 0.6267（+2.4%）；Hit@5 0.7600 → 0.7400（-2.6%）；Recall@5 0.5050 → 0.4383（-13.2%）
- 本基准上 MRR 最高的配置是 `hybrid_rerank`（MRR=0.6427），`full` 的 MRR=0.6267。**不隐藏 full 不是最优这一事实**，详见 README §5.1 分析。

## 2. 答案层指标（full 配置）

| 指标 | 数值 | 说明 |
|---|---|---|
| 引用有效率 | 100.0% | 引用的条号真实存在于语料的比例 |
| 引用命中率 | 34.0% | 引用中包含 gold 法条的比例 |
| gold 覆盖率 | 43.8% | 答案引用覆盖了 gold 法条的比例 |
| 无引用题数 | 0 | 答案里没有任何可回溯引用 |
| 平均生成耗时 | 4011 ms | 端到端（含检索） |
| LLM 模式 | `mock` | mock 为离线抽取式，非真实大模型 |

## 3. 错误分析

逐题明细见 `data/eval/retrieval_results.json` 的 `per_case` 字段（含 first_relevant_rank / ranked_top10 / relevant）。

建议按以下顺序排查：① 多意图题是否漏召回（对比 Recall@5）；② 第一命中排名是否靠后（对比 MRR@5）；③ 重排是否把 gold 挤出前 5；④ 零样本过滤是否误删 gold。

## 4. 可复现命令

```bash
python scripts/evaluate.py --ablations --answers
```
