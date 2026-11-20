# ⚖️ Intelligent Labor Law Consultation System（RAG + 多路召回）

> 基于 **LangChain + ChromaDB + Chainlit** 的端到端劳动法 RAG 咨询平台：
> 上游做**查询分解与术语扩展**，中游做 **BGE 稠密 + BM25 稀疏 → RRF 融合 → cross-encoder 重排 → 零样本 LLM 过滤**，
> 下游做**带引用约束的答案生成**，并提供**可复现的检索/答案双层评测**。

| 维度 | 说明 |
|---|---|
| 语料 | 4 部劳动法系法律、**357 条**法条原文（劳动合同法 98 / 劳动法 107 / 劳动争议调解仲裁法 54 / 社会保险法 98） |
| 检索 | Sentence-BERT（`bge-small-zh-v1.5`）+ BM25，RRF 融合，`bge-reranker-base` 精排 |
| 生成 | OpenAI 兼容 API（GPT-4o-mini 等）；**无 Key 时自动降级为离线抽取式 Mock，全链路仍可跑通** |
| 前端 | Chainlit 对话界面 + 检索轨迹可视化 |
| 评测 | 50 题人工场景基准，5 种消融配置，检索 4 指标 + 答案引用校验 + LLM-as-judge |
| LLM 成本 | 离线模式零成本；真实模式单次问答约 2 次调用（过滤 + 生成） |

---

## 1. 为什么这个项目不只是"把文档塞进向量库"

法律咨询有三个和通用 RAG 不一样的地方，本项目的设计都围绕它们展开：

| 痛点 | 具体表现 | 本项目的解法 |
|---|---|---|
| **问题本身是复合的** | "没签合同 + 被辞退 + 能要多少赔偿"是一句话问了 3 件事，整句向量会被平均，检索结果"每件都不准" | 查询**分解**成子意图，各自检索后再融合（`query_processing.py`） |
| **口语与法条术语错位** | 用户说"被开除""N+1""双倍工资"，法条写的是"解除劳动合同""经济补偿""二倍工资" | 人工维护的**术语扩展表**，把口语映射到法条用语（离线、可复现） |
| **"像"不等于"适用"** | 检索会召回"语义相近但法律上不适用"的条款（问违法解除却召回第 40 条无过失性辞退） | **零样本 LLM 过滤**：对候选做法条适用性判断，剔除近似条款（`filter.py`） |

## 2. 系统架构

```mermaid
flowchart TD
    U[用户问题] --> QP[查询理解<br/>多意图分解 + 术语扩展]
    QP -->|原始问题 / 子查询 / 扩展词| R

    subgraph R [多路召回]
        D[稠密检索<br/>bge-small-zh + ChromaDB<br/>cosine]
        B[稀疏检索<br/>BM25<br/>单字+二元组分词]
    end

    R --> F[RRF 融合<br/>score = Σ 1/(k+rank)]
    F --> RR[Cross-Encoder 精排<br/>bge-reranker-base]
    RR --> FL{零样本适用性过滤<br/>LLM 判断 keep/drop}
    FL -->|保留的直接适用条款| G[答案生成<br/>Prompt 约束：只能引用给定法条]
    FL -->|剔除近似条款| X[丢弃]
    G --> C[带引用回答 + 检索轨迹]
    C --> UI[Chainlit 前端]
```

一次请求的数据流（代码对应 `src/lrs/service.py::RAGService.answer`）：

1. **查询理解**：`plan_query()` 产出 `QueryPlan{original, sub_queries, expansions}`；
2. **多路召回**：每个检索式（原问题 1.0 / 复合查询 0.9 / 子查询 0.8 / 扩展词 0.4 加权）分别跑稠密 + BM25，各取 top-30；
3. **RRF 融合**：把「每个检索式的每一路排名」一起加权投票，`Σ w/(60+rank)` 累加，取 top-30；
4. **重排**：`bge-reranker-base` 对 top-30 逐条打分，取 top-10（延迟主要来自这一步）；
5. **零样本过滤**：LLM 返回 JSON 的 keep/drop；精排 top-3 无条件保留、至少保留 2 条（fail-open，解析失败不中断）；
6. **生成**：强制"只依据给定法条 + 每条结论标注条号"，输出后做**引用回溯校验**，最终注入 top-5。

## 3. 快速开始

### 3.1 安装

```bash
git clone https://github.com/YeeMX/rag.git && cd rag
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

> 首次运行会下载两个模型（约 100MB + 1.1GB，CPU 推理即可）。
> 没有网络时不需要改任何代码：embedding/rerank 会自动降级为哈希/词面实现，
> LLM 自动降级为离线 Mock，评测报告会**如实标注**用的是哪种 backend。

### 3.2 构建索引

```bash
python scripts/cli.py build
# 索引写入 data/chroma/（向量）与 data/index/（BM25 缓存）
```

### 3.3 命令行问答

```bash
# 离线 Mock 模式（无需 API Key，开箱即用）
python scripts/cli.py ask "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？"

# 真实 LLM 模式（任何 OpenAI 兼容端点）
export OPENAI_API_KEY=sk-xxx        # Windows: $env:OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL=https://api.openai.com/v1   # 可选：DeepSeek/通义/vLLM 等
python scripts/cli.py ask "..." --real-llm
```

### 3.4 Chainlit 前端

```bash
chainlit run app/chainlit_app.py -w
# 打开 http://localhost:8000
```

### 3.5 评测

```bash
# 检索消融（5 种配置 × 50 题）+ 答案层引用校验
python scripts/evaluate.py --ablations --answers

# 加上 LLM-as-judge（mock 走启发式；--real-llm 走真实模型）
python scripts/evaluate.py --ablations --answers --judge --real-llm
```

产出：`data/eval/retrieval_results.json`、`data/eval/answer_results.json`、`data/eval/report.md`。

## 4. 评测设计

### 4.1 基准集（`data/benchmark.jsonl`）

- **50 道**人工编写的真实劳动争议场景题，覆盖 32 个主题（试用期、未签合同、违法解除、经济补偿、加班费、社保、工伤、仲裁时效…）；
- 每题标注 **gold 法条**（平均 2.68 条/题，47 道多 gold 题）+ **hard negative**（语义相近但不适用的干扰条款，共 100 条）；
- 题目故意用口语、复合句式，模拟真实咨询，而不是"法条考试题"。

### 4.2 消融配置

| 配置 | 分解/扩展 | RRF 混合 | 重排 | 零样本过滤 | 验证什么 |
|---|:---:|:---:|:---:|:---:|---|
| `dense_only` | ✗ | ✗ | ✗ | ✗ | 纯向量检索基线 |
| `bm25_only` | ✗ | ✗ | ✗ | ✗ | 纯关键词检索基线 |
| `hybrid_rrf` | ✗ | ✓ | ✗ | ✗ | 混合召回是否优于单路 |
| `hybrid_rerank` | ✗ | ✓ | ✓ | ✗ | cross-encoder 精排的增益 |
| `full` | ✓ | ✓ | ✓ | ✓ | 完整链路（查询理解 + 过滤） |

### 4.3 指标定义

检索指标按**法条**（article_id）计算，同一法条多个 chunk 只算一次：

- **MRR**：第一个相关法条排名的倒数均值（"第一名找得准不准"）；
- **Hit@5**：前 5 条至少命中一条相关法条的题目占比（"有没有找到"）；
- **Recall@5**：前 5 条覆盖 gold 法条的比例（"找全没找全"，对多意图题更敏感）；
- **nDCG@5**：考虑排序位置的折损指标。

答案层指标：

- **引用有效率**：答案引用的条号真实存在于语料的比例（防模型编造条号）；
- **引用命中率 / gold 覆盖率**：引用是否落在 gold 法条上；
- **LLM-as-judge**：法律合规性、引用准确性各 1-5 分。

## 5. 实测结果

> 以下数字全部由 `python scripts/evaluate.py --ablations --answers` 生成，
> 原始输出见 `data/eval/`。运行环境：**Windows / CPU（无 GPU）/ Python 3.11**。

<!-- EVAL_RESULTS_START -->
**检索消融**（50 题，按法条 article_id 计算；CPU 推理）：

| 配置 | 分解+扩展 | RRF 混合 | 重排 | 零样本过滤 | MRR | MRR@5 | Hit@5 | Recall@5 | nDCG@5 | 平均耗时 |
|---|:---:|:---:|:---:|:---:|---|---|---|---|---|---|
| `dense_only` | ✗ | ✗ | ✗ | ✗ | 0.4987 | 0.4987 | 0.7600 | 0.4757 | 0.4095 | 9.8 ms |
| `bm25_only` | ✗ | ✗ | ✗ | ✗ | **0.6123** | 0.6123 | 0.7600 | 0.5050 | 0.4811 | 3.3 ms |
| `hybrid_rrf` | ✗ | ✓ | ✗ | ✗ | 0.5740 | 0.5740 | 0.7600 | **0.5100** | 0.4693 | 13.6 ms |
| `hybrid_rerank` | ✗ | ✓ | ✓ | ✗ | **0.6427** | 0.6427 | **0.8400** | **0.5267** | **0.5063** | 3884.6 ms |
| `full`（+查询理解） | ✓ | ✓ | ✓ | ✓ | 0.6267 | 0.6267 | 0.7400 | 0.4383 | 0.4569 | 4185.7 ms |

**答案层指标**（`full` 配置，离线 Mock LLM；完整报告见 [`data/eval/report.md`](data/eval/report.md)）：

| 指标 | 数值 | 说明 |
|---|---|---|
| 引用有效率 | **100.0%** | 答案引用的条号全部真实存在于语料（无编造条号） |
| 引用命中率 | 34.0% | 引用中包含 gold 法条的比例 |
| gold 覆盖率 | 43.8% | 答案引用覆盖了 gold 法条的比例 |
| 无引用题数 | 0 / 50 | 没有出现"无法回溯的引用" |
| 端到端平均耗时 | 4011 ms | 含检索 + 重排 + 过滤 + 生成（CPU） |

> "引用有效率 100%" 是因为生成后做了**引用回溯校验**：条号必须能在注入的法条材料里找到，
> 找不到就不会进入 citations。这也解释了为什么"引用命中率"不高——Mock 是抽取式拼装，
> 不具备真实 LLM 的"选对条款"能力；真实模型模式下这两项指标才有可比性。

### 5.1 怎么读这张表（真实结论，不美化）

1. **重排是收益最大的一环**：`hybrid_rrf → hybrid_rerank`，MRR +0.069、Hit@5 +8pt，代价是延迟从 14ms 涨到 3.9s（CPU 上 30 候选 × cross-encoder）。这正是"用算力换精度"的两段式检索，线上应换 GPU/小模型蒸馏。
2. **稠密单路不如 BM25 单路**（0.4987 vs 0.6123）：本语料是 357 条**短法条**，术语密度高、表述固定，关键词匹配天然占优；而 `bge-small-zh` 在"事实性长问句 vs 短法条"上的语义对齐并不完美。**这不是说稠密没用**——`hybrid_rrf` 的 Recall@5 比任意单路都高（0.5100），说明两路互补；只是 RRF 在 MRR 上被 BM25 的头部优势拖平了。
3. **查询理解在 full 反而拉低了指标**（MRR 0.6427 → 0.6267）：在本基准上，分解/扩展引入的检索式会稀释主查询信号，且 **Mock LLM 的过滤启发式会误删 gold**（过滤把 10 条裁到 3 条，Recall 从 0.53 掉到 0.44）。这是**真实的工程发现**：过滤与查询扩展的价值依赖"强 LLM + 足够大的候选池"，在没有真实模型时应默认关闭或调低其影响。代码里两者都可独立开关，评测配置也是按开关跑的。
4. **Hit@5 全部在 0.74~0.84**：说明"找得到"不是瓶颈，"排得准"和"过滤不误删"才是。

> **因此默认配置的选择**：`full` 适合演示架构全貌；**追求指标时应使用 `hybrid_rerank`**。
> 线上如果启用零样本过滤，必须接真实 LLM（`--real-llm`），并在自己的数据集上重新验证过滤的净收益。

<!-- EVAL_RESULTS_END -->

### 5.1 数据说明与"与毕设论文指标的差异"

本项目是毕设系统的**开源复刻版**，为了可复现做了三处主动裁剪，**不追求复现论文里的 MRR 0.76 / Hit 0.91**：

1. **语料规模**：这里用 4 部核心法律的 357 条法条（约 3.3 万字），毕设时使用的是更大规模的法规库；语料越小，检索任务的难度分布不同，指标不可直接横向比较；
2. **评测集规模**：50 题人工基准，与毕设时构造的 50 个复杂场景在题型分布上一致，但题目本身不同；
3. **生成模型**：默认离线 Mock（抽取式），只有配置 `OPENAI_API_KEY` 后的 `--real-llm` 模式才有真正的 LLM 生成与 judge。

**面试时正确的说法**是："这个开源版本用一套可复现的 50 题基准重新评测了同样的架构，消融实验显示每一层各自的贡献是 X / Y / Z；论文里的数字是在更大语料和 GPT-4o 上测的，两者环境不同，不作直接比较。"

## 6. 项目结构

```
01-labor-law-rag/
├── app/chainlit_app.py        # Chainlit 前端（复用 service，与评测同源）
├── data/
│   ├── corpus/                # 法条原文 + 结构化 JSONL + 解析/自举脚本
│   ├── benchmark.jsonl        # 50 题评测基准（由 benchmark_seed.py 生成并校验）
│   └── benchmark_seed.py      # 人工编写的题目与 gold / hard negative 标注
├── scripts/
│   ├── cli.py                 # build / ask / chat / eval 命令行
│   └── evaluate.py            # 消融评测 + 答案评测 + Markdown 报告生成
├── src/lrs/
│   ├── config.py              # 所有可调参数与消融预设
│   ├── schemas.py             # 数据结构（LegalArticle/Chunk/Retrieved/QueryPlan/RAGAnswer）
│   ├── corpus.py              # 语料加载与切分（LangChain 切分器）
│   ├── embeddings.py          # Sentence-BERT + 离线哈希降级
│   ├── lexical.py             # BM25（rank_bm25 + 内置实现）
│   ├── vector_store.py        # ChromaDB 封装
│   ├── retriever.py           # 稠密+稀疏召回 + RRF 融合
│   ├── reranker.py            # cross-encoder 精排 + 词面降级
│   ├── query_processing.py    # 查询分解 + 术语扩展
│   ├── filter.py              # 零样本适用性过滤（鲁棒 JSON 解析）
│   ├── prompts.py             # Prompt 工程集中管理（版本化）
│   ├── llm.py                 # OpenAI 兼容 + 离线 Mock
│   ├── service.py             # 服务装配与完整问答链路
│   └── metrics.py             # 检索指标（纯函数，可单测）
└── tests/                     # pytest：指标/查询理解/检索/过滤/端到端
```

## 7. 测试

```bash
# 单元 + 集成（HashEmbedder + 临时 Chroma，离线、几秒跑完）
pytest -q

# 需要下载 BGE 模型的真实集成测试
pytest -m integration -q

# 混合检索冒烟
pytest tests/test_pipeline.py -q
```

## 8. 可复现性设计

1. **固定随机性**：配置里统一 `seed`；
2. **失败可降级、降级可感知**：embedding/rerank/LLM 三层都有离线兜底，但**报告里会写明 backend**，不会把降级实现的结果冒充 BGE/GPT-4o 的成绩；
3. **评测与线上同源**：`evaluate.py` 调用的是 `RAGService` 的同一套 `retrieve → rerank → filter`，不存在"评测代码和演示代码两套逻辑"；
4. **一键出报告**：`report.md` 自动生成表格，避免手抄数字。

## 9. 面试讲解要点

1. **为什么 RRF 而不是加权分数相加**：稠密相似度与 BM25 分数尺度不可比，RRF 只用排名、无需调参，鲁棒性好；
2. **为什么召回之后还要 cross-encoder**：双塔没有 query-passage 交互，精排用小候选集换精度，是"粗排→精排"的标准两段式；
3. **零样本过滤解决什么**：解决"语义相似≠法律适用"，这是法律领域 RAG 与通用 RAG 最本质的区别；
4. **查询分解的价值**：多意图问题整句检索会被平均，分解后每段独立召回再用 RRF 合并，Recall 提升最明显；
5. **评测怎么防作弊**：引用回溯校验（条号必须在语料里）+ gold 覆盖率 + 同一套代码；
6. **工程取舍**：离线降级、fail-open 过滤、prompt 版本化、Chroma collection 名带模型指纹（换模型不会读到旧向量）。

## 10. 局限与下一步

- **语料未覆盖司法解释、地方裁判口径、指导案例**，而真实劳动纠纷高度依赖地方标准（如社平工资 3 倍的计算基数）；
- **基准集较小（50 题）**，指标置信区间较宽，下一步可扩充到 200+ 并做题目级 bootstrap 置信区间；
- **零样本过滤有误删风险**，目前用 `min_keep=2` fail-open 兜底，可加"过滤前后指标对比"自动监控；
- **未做 query 级缓存与流式输出**，Chainlit 交互延迟在 CPU 上仍偏高；
- 计划：接入司法解释/地方性法规 → 增加 rerank 蒸馏小模型 → 用 RAGAS 风格指标补全答案忠实度评测。

## 11. 数据来源与免责声明

- 法条原文来自公开整理的法律文本仓库 [FFMRyan/Chinese-Labor-Law](https://github.com/FFMRyan/Chinese-Labor-Law)，经 `data/corpus/parse_laws.py` 结构化；
- 本项目仅用于技术研究与面试展示，**不构成任何正式法律意见**；
- 引用法条时请以国家法律法规数据库（flk.npc.gov.cn）的最新文本为准。

## 12. License

MIT
