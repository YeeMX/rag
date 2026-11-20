> **最快运行**：仓库根目录双击 `start.cmd` → 选 `1) RAG 网页版`；或 `powershell -File start.ps1 -Task rag-ui`。
> 本机环境已就绪（`.venv-rag`）；换机器先运行 `scripts\setup_envs.ps1` 安装依赖。

# 项目① 劳动法 RAG 系统 · 怎么看 / 怎么运行

> 目录：[`01-labor-law-rag/`](../01-labor-law-rag/README.md)
> 一句话：4 部法律 **357 条真实法条**的检索增强问答，链路是
> **查询分解/扩展 → BGE 稠密 + BM25 → RRF 融合 → cross-encoder 重排 → 零样本 LLM 过滤 → 带引用回答**。

---

## 1. 先看什么（不跑代码，5 分钟看懂）

| 顺序 | 打开什么 | 能看到什么 |
|---|---|---|
| 1 | [README.md](../01-labor-law-rag/README.md) §1~§2 | 为什么做、架构图（mermaid）、一次请求的 6 个阶段 |
| 2 | **Chainlit UI**：http://127.0.0.1:8000 | 直接提问，看到回答 + 引用法条 + 检索轨迹表（最直观） |
| 3 | [`data/eval/report.md`](../01-labor-law-rag/data/eval/report.md) | 50 题消融评测报告（每层配置的 MRR/Hit/Recall） |
| 4 | [`data/benchmark.jsonl`](../01-labor-law-rag/data/benchmark.jsonl) | 50 道人工场景题（含 gold 法条 + hard negative） |
| 5 | README §5 | 实测结果与「为什么 full 不是最优」的诚实分析 |

如果 UI 没在运行，看本文 §2.1 启动；或直接看 [`docs/生成结果/RAG评测报告.md`](生成结果/RAG评测报告.md) 的快照。

---

## 2. 怎么运行

### 2.1 启动 Chainlit 界面（最推荐）

```powershell
cd "D:\Vibe Coding Projects\labor-law-ai-portfolio\01-labor-law-rag"
$env:HF_HUB_DISABLE_XET = "1"      # 本机必须关 XET，否则模型下载会卡住
& "D:\Vibe Coding Projects\labor-law-ai-portfolio\.venv-rag\Scripts\chainlit.exe" run app\chainlit_app.py --host 127.0.0.1 --port 8000
```

浏览器打开 **http://127.0.0.1:8000**，直接输入问题（第一次打开需 20~30 秒加载模型）：

- `公司违法辞退我，月薪12000、工作3年7个月，能赔多少？`
- `入职8个月没签劳动合同，被辞退能同时要双倍工资和赔偿金吗？`
- `公司一直不给我缴社保，我可以要求补缴吗？`

详细界面说明见 [Chainlit 使用指南](Chainlit使用指南.md)。

### 2.2 命令行问答（不开浏览器）

```powershell
cd "D:\Vibe Coding Projects\labor-law-ai-portfolio\01-labor-law-rag"
$env:HF_HUB_DISABLE_XET = "1"
$py = "D:\Vibe Coding Projects\labor-law-ai-portfolio\.venv-rag\Scripts\python.exe"

# 单题：打印回答 + 引用 + 检索轨迹
& $py scripts\cli.py ask "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？"

# 交互式连续提问（exit 退出）
& $py scripts\cli.py chat

# JSON 输出
& $py scripts\cli.py ask "公司违法辞退我" --json

# 建索引（一般首次自动建；换模型/改语料后手动重建）
& $py scripts\cli.py build --force
```

### 2.3 复现 50 题评测（约 15 分钟，CPU）

```powershell
cd "D:\Vibe Coding Projects\labor-law-ai-portfolio\01-labor-law-rag"
$env:HF_HUB_DISABLE_XET = "1"
& "D:\Vibe Coding Projects\labor-law-ai-portfolio\.venv-rag\Scripts\python.exe" scripts\run_eval.py
```

产出：`data/eval/retrieval_results.json`、`answer_results.json`、`report.md`。

### 2.4 换到别的机器（从 GitHub clone 后）

```bash
cd 01-labor-law-rag
pip install -r requirements.txt
python scripts/cli.py ask "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？"
chainlit run app/chainlit_app.py -w
```

> 没有网络也能跑：embedding/rerank 自动降级为哈希/词面实现，LLM 自动降级为离线 Mock，
> 报告里会如实标注 backend。

---

## 3. 会看到什么

### 3.1 命令行输出（真实节选）

```
查询理解：单意图，触发 1 个术语扩展 → 解除劳动合同
回答（LLM 模式：mock）：
[MOCK] 以下是基于检索到的法条原文的抽取式回答（未调用真实大模型）：
1. 《劳动合同法》第八十七条规定：用人单位违反本法规定解除或者终止劳动合同的，
   应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。
引用法条：
  · 《劳动合同法》第八十七条：……
检索明细（重排后，★=通过零样本过滤）：
  ★ #1  《劳动合同法》第八十七条   d=0.70 b=0.0 rrf=0.57 rr=0.87 src=dense
耗时(ms)：{"plan":0.3,"retrieve":68.7,"rerank":1027.3,"filter":1.0,"generate":0.2,"total":1097.5}
```

### 3.2 评测结果（当前实测，`data/eval/report.md`）

| 配置 | MRR | Hit@5 | Recall@5 |
|---|---|---|---|
| dense_only | 0.4987 | 0.7600 | 0.4757 |
| bm25_only | 0.6123 | 0.7600 | 0.5050 |
| hybrid_rrf | 0.5740 | 0.7600 | 0.5100 |
| **hybrid_rerank（最优）** | **0.6427** | **0.8400** | **0.5267** |
| full（含查询理解+Mock过滤） | 0.6267 | 0.7400 | 0.4383 |

答案层：引用有效率 **100%**、引用命中率 34.0%、gold 覆盖率 43.8%。

---

## 4. 常见问题

| 现象 | 解决 |
|---|---|
| UI 报 `No module named 'lrs'` | 已修复（Chainlit 2.12 的 `sys.path.pop(0)` 问题）；若自行改动过，设 `$env:PYTHONPATH="src"` |
| 模型下载卡住 | 必须设 `$env:HF_HUB_DISABLE_XET="1"`（本机 hf-xet 会挂死） |
| 第一次很慢 | 首次加载 BGE + reranker 约 20~30 秒，之后每条 2~4 秒 |
| 回答带 `[MOCK]` | 离线模式；配置 `OPENAI_API_KEY` 后加 `--real-llm` 即用真实模型 |
| 想换检索模型 | 改 `src/lrs/config.py` 的 `dense_model` / `rerank_model`，再 `cli.py build --force` |

---

## 5. 文件地图

```
01-labor-law-rag/
├── app/chainlit_app.py        # UI（复用 RAGService，与评测同源）
├── src/lrs/
│   ├── config.py              # 全部参数与消融预设
│   ├── corpus.py              # 语料加载/切分（LangChain splitter）
│   ├── embeddings.py          # BGE + 离线哈希降级
│   ├── lexical.py             # BM25（单字+二元组分词）
│   ├── vector_store.py        # ChromaDB（collection 名带模型指纹）
│   ├── retriever.py           # 稠密+BM25 加权 RRF
│   ├── reranker.py            # cross-encoder 精排
│   ├── query_processing.py    # 查询分解 + 术语扩展
│   ├── filter.py              # 零样本适用性过滤（fail-open）
│   ├── prompts.py             # Prompt 工程（版本化）
│   ├── llm.py                 # OpenAI 兼容 + 离线 Mock
│   ├── service.py             # 装配 + 完整 answer() 链路
│   └── metrics.py             # MRR/Hit/Recall/nDCG（纯函数）
├── scripts/
│   ├── cli.py                 # build / ask / chat / eval
│   ├── evaluate.py            # 消融评测详细入口
│   └── run_eval.py            # 一键评测
├── data/
│   ├── corpus/                # 357 条法条 + 解析脚本
│   ├── benchmark.jsonl        # 50 题基准
│   └── eval/                  # ★ 评测报告与逐题明细
└── tests/                     # 37 个 pytest 用例
```
