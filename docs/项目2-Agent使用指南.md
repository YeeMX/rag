> **最快运行**：仓库根目录双击 `start.cmd` → 选 `4) Agent demo`；或 `powershell -File start.ps1 -Task agent-demo`。
> 本机环境已就绪（`.venv-rag`）；换机器先运行 `scripts\setup_envs.ps1` 安装依赖。

# 项目② 工具调用 Agent · 怎么看 / 怎么运行

> 目录：[`02-labor-law-agent/`](../02-labor-law-agent/README.md)
> 一句话：能调用 **3 个工具**（法条检索 / 赔偿精算 / 引用核验）的劳动法 Agent，
> 用 **原生 function calling** 和 **LangGraph StateGraph** 两种方式实现。

---

## 1. 先看什么（5 分钟看懂项目）

| 顺序 | 看什么 | 能看到什么 |
|---|---|---|
| 1 | [README.md](../02-labor-law-agent/README.md) 第 1~2 节 | 为什么做、架构图（两种实现的 mermaid） |
| 2 | 跑一次 demo（见 §2） | 完整 trace：思考 → 工具调用 → 参数 → 返回 → 最终答案 |
| 3 | README「一次完整 trace」章节 | 真实输出截图级文本，不用跑也能看 |
| 4 | README「两种实现对比」表 | function calling vs LangGraph 的差异与取舍 |
| 5 | `tests/` | 106 个测试，覆盖金额精算、坏 JSON、未知工具、轮数上限 |

**没有任何界面**：这个项目是命令行 demo，输出就是终端里的 trace（适合面试现场直接跑）。

---

## 2. 怎么运行（复制即可）

### 2.1 用本机现成环境（当前机器）

```powershell
cd "D:\Vibe Coding Projects\labor-law-ai-portfolio\02-labor-law-agent"

# 5 个内置问题，打印完整 trace（离线 Mock，不需要 API key，约 5 秒）
& "D:\Vibe Coding Projects\labor-law-ai-portfolio\.venv-rag\Scripts\python.exe" -m labor_agent.cli demo --no-color
```

### 2.2 换到别的机器（从 GitHub clone 后）

```bash
cd 02-labor-law-agent
pip install -r requirements.txt          # rank-bm25 / langgraph / openai
python -m labor_agent.cli demo --no-color
```

### 2.3 其他常用命令

```powershell
$py = "D:\Vibe Coding Projects\labor-law-ai-portfolio\.venv-rag\Scripts\python.exe"

# 问单个问题
& $py -m labor_agent.cli ask "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？"

# 指定用 LangGraph 实现（默认 function_calling）
& $py -m labor_agent.cli ask "公司违法辞退我" --framework langgraph

# 打印 LangGraph 的图定义（mermaid，可贴进文档）
& $py -m labor_agent.cli graph

# JSON 输出（便于脚本/截图）
& $py -m labor_agent.cli ask "公司违法辞退我" --json

# 跑测试（106 个用例）
& $py -m pytest tests -q
```

---

## 3. 会看到什么（真实输出节选）

`demo` 的第 1 题：

```
❓ 问题：公司违法辞退我，月薪12000，工作3年7个月，能赔多少？依据哪一条？
🧭 框架：function_calling｜模式：MOCK 离线模拟（规则引擎，非真实大模型）｜模型轮数：2｜工具调用：2 次
[2] 🧠 第 1 轮模型决策
    → 工具调用 calc_severance({"scenario": "unlawful_termination", "monthly_salary": 12000.0, "years": 3, "months": 7})
    → 工具调用 search_labor_law({"query": "违法解除劳动合同 赔偿金 经济补偿 第四十七条 第八十七条", "top_k": 3})
[3] 🛠 工具返回：calc_severance
    ✓ 经济补偿金 N = 48000.00 元；违法解除赔偿金 2N = 96000.00 元
[6] ✅ 最终回答（约 1000 字）
```

关键点：**金额是确定性代码算的**（3 年 7 个月 → N=4 → 2N=96000），LLM 只负责选工具、填参数。

---

## 4. 真实大模型模式（可选）

```powershell
$env:OPENAI_API_KEY  = "sk-你的key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"   # 可选
$env:OPENAI_MODEL    = "gpt-4o-mini"                  # 可选
& $py -m labor_agent.cli ask "公司违法辞退我，月薪12000，工作3年7个月，能赔多少？" --openai
```

> 无 API key 时默认走 Mock（规则引擎），README 里明确标注"不是真实大模型输出"。

---

## 5. 常见问题

| 现象 | 解决 |
|---|---|
| `ModuleNotFoundError: labor_agent` | 必须在 `02-labor-law-agent` 目录下执行 |
| 没有颜色 / 中文乱码 | 加 `--no-color`，并设 `$env:PYTHONIOENCODING="utf-8"` |
| 想确认算得对不对 | 看 `tests/test_tools.py`（每个金额都有断言） |
| 想换模型 | 设 `OPENAI_*` 环境变量并加 `--openai` |

---

## 6. 文件地图

```
02-labor-law-agent/
├── labor_agent/
│   ├── tools.py            # 3 个工具（JSON Schema + 实现）
│   ├── function_calling.py # 方案 A：原生 function calling 循环
│   ├── langgraph_agent.py  # 方案 B：LangGraph StateGraph
│   ├── llm.py              # MockLLM / OpenAI 兼容 LLM
│   ├── corpus.py           # 读取共享法条语料 + BM25
│   ├── cli.py              # 命令行入口（ask / demo / graph）
│   └── state.py            # Agent 状态定义
├── tests/                  # 106 个 pytest 用例
└── README.md               # 设计、trace、对比表、面试要点
```
