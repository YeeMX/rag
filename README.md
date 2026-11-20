# AI 应用作品集 · 劳动法领域三个项目

> 一个仓库讲完一条完整的 AI 应用能力链：
> **RAG 做知识 → Agent 做行动 → 视觉做多模态输入**，三个项目共享同一份真实法条语料与同一套评测方法论。

| 项目 | 一句话 | 技术栈 | 实测结果（CPU） | 目录 |
|---|---|---|---|---|
| **① 劳动法 RAG 咨询系统** | 4 部法律 357 条法条的检索增强问答，查询分解 + 双路召回 + RRF + 精排 + 零样本过滤，带 50 题消融评测 | LangChain · ChromaDB · sentence-transformers · BM25 · Chainlit · OpenAI | 50 题基准：**MRR 0.643 / Hit@5 0.840**（hybrid+rerank）；引用有效率 **100%** | [`01-labor-law-rag/`](01-labor-law-rag/README.md) |
| **② 劳动法工具调用 Agent** | 会调用「法条检索 / 赔偿计算 / 引用核验」3 个工具的 Agent，function calling 与 LangGraph 双实现 | OpenAI function calling · LangGraph · 自研工具集 | **106 个测试全绿**；5 个 demo 全流程跑通（2N=96000 精确计算） | [`02-labor-law-agent/`](02-labor-law-agent/README.md) |
| **③ 文档理解视觉 demo** | YOLO 版面检测 → 区域裁剪 → PaddleOCR → 结构化抽取，附 mAP / CER / 耗时实测 | Ultralytics YOLO · PaddleOCR · OpenCV | **mAP@0.5 0.9944 / mAP@0.5:0.95 0.8831**；OCR CER 0.0450；抽取准确率 0.9167；**65 个测试全绿** | [`03-vision-ocr-demo/`](03-vision-ocr-demo/README.md) |

三个项目串起来的故事：

```
劳动合同扫描件 ──③ 视觉/OCR──► 结构化字段 ──① RAG 检索+引用──► 带法条依据的结论
                                                      │
                                                      └──② Agent 计算赔偿/核验引用/给出行动建议
```

## 共同的设计原则

1. **可评测**：每个项目都有可复现的指标脚本，README 里的数字全部来自实际运行；
2. **可解释**：RAG 展示检索轨迹与引用，Agent 展示工具调用 trace，视觉展示区域与识别结果；
3. **可离线**：默认无 API Key、无网络也能跑通（自动降级并**如实标注**降级 backend），
   保证任何人 clone 后能立刻验证；
4. **不造假**：没有实测的数字一律写「未测量」；合成数据/离线降级模式都会明确声明。

## 目录结构

```
.
├── 01-labor-law-rag/          # 主项目（RAG）
│   ├── src/lrs/               # 检索/重排/过滤/生成核心包
│   ├── scripts/               # CLI 与评测脚本
│   ├── app/                   # Chainlit 前端
│   ├── data/                  # 法条语料 + 50 题基准 + 评测输出
│   └── tests/
├── 02-labor-law-agent/        # Agent demo
├── 03-vision-ocr-demo/        # YOLO + OCR demo
├── shared/                    # 共享法条语料（供 Agent 项目读取）
└── INTERVIEW_NOTES.md         # 面试讲解手册（三个项目通用）
```

## 环境

- Python **3.11**（Windows 实测；Linux/macOS 同理）
- 所有项目以 CPU 为基准，不需要 GPU；
- 各项目依赖见各自的 `requirements.txt` / `pyproject.toml`。

## 快速开始（零配置）

本机目录下已就绪三个 Python 环境（`.venv-rag` / `.venv-vision` / `.venv-paddle`，是指向已验证环境的目录联接，**未纳入 git**）和一键启动脚本：

```powershell
# 方式 A：双击 start.cmd（或命令行运行）—— 弹出菜单选择要跑哪个 demo
.\start.cmd

# 方式 B：命令行直接指定任务（不用菜单）
powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -Task rag-ui
powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -Task agent-demo
powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1 -Task vision-demo
```

菜单包含：RAG 网页版（Chainlit）、RAG 命令行提问/评测、Agent demo/测试、视觉 demo/测试、全部测试。

> 如果环境缺失或换机器：先运行一次 `scripts\setup_envs.ps1`（自动创建三个 venv 并安装锁定版本依赖，
> 约 2.5 GB，首次约 10~20 分钟）。

### 手动方式（等价命令）

```bash
# 项目 ①：RAG
cd 01-labor-law-rag
python scripts/cli.py ask "公司违法辞退我，月薪12000、工作3年7个月，能赔多少？"
python scripts/run_eval.py                               # 复现 50 题评测

# 项目 ②：Agent
cd ../02-labor-law-agent
python -m labor_agent.cli demo --no-color

# 项目 ③：视觉
cd ../03-vision-ocr-demo
python run_demo.py --skip-train                          # 复用已训练权重（约 11 分钟）
```

**文档入口**：[`docs/`](docs/交付说明.md) —— 使用指南（Chainlit 演示问题、项目①/②/③ 运行步骤）、评测结果汇总、生成结果快照。

## 关于简历/毕设的说明

本仓库是毕设《Intelligent Labor Law System Based on RAG & Multi-Channel Recall》的**可复现开源复刻版**：
保留了「查询分解与扩展 → 混合召回 → RRF → 重排 → 零样本过滤 → 引用约束生成」的完整架构，
但主动缩小了语料与评测规模以保证任何人能一键跑通。**不追求复现论文中的具体指标数字**，
各项目 README 里对指标差异有专门说明，面试时建议用"架构一致、环境不同、不作直接比较"来表述。

## License

MIT
