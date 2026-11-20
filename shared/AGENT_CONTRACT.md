# 工作区共享约定（所有 Agent 必须遵守）

## 目录布局（仓库根目录 = `D:\Vibe Coding Projects`）

```
01-labor-law-rag/      # 主项目：Intelligent Labor Law System (RAG) —— 由 lead 负责，别碰
02-labor-law-agent/    # 小 Agent demo（function calling + LangGraph）—— agent-builder 负责
03-vision-ocr-demo/    # 图像 demo（YOLO + PaddleOCR）—— vision-builder 负责
shared/                # 共享资源（只读，禁止修改）
  corpus/
    raw/*.md                     # 四部劳动法全文（原始 Markdown）
    labor_law_articles.jsonl     # 357 条法条，结构化 JSONL
    provenance.json              # 数据来源
    parse_laws.py                # 解析脚本
  ENV.md                         # Python 解释器路径与已装依赖
  pip-install.log
README.md              # 顶层说明（lead 负责，别改）
INTERVIEW_NOTES.md     # 面试讲解稿（lead 负责，别改）
```

## 环境

- **唯一可用的 Python 解释器**：`D:\Vibe Coding Projects\.venv-rag\Scripts\python.exe`（Python 3.11）
- **绝对不要** `pip install` 任何包，也不要创建新 venv。需要新依赖时先问 lead。
- 若某个包缺失：先用标准库或已装包实现，在 README 的「依赖」一节声明即可。
- 所有脚本必须能在**离线、无 API key** 的情况下跑通（重模型不存在时自动降级并明确提示）。
- Windows + PowerShell；所有文件读写显式 `encoding="utf-8"`；控制台输出若含中文，脚本里用
  `sys.stdout.reconfigure(encoding="utf-8")`（Windows 控制台默认 GBK）。

## 法条语料（`shared/corpus/labor_law_articles.jsonl`，357 行）

每行一个 JSON：
```json
{"id": "lcl:第三十九条", "law_id": "lcl", "law": "中华人民共和国劳动合同法",
 "chapter": "第四章  劳动合同的解除和终止", "article_no": "第三十九条", "article_num": 39,
 "title": "《劳动合同法》第三十九条", "text": "劳动者有下列情形之一的……", "paragraphs": ["..."], "char_len": 197}
```
`law_id`：`lcl`=劳动合同法(98条) `ll`=劳动法(107条) `lda`=劳动争议调解仲裁法(54条) `sil`=社会保险法(98条)。

## 硬性质量要求

1. **不许造假**：README / 代码里所有指标数字必须是**实际跑出来的**，并注明是哪条命令生成的、
   什么环境（CPU/GPU）、什么数据集。跑不出来就写「未测量」，绝不编数。
2. 所有数字都要能复现：`python xxx.py` 一条命令产出结果文件（JSON/CSV/Markdown）。
3. 每个项目必须有：可运行的 CLI、`README.md`（含架构图 mermaid + 快速开始 + 实测结果 + 面试讲解要点）、
   `tests/` 下至少 3 个 pytest 用例且 `pytest` 能过。
4. 代码注释/文档用中文，变量名用英文；commit 风格见顶层 README。
5. 只用自己负责的目录；`shared/` 只读。
6. 别把 `.venv-*`、模型权重、大图片、日志提交进 git（各自项目的 `.gitignore` 负责）。

## 对外叙事（三个人设保持一致）

这是一个「AI 应用求职作品集」，三个项目共同讲一个故事：
**RAG 做知识（检索增强）→ Agent 做行动（工具调用/编排）→ 视觉 demo 做多模态落地**。
面试讲解要能回答：为什么这么设计、瓶颈在哪、指标怎么算、下一步怎么优化。
