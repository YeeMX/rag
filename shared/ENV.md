# Python 环境说明

## 解释器

唯一解释器（Python 3.11.9）：

```
D:\Vibe Coding Projects\.venv-rag\Scripts\python.exe
```

不要 `pip install`，不要新建 venv（除非 lead 明确要求）。缺包先问 lead。

## 已安装依赖（随 lead 的安装任务完成情况更新）

RAG / Agent 相关：

- torch（CPU）
- sentence-transformers、transformers
- chromadb
- rank-bm25
- langchain、langchain-openai、langchain-community
- langgraph
- chainlit
- openai
- pytest、pyyaml、rich、tqdm、pypdf

视觉相关（另有专用环境，见下）：

- ultralytics 8.4.216、torch 2.14.1+cpu、opencv-python（`.venv-vision`）
- paddlepaddle 3.3.1、paddleocr 3.7.0（`.venv-paddle`，Python 3.11）

## 视觉 demo 的可用环境（03-vision-ocr-demo）

已有两个现成环境（Python 3.11，无 GPU）：

```
D:\Vibe Coding Projects\.venv-vision\Scripts\python.exe   # ultralytics + torch + opencv
D:\Vibe Coding Projects\.venv-paddle\Scripts\python.exe   # paddlepaddle + paddleocr + numpy
```

注意：这两个环境里**没有** pytest（需要就用 `python -m unittest` 或把测试写成可直接 `python tests/xxx.py` 运行）。
优先复用这两个环境，不要试图在 `.venv-rag` 里装 paddle。
