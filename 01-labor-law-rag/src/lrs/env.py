"""环境变量与运行时兼容性设置。

必须在**导入 sentence_transformers / huggingface_hub 之前**调用，因此在包入口
（cli/evaluate/chainlit）和 embeddings/reranker 模块顶部都会 import 本模块。

为什么要关 hf-xet：在部分 Windows + 代理 / 企业网络环境下，hf-xet 的分块下载
会创建一个 0 字节的 `.incomplete` 文件后永久阻塞（表现为进程挂起、无 CPU 占用）。
关闭后走经典 HTTP 下载，速度略慢但稳定。
"""

from __future__ import annotations

import os


def configure_runtime() -> None:
    # 关闭 XET 分块下载，避免在部分网络环境挂死
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # 关掉 tokenizers 的并行 fork 警告（Windows 上 fork 会报 warning，不影响结果）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


configure_runtime()
