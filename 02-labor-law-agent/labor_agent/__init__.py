"""劳动法咨询 Agent：3 个工具 + 原生 function calling / LangGraph 两种实现。

快速使用::

    from labor_agent import run_function_calling, run_langgraph, MockLLM

    result = run_function_calling("公司违法辞退我，月薪12000，工作3年7个月，能赔多少？")
    print(result.answer)
    for event in result.trace:
        print(event["index"], event["title"])
"""

from __future__ import annotations

from .corpus import (
    Article,
    Corpus,
    CorpusError,
    corpus_stats,
    get_article,
    get_corpus,
    search_articles,
)
from .function_calling import run_function_calling
from .langgraph_agent import (
    LangGraphUnavailable,
    graph_mermaid,
    is_langgraph_available,
    run_langgraph,
)
from .llm import (
    BaseLLM,
    LLMError,
    LLMResponse,
    LLMUnavailableError,
    MockLLM,
    OpenAICompatLLM,
    ToolCall,
    build_llm,
)
from .state import AgentResult, AgentState
from .tools import (
    TOOL_SCHEMAS,
    calc_severance,
    execute_tool_call,
    search_labor_law,
    verify_citation,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # corpus
    "Article",
    "Corpus",
    "CorpusError",
    "corpus_stats",
    "get_article",
    "get_corpus",
    "search_articles",
    # tools
    "TOOL_SCHEMAS",
    "calc_severance",
    "execute_tool_call",
    "search_labor_law",
    "verify_citation",
    # llm
    "BaseLLM",
    "LLMError",
    "LLMResponse",
    "LLMUnavailableError",
    "MockLLM",
    "OpenAICompatLLM",
    "ToolCall",
    "build_llm",
    # agents
    "AgentResult",
    "AgentState",
    "run_function_calling",
    "run_langgraph",
    "LangGraphUnavailable",
    "graph_mermaid",
    "is_langgraph_available",
]
