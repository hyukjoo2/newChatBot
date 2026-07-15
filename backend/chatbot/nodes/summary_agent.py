"""
문서 요약 에이전트 노드: 지식베이스에서 문서를 검색해 핵심 내용을 요약한다.
search_documents 도구를 바인딩한 ReAct 패턴으로 동작한다.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import SUMMARY_AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import search_documents
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_TOOLS = [search_documents]

_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    return _INTRO_RE.sub("", text).strip()


@lru_cache(maxsize=1)
def _get_model() -> ChatOllama:
    base = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )
    return base.bind_tools(_TOOLS)


def summary_agent_node(state: ChatState) -> dict:
    """문서 요약 에이전트 노드: search_documents 도구로 문서를 검색하고 요약한다."""
    model = _get_model()
    messages = [SystemMessage(content=SUMMARY_AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)

    if response.content and not response.tool_calls:
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}
