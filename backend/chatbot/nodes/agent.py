"""
에이전트 노드: LLM이 search_documents 도구 호출 여부를 직접 판단한다.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import search_documents
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


def agent_node(state: ChatState) -> dict:
    """LLM 에이전트 노드: 도구 사용 여부를 모델이 직접 결정한다."""
    model = _get_model()
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)

    # 최종 텍스트 답변(tool call 없음)에만 자기소개 필터 적용
    if response.content and not response.tool_calls:
        response.content = _strip_intro(response.content)

    return {"messages": [response]}
