"""
직접 답변 에이전트 노드: 문서 검색 없이 일반 질문에 직접 답변한다.
supervisor 가 'direct_agent' 로 라우팅할 때 호출된다.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    return _INTRO_RE.sub("", text).strip()


@lru_cache(maxsize=1)
def _get_model() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )


def direct_agent_node(state: ChatState) -> dict:
    """직접 답변 노드: 도구 없이 일반 지식으로 답변한다."""
    model = _get_model()
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)

    if response.content and not getattr(response, "tool_calls", None):
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}
