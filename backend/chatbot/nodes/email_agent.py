"""
이메일 초안 에이전트 노드: 사용자 요청을 바탕으로 이메일 초안을 작성한다.
도구 없이 직접 생성한다.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import EMAIL_AGENT_SYSTEM_PROMPT
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


def email_agent_node(state: ChatState) -> dict:
    """이메일 초안 에이전트 노드: 도구 없이 이메일 초안을 생성한다."""
    model = _get_model()
    messages = [SystemMessage(content=EMAIL_AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)

    if response.content and not getattr(response, "tool_calls", None):
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}
