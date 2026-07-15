from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import CHAT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.config import settings

# 자기소개 패턴 — 텍스트 어디서나 제거
_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    """응답 전체에서 자기소개 문구를 모두 제거한다."""
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


def chat_node(state: ChatState) -> dict:
    """일반 대화 노드: 시스템 프롬프트 + 대화 기록으로 응답 생성."""
    model = _get_model()
    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)
    response.content = _strip_intro(response.content)
    return {"messages": [response]}
