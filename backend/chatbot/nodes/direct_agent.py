"""
직접 답변 에이전트 노드: 일반 질문에 답변한다.
확실한 지식(ucf54딩·수학·개념)은 마로 답하고,
특정 장소·인물·사실 등 확실하지 않으면 web_search 도구를 호출한다.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import web_search
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_TOOLS = [web_search]

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


def direct_agent_node(state: ChatState) -> dict:
    """직접 답변 노드: 확실한 지식은 바로 답하고, 불확실하면 web_search 도구를 호출한다."""
    model = _get_model()
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)

    if response.content and not getattr(response, "tool_calls", None):
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}
