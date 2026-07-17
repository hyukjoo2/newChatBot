"""
Reasoning Agent 노드: 일반 질문에 답하며 능동적으로 추론·행동한다.
확실한 지식은 바로 답하고, 불확실하면 web_search / get_weather 도구를 호출한다.
답변 후 후속 행동이 도움이 될지 자율 판단 (날씨 → 맛집 검색 등).
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import web_search
from backend.chatbot.language_utils import strip_leaked_prompt, today_context
from backend.config import settings

_TOOLS = [web_search]  # get_weather는 weather_agent가 전담 처리

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
        num_predict=2048,   # 종합 시 답변이 길어지림 — settings.num_predict(1024)보다 충분하게
    )
    return base.bind_tools(_TOOLS)


import re as _re

_FOLLOWUP_RE = _re.compile(
    r"\[FOLLOWUP\](.*?):::(.*?)\[/FOLLOWUP\]",
    _re.DOTALL,
)

# LLM이 웹 검색 결과 번호를 인용하는 패턴 제거
# (출처 N) / (출처 N, M) — 한국어 형식
# ([N]) / ([N], [M]) — 대괄호 번호 형식
_FAKE_CITE_RE = _re.compile(
    r"\s*(?:"
    r"\(출처\s*[\d,\s]+\)"    # (출처 1, 5)
    r"|\(\[\d+\](?:,\s*\[\d+\])*\)"  # ([1], [3])
    r")",
    _re.IGNORECASE,
)


def _extract_followup(text: str) -> tuple[str, str | None]:
    """
    응답에서 [FOLLOWUP]...[/FOLLOWUP] 마커를 추출하고 제거한다.
    Returns: (cleaned_text, "question:::action") or (text, None)
    """
    m = _FOLLOWUP_RE.search(text)
    if not m:
        return text, None
    question = m.group(1).strip()
    action   = m.group(2).strip()
    cleaned  = _FOLLOWUP_RE.sub("", text).rstrip()
    return cleaned, f"{question}:::{action}"


def reasoning_agent_node(state: ChatState) -> dict:
    """추론 에이전트: 이전 검색 결과를 포함한 컨텍스트를 바탕으로 답변 생성."""
    model = _get_model()
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT + today_context()), *state["messages"]]
    response = model.invoke(messages)

    result: dict = {}

    if response.content and not getattr(response, "tool_calls", None):
        # [FOLLOWUP] 마커 제거 (안전망)
        cleaned, _ = _extract_followup(response.content)
        # (출처 N), (출처 1, 5) 등 가짜 인용 번호 제거
        cleaned = _FAKE_CITE_RE.sub("", cleaned)
        response.content = _strip_intro(strip_leaked_prompt(cleaned))

    result["messages"] = [response]
    return result
