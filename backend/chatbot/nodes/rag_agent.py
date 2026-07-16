"""
RAG 에이전트 노드: 로컬 지식베이스에서 문서를 검색해 답변한다.

search_documents 도구를 바인딩한 ReAct 패턴으로 동작한다.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import RAG_AGENT_SYSTEM_PROMPT
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


def rag_agent_node(state: ChatState) -> dict:
    """RAG 에이전트 노드: search_documents 도구를 사용해 문서 기반 답변을 생성한다."""
    model = _get_model()

    # Corrective RAG: 재시도 시 다른 키워드로 재검색하도록 힌트 추가
    retry_count: int = state.get("rag_retry_count") or 0
    system_prompt = RAG_AGENT_SYSTEM_PROMPT
    if retry_count > 0:
        system_prompt += (
            "\n\n⚠️ 이전 검색 결과가 질문에 충분히 답하지 못했습니다. "
            "반드시 다른 키워드 조합으로 search_documents 도구를 다시 호출하세요. "
            "동일한 쿼리를 반복하지 마세요."
        )

    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    response = model.invoke(messages)

    if response.content and not response.tool_calls:
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}
