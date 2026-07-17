"""
RAG 에이전트 노드: 로컬 지식베이스에서 문서를 검색해 답변한다.

- rag_agent_node : ReAct LLM (도구 바인딩)
- rag_tools_node : 결정론적 custom ToolNode
  1. 쿼리에서 라우팅 지시어를 코드로 자동 제거 ("rag에서", "검색해줘" 등)
  2. 중복 쿼리를 메시지 히스토리 분석으로 차단
  → LLM 프롬프트 지시 불필요
"""
from __future__ import annotations

import re

# ── 라우팅 지시어 정제 패턴 ──────────────────────────────────────────────────
# "rag에서 autonomous testing 찾아줘" → "autonomous testing"
_ROUTING_PREFIX_RE = re.compile(
    r"^(rag에서|문서에서|로컬에서|자료에서|지식베이스에서|db에서)\s*",
    re.IGNORECASE,
)
_ROUTING_SUFFIX_RE = re.compile(
    r"\s*(검색해\s*(?:줘|주세요|줄래요?)?|찾아\s*(?:줘|주세요|줄래요?)?|"
    r"알려\s*(?:줘|주세요)?|설명해\s*(?:줘|주세요)?)$",
    re.IGNORECASE,
)

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import RAG_AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import search_documents
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_TOOLS = [search_documents]

# 한 번의 rag_agent 실행에서 허용하는 최대 tool call 횟수
# 초과 시 도구 없는 모델로 강제 답변 생성 → 무한루프 방지
_MAX_TOOL_CALLS = 2

_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    return _INTRO_RE.sub("", text).strip()


def _get_model_with_tools() -> ChatOllama:
    """도구 바인딩 모델 — tool call 허용."""
    base = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )
    return base.bind_tools(_TOOLS)


def _get_model_no_tools() -> ChatOllama:
    """도구 없는 모델 — 강제 최종 답변 생성용."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )


def rag_agent_node(state: ChatState) -> dict:
    """
    RAG 에이전트: 검색 → 답변 생성.
    tool call이 _MAX_TOOL_CALLS 회를 넘으면 강제로 최종 답변을 생성해 무한루프를 막는다.
    """
    messages = state["messages"]
    retry_count: int = state.get("rag_retry_count") or 0

    # 현재까지 tool call을 몇 번 했는지 카운트
    tool_call_count = sum(
        1 for msg in messages
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
    )

    system_prompt = RAG_AGENT_SYSTEM_PROMPT
    if retry_count > 0:
        system_prompt += (
            "\n\n⚠️ 이전 검색 결과가 충분하지 않았습니다. "
            "다른 키워드 조합으로 search_documents를 호출하세요. "
            "동일한 쿼리 반복 금지."
        )

    if tool_call_count >= _MAX_TOOL_CALLS:
        # 최대 횟수 초과 → 도구 없는 모델로 현재까지 검색 결과를 바탕으로 답변 강제 생성
        model = _get_model_no_tools()
        system_prompt += (
            "\n\n검색이 완료됐습니다. "
            "지금까지의 검색 결과를 바탕으로 최종 답변을 작성하세요. "
            "더 이상 도구를 호출하지 마세요."
        )
    else:
        model = _get_model_with_tools()

    full_messages = [SystemMessage(content=system_prompt), *messages]
    response = model.invoke(full_messages)

    if response.content and not response.tool_calls:
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}


# ── 결정론적 중복 쿼리 차단 ToolNode ────────────────────────────────────────

def rag_tools_node(state: ChatState) -> dict:
    """
    RAG 도구 실행 노드 (custom ToolNode 대체).

    메시지 히스토리를 분석해 이미 실행한 search_documents 쿼리를 코드 레벨에서 차단한다.
    프롬프트 지시 없이 결정론적으로 동작 — LLM 판단에 의존하지 않는다.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    messages = state["messages"]
    last_ai = messages[-1] if messages else None

    if not isinstance(last_ai, AIMessage) or not getattr(last_ai, "tool_calls", None):
        return {}

    # 이전 tool call에서 실제 검색된 쿼리 수집 (정규화: 소문자·strip)
    seen_queries: set[str] = set()
    for msg in messages[:-1]:  # last_ai 제외
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", None) or []):
                if tc.get("name") == "search_documents":
                    q = (tc.get("args") or {}).get("query", "").strip().lower()
                    if q:
                        seen_queries.add(q)

    tool_messages: list[ToolMessage] = []

    for tc in last_ai.tool_calls:
        tool_name = tc.get("name", "")
        tool_id   = tc.get("id", "")
        args      = tc.get("args") or {}

        if tool_name == "search_documents":
            query     = args.get("query", "").strip()

            # ── [결정론적] 라우팅 지시어 제거 ─────────────────────────────
            # "rag에서 autonomous testing 찾아줘" → "autonomous testing"
            # LLM 프롬프트 지시 없이 코드로 보장
            query = _ROUTING_PREFIX_RE.sub("", query).strip()
            query = _ROUTING_SUFFIX_RE.sub("", query).strip()
            if not query:
                query = args.get("query", "").strip()  # 정제 후 빈 쿼리면 원본 유지

            query_key = query.lower()

            if query_key in seen_queries:
                # ── 결정론적 차단: 동일 쿼리는 실행하지 않음 ────────────────
                content = (
                    f"[중복 쿼리 차단] '{query}'는 이미 검색했습니다. "
                    "다른 키워드를 사용하거나 현재 결과로 최종 답변을 작성하세요."
                )
            else:
                # ── 신규 쿼리: 실제 검색 실행 ────────────────────────────────
                content = search_documents.invoke({"query": query, "state": state})

        else:
            # search_documents 외 도구는 그대로 실행
            matched = next((t for t in _TOOLS if t.name == tool_name), None)
            content = matched.invoke(args) if matched else f"[알 수 없는 도구: {tool_name}]"

        tool_messages.append(ToolMessage(
            content=str(content),
            tool_call_id=tool_id,
            name=tool_name,
        ))

    return {"messages": tool_messages}

