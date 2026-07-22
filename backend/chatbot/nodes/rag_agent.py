"""
RAG 에이전트 노드: 로컬 지식베이스에서 문서를 검색해 답변한다.

- rag_agent_node : ReAct LLM (도구 바인딩)
- rag_tools_node : 결정론적 custom ToolNode
  1. 쿼리에서 라우팅 지시어를 코드로 자동 제거 ("rag에서", "검색해줘" 등)
  2. 중복 쿼리를 메시지 히스토리 분석으로 차단
  → LLM 프롬프트 지시 불필요
"""
from __future__ import annotations

import logging
import re

# ── 라우팅 지시어 정제 패턴 ──────────────────────────────────────────────────
# "rag에서 autonomous testing 찾아줘" → "autonomous testing"
_ROUTING_PREFIX_RE = re.compile(
    r"^(rag에서?|문서에서|로컬(\s*디비)?에서?|자료에서|지식베이스에서?|디비에서|db에서)\s*",
    re.IGNORECASE,
)
_ROUTING_SUFFIX_RE = re.compile(
    r"\s*(검색해\s*(?:줘|주세요|줄래요?)?|찾아\s*(?:줘|주세요|줄래요?)?|"
    r"알려\s*(?:줘|주세요)?|설명해\s*(?:줘|주세요)?)$",
    re.IGNORECASE,
)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import RAG_AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import search_documents
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_TOOLS = [search_documents]

_log = logging.getLogger(__name__)

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


def _get_model_with_tools(force_call: bool = False) -> ChatOllama:
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
    """도구 없는 모델 — 강제 최종 답변 생성용. temperature=0 으로 결과 일관성 보장."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )


def rag_agent_node(state: ChatState) -> dict:
    """
    RAG 에이전트: 검색 → 답변 생성.

    구조:
      1. search_documents.func() 로 직접 검색 (LLM 도구 호출 결정 불필요)
      2. 검색 결과를 시스템 프롬프트에 주입 (ToolMessage 없음 — Ollama 호환성)
      3. LLM은 검색 결과를 읽고 최종 답변만 생성
    """
    messages = list(state["messages"])
    retry_count: int = state.get("rag_retry_count") or 0

    # ── 1. 검색 실행 ────────────────────────────────────────────────────────
    # 마지막 HumanMessage에서 쿼리 추출
    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            user_query = c if isinstance(c, str) else str(c)
            break
    clean_query = _ROUTING_PREFIX_RE.sub("", user_query).strip()
    # 접미사 제거: "검색해줘", "알려줘" → 제거
    clean_query = _ROUTING_SUFFIX_RE.sub("", clean_query).strip()
    # "에 대해", "에 대한" 제거: "암호기술의 분류에 대해" → "암호기술의 분류"
    # 제거하지 않으면 임베딩이 기술 문서에서 멀어져 관련 없는 문서가 상위에 올 수 있음
    import re as _re
    clean_query = _re.sub(r"\s*에\s*대해?한?\s*$", "", clean_query).strip()
    if not clean_query:
        clean_query = user_query

    if retry_count > 0:
        # 재시도: 이전과 다른 키워드 사용 (영어 변환 시도)
        _log.info("rag_agent retry=%d query=%r", retry_count, clean_query[:60])
    else:
        _log.info("rag_agent: 직접 검색 query=%r", clean_query[:60])

    search_result = search_documents.func(query=clean_query, state=state)
    _log.debug("rag_agent: 검색 결과 %d자", len(search_result))

    # ── 2. 시스템 프롬프트에 결과 주입 ─────────────────────────────────────
    found = "찾을 수 없습니다" not in search_result and len(search_result) > 50

    if found:
        # 검색 결과가 있을 때: 단순 명확한 지시 — 복잡한 규칙 없음
        # (복잡한 RAG_AGENT_SYSTEM_PROMPT는 LLM을 혼란시킴)
        body = search_result.split("[document_refs:")[0]
        system_prompt = (
            "아래 문서 내용을 참고해서 사용자 질문에 한국어로 답하세요.\n"
            "문서 내용 외의 정보를 지어내지 마세요.\n\n"
            f"{body}"
        )
    else:
        system_prompt = RAG_AGENT_SYSTEM_PROMPT + "\n\n로컬 문서에서 관련 내용을 찾지 못했습니다. '관련 문서를 찾지 못했습니다'라고 답하세요."

    # ── 3. LLM 답변 생성 ────────────────────────────────────────────────────
    model = _get_model_no_tools()
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

