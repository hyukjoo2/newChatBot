"""
웹 검색 에이전트 노드: 항상 먼저 검색하고 결과를 바탕으로 답변한다.

1단계: 규칙 기반으로 최적 쿼리 2개 생성 (LLM 스트리밍 오염 방지)
2단계: 두 쿼리 모두 검색 실행
3단계: 통합 결과로 답변 생성
→ 검색 실패 시 왜 못 찾았는지 설명하고 대안 제시
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import WEB_SEARCH_AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.tools import web_search
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_TOOLS = [web_search]
_log = logging.getLogger(__name__)

_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)
# 쿼리에서 메타 명령어 제거
_META_RE = re.compile(
    r"\s*(검색해\s*(?:줘|주세요|줄래요?)?|"
    r"찾아\s*(?:줘|주세요|줄래요?)?|"
    r"알려\s*(?:줘|주세요|줄래요?)?|"
    r"설명해\s*(?:줘|주세요|줄래요?)?)$",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    return _INTRO_RE.sub("", text).strip()


def _extract_context(state: ChatState) -> tuple[str, str]:
    """
    마지막 사용자 질문과, 이전 대화에서 파악된 주제(entity)를 반환한다.
    follow-up 질문 ("사진작가 중에서 찾아줘") 에도 컨텍스트를 반영한다.
    """
    messages = state["messages"]
    last_query = ""
    prev_ai_content = ""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and not last_query:
            content = msg.content
            last_query = content if isinstance(content, str) else str(content)
        elif isinstance(msg, AIMessage) and not prev_ai_content:
            c = msg.content
            prev_ai_content = (c if isinstance(c, str) else str(c))[:200]

    return last_query, prev_ai_content


def _plan_queries(question: str, context: str) -> list[str]:
    """
    규칙 기반으로 검색 쿼리 1~2개를 생성한다.
    LLM을 사용하지 않으므로 스트리밍 토큰 오염이 없다.
    """
    queries: list[str] = []

    # 1) 메타 명령어·조사 제거한 핵심 키워드 추출
    core = _META_RE.sub("", question.strip()).strip(" ,.")
    # "중에서", "안에서" 같은 조사도 제거
    core = re.sub(r"\s*(중에서?|안에서?|에서|에게서|로부터)\s*$", "", core).strip()
    if len(core) < 2:
        core = question.strip()

    # 2) "X에 대해(서)" 패턴 → 주제(X) 추출
    for pat in ["에 대해서", "에 대해", "에 대한", "에 관해서", "에 관해", "에 관한"]:
        if pat in core:
            topic = core.split(pat)[0].strip()
            if topic:
                queries.append(topic)
                queries.append(topic + " 소개 정보")
            break

    # 3) follow-up 질문 + 이전 컨텍스트 합성
    #    예: "사진작가 중에서 찾아줘" + 이전 답변 첫 줄 "이혁주..."
    if not queries and context:
        # 이전 AI 응답 첫 줄에서 첫 번째 의미 있는 명사구 추출 (최대 10자)
        first_line = re.split(r"[\n\r。.]", context)[0].strip()
        entity_match = re.match(r"^([가-힣a-zA-Z0-9]{2,10})", first_line)
        if entity_match:
            entity = entity_match.group(1)
            queries.append(entity + " " + core)
        queries.append(core)

    # 4) 기본
    if not queries:
        queries.append(core)
        if len(core) > 3:
            queries.append(core + " 정보")

    seen: list[str] = []
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.append(q)
    return seen[:2]


def _is_poor_result(result: str) -> bool:
    poor_indicators = [
        "검색 결과를 찾을 수 없습니다",
        "오류가 발생했습니다",
        "NAVER_CLIENT_ID",
    ]
    return any(ind in result for ind in poor_indicators) or len(result.strip()) < 50


@lru_cache(maxsize=1)
def _get_model() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )


def web_search_agent_node(state: ChatState) -> dict:
    """
    1) 규칙 기반으로 쿼리 계획 (LLM 없음 → 스트리밍 오염 없음)
    2) 두 쿼리 모두 검색
    3) 통합 결과로 답변 생성
    """
    user_query, prev_context = _extract_context(state)
    _log.info("web_search_agent: query=%r", user_query)

    queries = _plan_queries(user_query, prev_context)
    _log.info("planned queries: %s", queries)

    # ── 다중 검색 실행 ─────────────────────────────────────────────────────
    result_blocks: list[str] = []
    for q in queries:
        _log.info("searching: %r", q)
        result = web_search.invoke({"query": q})
        if not _is_poor_result(result):
            result_blocks.append(f"[검색어: {q}]\n{result}")

    if result_blocks:
        combined = "\n\n" + ("\n" + "=" * 40 + "\n").join(result_blocks)
    else:
        combined = "모든 쿼리에서 검색 결과를 찾을 수 없습니다."

    # ── 답변 생성 (마지막 질문만 전달, 오염된 히스토리 제외) ───────────────
    system_with_results = (
        WEB_SEARCH_AGENT_SYSTEM_PROMPT
        + f"\n\n[검색 결과]\n{combined}"
    )
    messages = [SystemMessage(content=system_with_results), HumanMessage(content=user_query)]
    response = _get_model().invoke(messages)

    if response.content:
        response.content = _strip_intro(strip_leaked_prompt(response.content))

    return {"messages": [response]}
