"""
Corrective RAG: 답변 품질 평가 노드.

LLM 대신 결정론적 패턴 매칭으로 평가한다.
- LLM 호출 없음 → 빠르고 예측 가능, 무한루프/크래시 없음
- "없습니다/찾을 수 없습니다" 등 실패 패턴 → not_relevant
- 충분한 답변 → relevant
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from backend.chatbot.state import ChatState

_log = logging.getLogger(__name__)

MAX_RETRIES = 2

# 답변이 정보를 찾지 못했음을 나타내는 패턴 목록
_NOT_FOUND_PATTERNS = [
    "찾을 수 없습니다",
    "관련 문서를 찾을 수 없",
    "관련 정보가 없",
    "관련 내용이 없",
    "해당 정보가 없",
    "없습니다",           # broad but effective
    "찾지 못했습니다",
    "검색 결과가 없",
    "문서에 없",
    "정보를 찾을 수",
    "[중복 쿼리 차단]",   # rag_tools_node가 중복 차단한 경우
    "no results",
    "not found",
]

# 답변으로 보기에 너무 짧은 기준 (단순 "없음" 류 제외)
_MIN_ANSWER_LEN = 60


def grade_answer_node(state: ChatState) -> dict:
    """
    결정론적 답변 품질 평가.

    LLM 없이 패턴 매칭으로 즉시 판단:
    - "없습니다" 류 패턴 포함 → not_relevant
    - 너무 짧은 답변       → not_relevant
    - 그 외               → relevant
    """
    retry_count: int = state.get("rag_retry_count") or 0
    messages = state["messages"]

    if retry_count >= MAX_RETRIES:
        _log.info("MAX_RETRIES(%d) reached — falling back to web search", MAX_RETRIES)
        return {"answer_grade": "not_relevant"}

    # 마지막 AI 최종 답변 (tool_calls 없는 것)
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not answer:
        return {"answer_grade": "relevant"}

    # ── 결정론적 판단 ────────────────────────────────────────────────────────
    grade = "relevant"

    if len(answer.strip()) < _MIN_ANSWER_LEN:
        grade = "not_relevant"
        _log.info("grade=not_relevant (too short: %d chars)", len(answer.strip()))
    else:
        for pattern in _NOT_FOUND_PATTERNS:
            if pattern in answer:
                grade = "not_relevant"
                _log.info("grade=not_relevant (pattern matched: %r)", pattern)
                break

    _log.info("Answer grade=%s  retry=%d/%d", grade, retry_count, MAX_RETRIES)
    new_retry = retry_count + 1 if grade == "not_relevant" else retry_count
    return {
        "answer_grade": grade,
        "rag_retry_count": new_retry,
    }


    new_retry = retry_count + 1 if grade == "not_relevant" else retry_count
    return {
        "answer_grade": grade,
        "rag_retry_count": new_retry,
    }
