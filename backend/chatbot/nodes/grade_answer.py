"""
Corrective RAG: 답변 품질 평가 노드.

RAG 에이전트가 생성한 답변이 사용자 질문에 충분히 답하고 있는지 LLM으로 평가한다.
- "relevant"    → 답변 적절, 종료
- "not_relevant" → 재검색 필요, rag_agent로 재라우팅
최대 재시도 횟수(MAX_RETRIES)에 도달하면 현재 답변을 그대로 반환한다.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import ANSWER_GRADER_PROMPT
from backend.chatbot.state import ChatState
from backend.config import settings

_log = logging.getLogger(__name__)

# rag_agent 재시도 최대 횟수 (초과 시 현재 답변을 그대로 수용)
MAX_RETRIES = 2


@lru_cache(maxsize=1)
def _get_grader_model() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,   # 평가는 결정론적으로
        num_ctx=settings.num_ctx,
        num_predict=32,    # {"grade": "relevant"} 수준의 짧은 출력
    )


def _extract_grade(text: str) -> str:
    """LLM 응답에서 grade 값을 파싱한다. 파싱 실패 시 'relevant' 반환."""
    try:
        # 응답이 JSON 블록 안에 있을 수도 있으므로 첫 번째 { } 범위만 추출
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        return parsed.get("grade", "relevant")
    except (ValueError, json.JSONDecodeError):
        _log.warning("Grader JSON parse failed — raw: %r", text[:200])
        return "relevant"


def grade_answer_node(state: ChatState) -> dict:
    """
    답변 품질 평가 노드.

    - answer_grade: "relevant" | "not_relevant"
    - rag_retry_count: 재시도 카운터 (not_relevant 일 때 증가)
    """
    retry_count: int = state.get("rag_retry_count") or 0
    messages = state["messages"]

    # 최대 재시도 초과 → not_relevant 반환 (graph.py가 web_search_agent로 폴백)
    if retry_count >= MAX_RETRIES:
        _log.info("MAX_RETRIES(%d) reached — falling back to web search", MAX_RETRIES)
        return {"answer_grade": "not_relevant"}

    # 마지막 사용자 질문 추출
    question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            question = content if isinstance(content, str) else str(content)
            break

    # 마지막 AI 최종 답변 추출 (tool_calls 없는 것)
    answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not question or not answer:
        _log.debug("grade_answer: question or answer missing — skipping grade")
        return {"answer_grade": "relevant"}

    # LLM으로 품질 평가
    model = _get_grader_model()
    prompt = ANSWER_GRADER_PROMPT.format(
        question=question,
        answer=answer[:1200],   # 너무 긴 답변은 앞부분만 평가
    )
    try:
        response = model.invoke([SystemMessage(content=prompt)])
        grade = _extract_grade(response.content.strip())
    except Exception:
        _log.exception("Grader model invocation failed — accepting answer")
        grade = "relevant"

    _log.info("Answer grade=%s  retry=%d/%d", grade, retry_count, MAX_RETRIES)

    new_retry = retry_count + 1 if grade == "not_relevant" else retry_count
    return {
        "answer_grade": grade,
        "rag_retry_count": new_retry,
    }
