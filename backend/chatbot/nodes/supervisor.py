"""
Supervisor 노드: 사용자 의도를 분석해 적절한 에이전트에 라우팅한다.
답변 생성은 하지 않고 라우팅 결정만 담당한다.

라우팅 대상:
  - rag_agent    : 문서·파일·지식베이스 검색 질문
  - email_agent  : 이메일 작성·초안 요청
  - summary_agent: 문서 요약 요청
  - image_agent  : 업로드된 이미지 시각적 분석 질문
  - direct_agent : 일반 상식·코딩 등 직접 답변 가능한 질문
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import SUPERVISOR_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.config import settings
from backend.database.repositories.document_repository import DocumentRepository

_log = logging.getLogger(__name__)
_doc_repo = DocumentRepository()

_ROUTE_VALUES = Literal["rag_agent", "email_agent", "summary_agent", "image_agent", "task_agent", "direct_agent"]

# ── 다중 작업 결정론적 감지 ─────────────────────────────────────────────────
# LLM 판단에 의존하지 않고 패턴으로 task_agent 강제 트리거

_SEQ_CONNECTORS = re.compile(
    r"(\s|,)*(하고|해서|아서|어서|한\s*다음|그\s*다음|다음으로|이어서|그러고\s*나서|그\s*후에?|또한|그리고|and\s+then|then)",
    re.IGNORECASE,
)
_ACTION_VERBS = re.compile(
    r"(검색|찾아|요약|정리|비교|분석|작성|써줘|만들어|설명|조사|보고|정리해|알려줘|해줘|해주세요)",
)


def _detect_multi_task(text: str) -> bool:
    """
    사용자 메시지에 2개 이상의 독립 작업이 포함되어 있는지 결정론적으로 감지한다.

    감지 기준 (하나라도 해당되면 multi-task):
    1. 번호 목록 패턴: '1. ... 2. ...' 또는 '첫째 ... 둘째 ...'
    2. 순차 접속사 + 동작 동사가 2회 이상
    3. 명시적 다중 작업 표현
    """
    # 1) 번호 목록 (줄바꿈 또는 한 줄 안에 연속으로 나오는 경우 모두 처리)
    numbered = re.findall(r"(?:^|\s)[1-9]\.", text, re.MULTILINE)
    if len(numbered) >= 2:
        return True
    # 첫째/둘째 형식
    if re.search(r"첫\s*째.{0,80}(둘\s*째|두\s*번\s*째)", text, re.DOTALL):
        return True

    # 2) 순차 접속사 + 동작 동사 2개 이상
    if _SEQ_CONNECTORS.search(text) and len(_ACTION_VERBS.findall(text)) >= 2:
        return True

    # 3) 명시적 표현
    explicit = [
        "순서대로", "하나씩 처리", "차례로", "다음을 해줘", "다음 작업",
        "할 일 목록", "작업 목록", "step by step", "one by one",
    ]
    return any(kw in text for kw in explicit)

# LLM이 출력할 수 있는 다양한 표현을 정규 에이전트 이름으로 매핑
_AGENT_ALIASES: dict[str, str] = {
    "rag": "rag_agent",
    "rag_agent": "rag_agent",
    "summary": "summary_agent",
    "summary_agent": "summary_agent",
    "email": "email_agent",
    "email_agent": "email_agent",
    "image": "image_agent",
    "image_agent": "image_agent",
    "task": "task_agent",
    "task_agent": "task_agent",
    "direct": "direct_agent",
    "direct_agent": "direct_agent",
}


# ── 다중 작업 결정론적 감지 ──────────────────────────────────────────────────

def _build_doc_context(session_id: str | None) -> str:
    """검색 가능한 문서 목록을 프롬프트용 텍스트로 생성한다."""
    try:
        docs = _doc_repo.list_active()
        visible = [
            d for d in docs
            if d.scope == "GLOBAL" or (d.scope == "SESSION" and str(d.session_id) == str(session_id))
        ]
        if not visible:
            return ""
        names = "\n".join(f"  - {d.file_name} ({d.scope})" for d in visible[:20])
        return f"\n\n[현재 지식베이스에 검색 가능한 문서]\n{names}\n→ 사용자 질문이 위 문서 내용과 관련될 가능성이 있으면 rag_agent를 선택하세요."
    except Exception:
        return ""


@tool
def route(agent: Literal["rag_agent", "email_agent", "summary_agent", "image_agent", "task_agent", "direct"]) -> str:
    """
    사용자 질문을 처리할 에이전트를 선택합니다.

    Args:
        agent: 라우팅할 대상.
               - rag_agent    : 문서·파일·지식베이스 검색이 필요한 질문
               - email_agent  : 이메일 작성·초안 요청
               - summary_agent: 문서 요약·정리 요청
               - image_agent  : 업로드된 이미지의 시각적 내용 분석
               - task_agent   : 여러 작업을 순서대로 처리하는 다중 작업 요청
               - direct       : 일반 상식, 코딩, 수학, 일상 대화 등
    """
    return agent


@lru_cache(maxsize=1)
def _get_supervisor_model() -> ChatOllama:
    base = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_ctx=settings.num_ctx,
        num_predict=32,
    )
    return base.bind_tools([route], tool_choice="any")


def _parse_route_fallback(last_message: str) -> _ROUTE_VALUES:
    """LLM tool call 실패 시 키워드 기반 폴백 라우팅."""
    text = last_message.lower()

    image_keywords = ["이미지", "사진", "그림", "photo", "image", "picture", "screenshot", "스크린샷"]
    email_keywords = ["이메일", "메일", "email", "mail", "초안", "draft", "발송", "수신", "보내기"]
    summary_keywords = ["요약", "정리해", "summarize", "summary", "핵심만", "간단히"]
    task_keywords = [
        "순서대로", "하나씩", "여러 가지", "다음을 해줘", "할 일들",
        "작업 목록", "할 일 목록", "코스안 작성", "순서대로 해줘",
        "첫째", "두번째", "세번째", "1.", "2.", "3.",
        "and then", "also do", "additionally", "step by step",
    ]
    doc_keywords = [
        "문서", "파일", "자료", "pdf", "검색", "찾아", "알려줘", "내용", "보고서",
        "document", "search", "find", "content", "report", "paper", "patent", "특허",
    ]

    if any(kw in text for kw in image_keywords):
        return "image_agent"
    if any(kw in text for kw in email_keywords):
        return "email_agent"
    if any(kw in text for kw in summary_keywords):
        return "summary_agent"
    if any(kw in text for kw in task_keywords):
        return "task_agent"
    if any(kw in text for kw in doc_keywords):
        return "rag_agent"
    return "direct_agent"


def supervisor_node(state: ChatState) -> dict:
    """
    Supervisor 노드: 라우팅 결정만 수행하고 state['next']를 설정한다.
    답변 생성은 하지 않는다 — 각 worker 에이전트가 담당한다.
    """
    messages = state["messages"]
    session_id = state.get("session_id")
    last_user_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            last_user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # 사용 가능한 문서 목록을 supervisor 프롬프트에 주입
    doc_context = _build_doc_context(session_id)
    system_prompt = SUPERVISOR_SYSTEM_PROMPT + doc_context

    # ── 다중 작업 선(先) 감지: LLM 호출 전에 결정론적으로 task_agent 강제 ──
    if _detect_multi_task(last_user_text):
        _log.debug("Supervisor → task_agent (multi-task detected, bypassing LLM)")
        return {"next": "task_agent"}

    next_agent: _ROUTE_VALUES = "direct_agent"
    try:
        supervisor_llm = _get_supervisor_model()
        resp = supervisor_llm.invoke(
            [SystemMessage(content=system_prompt), *messages]
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            raw = tool_calls[0].get("args", {}).get("agent", "direct")
            # .strip().lower() 후 alias 테이블로 정규화
            normalized = _AGENT_ALIASES.get(raw.strip().lower())
            next_agent = normalized if normalized else _parse_route_fallback(last_user_text)  # type: ignore[assignment]
        else:
            next_agent = _parse_route_fallback(last_user_text)
    except Exception as e:
        _log.warning("Supervisor routing failed, using fallback: %s", e)
        next_agent = _parse_route_fallback(last_user_text)

    _log.debug("Supervisor → %s (docs_context=%s)", next_agent, bool(doc_context))
    return {"next": next_agent}
