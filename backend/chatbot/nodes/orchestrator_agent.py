"""
Orchestrator Agent: supervisor + orchestrator 통합 노드.

역할:
  1. 사용자 의도 파악 + 작업 계획 수립 (LLM, 일회성)
  2. 각 작업 결과 평가 + 계획 업데이트 (결정론적)
  3. 다음 에이전트로 라우팅 (결정론적, 계획 기반)

계획 형식 (결정론적 파싱):
  [PLAN]
  번호|에이전트|작업설명
  [/PLAN]
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import ORCHESTRATOR_PLAN_PROMPT
from backend.chatbot.state import ChatState, OrchestratorTask
from backend.chatbot.language_utils import today_context
from backend.config import settings
from backend.database.repositories.document_repository import DocumentRepository

_log = logging.getLogger(__name__)
_doc_repo = DocumentRepository()

# ── 유효 에이전트 이름 ───────────────────────────────────────────────────────
_VALID_AGENTS = {
    "rag_agent", "web_search_agent", "email_agent",
    "summary_agent", "image_agent", "reasoning_agent",
}

# 에이전트 이름 정규화 (LLM이 다양한 표현을 쓸 수 있음)
_AGENT_ALIASES: dict[str, str] = {
    "rag": "rag_agent",
    "rag_agent": "rag_agent",
    "web": "web_search_agent",
    "web_search": "web_search_agent",
    "web_search_agent": "web_search_agent",
    "websearch": "web_search_agent",
    "email": "email_agent",
    "email_agent": "email_agent",
    "summary": "summary_agent",
    "summary_agent": "summary_agent",
    "image": "image_agent",
    "image_agent": "image_agent",
    "reasoning": "reasoning_agent",
    "reasoning_agent": "reasoning_agent",
    "direct": "reasoning_agent",
    "direct_agent": "reasoning_agent",
    "weather": "weather_agent",
    "weather_agent": "weather_agent",
}

# ── 계획 파싱 ────────────────────────────────────────────────────────────────
_PLAN_BLOCK_RE = re.compile(r"\[PLAN\](.*?)\[/PLAN\]", re.DOTALL | re.IGNORECASE)
_TASK_LINE_RE  = re.compile(r"^\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(.+)\s*$")

def _parse_plan(text: str) -> list[OrchestratorTask]:
    """LLM 출력에서 [PLAN]...[/PLAN] 블록을 결정론적으로 파싱한다."""
    block = _PLAN_BLOCK_RE.search(text)
    if not block:
        return []

    tasks: list[OrchestratorTask] = []
    for line in block.group(1).splitlines():
        m = _TASK_LINE_RE.match(line)
        if not m:
            continue
        task_id   = int(m.group(1))
        raw_agent = m.group(2).strip().lower()
        agent     = _AGENT_ALIASES.get(raw_agent, "reasoning_agent")
        desc      = m.group(3).strip()
        tasks.append({
            "id": task_id,
            "agent": agent,
            "description": desc,
            "status": "pending",
            "result": None,
        })

    return tasks


# ── 결과 품질 판단 (결정론적) ────────────────────────────────────────────────
_FAILURE_PATTERNS = [
    "찾을 수 없습니다",
    "관련 문서를 찾을 수 없",
    "관련 정보가 없",
    "없습니다",
    "찾지 못했습니다",
    "검색 결과가 없",
    "[중복 쿼리 차단]",
    "not found",
]

def _is_failed(result: str | None) -> bool:
    if not result or len(result.strip()) < 60:
        return True
    return any(p in result for p in _FAILURE_PATTERNS)


# ── LLM 기반 후속 작업 제안 ──────────────────────────────────────────────────
# 하드코딩 템플릿이 아닌 LLM이 맥락을 보고 판단.
# 소형 호출(no_think, 128토큰)로 빠르게 처리.

_FOLLOWUP_JUDGE_PROMPT = """아래 사용자 질문과 AI 응답을 보고, 사용자에게 도움이 될 구체적인 후속 작업이 있는지 판단하세요.

사용자 질문: {query}
AI 응답 요약: {summary}

[판단 원칙]
기본값은 "후속 없음"입니다. 아래 경우에만 제안하세요:
- RAG 검색 결과가 없거나 부족 → 웹 검색 추가 가능
- 날씨 정보 → 사용자 상황에 맞는 후속 행동 (세차 계획이면 다른 날 추천, 나들이면 코스 추천 등)
- 특정 장소/인물 정보 → 관련된 추가 정보
- 그 외 자연스러운 다음 단계가 명확한 경우

후속이 필요 없으면 반드시 아무것도 출력하지 마세요.
후속이 있다면 아래 형식 하나만 출력하세요:
[FOLLOWUP]구체적인 질문:::실행할 작업[/FOLLOWUP]

예시:
[FOLLOWUP]웹에서 추가 검색할까요?:::autonomous testing 최신 논문 웹 검색[/FOLLOWUP]
[FOLLOWUP]내일 날씨도 확인할까요?:::광명 내일 날씨[/FOLLOWUP]
/no_think"""

_FOLLOWUP_RE = re.compile(r"\[FOLLOWUP\](.*?):::(.*?)\[/FOLLOWUP\]", re.DOTALL)

# 정크 필터: LLM이 플레이스홀더를 그대로 출력한 경우 버림
_FOLLOWUP_JUNK = [
    "사용자에게 보여줄", "실제_질문", "실제_작업",
    "검색/실행", "질문:::작업", "내용:::내용",
]


def _llm_derive_followup(agent: str, user_query: str, result: str) -> str | None:
    """
    LLM이 맥락 전체를 보고 후속 작업을 제안한다.
    에이전트 타입·결과 패턴 등 어떤 하드코딩 기준도 없음.

    유일한 결정론적 가드:
    - 결과가 너무 짧으면 스킵 (의미 있는 내용이 없음)
    - LLM 출력이 정크/잘못된 형식이면 버림
    """
    if not result or len(result.strip()) < 30:
        return None

    try:
        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_ctx=2048,
            num_predict=100,
        )
        prompt = _FOLLOWUP_JUDGE_PROMPT.format(
            query=user_query[:200],
            summary=result[:200],
        )
        response = model.invoke([SystemMessage(content=prompt)])
        raw = response.content.strip() if response.content else ""
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not clean:
            return None

        m = _FOLLOWUP_RE.search(clean)
        if not m:
            return None

        question = m.group(1).strip()
        action   = m.group(2).strip()

        # 결정론적 정크 필터 (LLM이 예시 텍스트를 그대로 출력한 경우)
        for junk in _FOLLOWUP_JUNK:
            if junk in question or junk in action:
                _log.debug("followup junk filtered: %r", question[:40])
                return None

        if len(question) < 5 or len(question) > 80:
            return None

        return f"{question}:::{action}"

    except Exception as e:
        _log.debug("followup LLM call failed: %s", e)
        return None

    # ── 2단계: LLM 생성 ────────────────────────────────────────────────────
    try:
        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_ctx=2048,
            num_predict=100,
        )
        prompt = _FOLLOWUP_JUDGE_PROMPT.format(
            query=user_query[:200],
            summary=result[:200],
        )
        response = model.invoke([SystemMessage(content=prompt)])
        raw = response.content.strip() if response.content else ""
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not clean:
            return None

        m = _FOLLOWUP_RE.search(clean)
        if not m:
            return None

        question = m.group(1).strip()
        action   = m.group(2).strip()

        # ── 3단계: 정크 필터 ────────────────────────────────────────────────
        for junk in _FOLLOWUP_JUNK:
            if junk in question or junk in action:
                _log.debug("followup junk filtered: %r", question[:40])
                return None

        if len(question) < 5 or len(question) > 80:
            return None

        return f"{question}:::{action}"

    except Exception as e:
        _log.debug("followup LLM call failed: %s", e)
        return None


# ── 키워드 기반 폴백 라우팅 (LLM 실패 시) ───────────────────────────────────
_META_SEQ_RE    = re.compile(r"(\s|,)*(하고|해서|한\s*다음|그\s*다음|다음으로|이어서|그리고)", re.IGNORECASE)
_META_ACTION_RE = re.compile(r"(검색|찾아|요약|정리|비교|분석|작성|써줘|만들어|설명|조사|해줘|해주세요)")

_DOC_KEYWORDS = {"문서에서", "파일에서", "업로드한", "첨부한", ".pdf", ".docx", ".txt", "특허"}
_WEB_KEYWORDS = {
    "어디", "위치", "주소", "영업", "운영", "입장료", "가격", "전화번호",
    "최신", "최근", "요즘", "현재", "오늘", "지금", "올해", "이번",
    "뉴스", "소식", "업데이트", "발표", "출시", "어떤 곳", "에 대해",
}
_ENTITY_PATTERNS = {
    "에 대해", "에 대한", "란 뭐야", "이 뭐야", "가 뭐야",
    "를 소개", "을 소개", "검색해줘", "검색해 줘", "where is", "what is",
}
_WEATHER_KEYWORDS = {"날씨", "기온", "강수", "미세먼지", "폭염", "태풍", "weather", "forecast"}
_EMAIL_KEYWORDS   = {"이메일", "메일", "email", "mail", "초안", "draft", "발송"}
_SUMMARY_KEYWORDS = {"요약", "정리해", "summarize", "summary", "핵심만", "간단히"}
_IMAGE_KEYWORDS   = {"이미지", "사진", "그림", "photo", "image", "picture", "screenshot"}


def _fallback_agent(query: str) -> str:
    """LLM 계획 수립 실패 시 키워드 기반 결정론적 에이전트 선택."""
    t = query.lower()
    if any(k in t for k in _IMAGE_KEYWORDS):   return "image_agent"
    if any(k in t for k in _EMAIL_KEYWORDS):   return "email_agent"
    if any(k in t for k in _SUMMARY_KEYWORDS): return "summary_agent"
    if any(k in t for k in _WEATHER_KEYWORDS): return "weather_agent"
    if any(pat in t for pat in _ENTITY_PATTERNS): return "web_search_agent"
    if any(k in t for k in _WEB_KEYWORDS):     return "web_search_agent"
    if any(k in t for k in _DOC_KEYWORDS):     return "rag_agent"
    return "reasoning_agent"


# ── LLM 모델 ────────────────────────────────────────────────────────────────
def _get_model() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_ctx=settings.num_ctx,
        num_predict=256,
    )


def _last_user_message(state: ChatState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else str(c)
    return ""


def _last_ai_result(state: ChatState) -> str | None:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            c = msg.content
            return c if isinstance(c, str) else str(c)
    return None


def _doc_context(session_id: str | None) -> str:
    """로컬 지식베이스에 문서가 있으면 프롬프트에 힌트를 추가한다."""
    try:
        docs = _doc_repo.list_active()
        visible = [
            d for d in docs
            if d.scope == "GLOBAL" or (d.scope == "SESSION" and str(d.session_id) == str(session_id))
        ]
        if not visible:
            return ""
        names = "\n".join(f"  - {d.file_name}" for d in visible[:10])
        return f"\n\n[로컬 문서 목록]\n{names}\n위 문서와 관련된 질문이면 rag_agent를 우선 선택하세요."
    except Exception:
        return ""


# ── 메인 노드 ────────────────────────────────────────────────────────────────

def orchestrator_agent_node(state: ChatState) -> dict:
    """
    Phase 1 (plan == None):
        LLM으로 의도 파악 + [PLAN] 수립 → 첫 작업 에이전트로 라우팅 준비
    Phase 2+ (plan exists):
        마지막 결과 결정론적 평가 → 계획 업데이트 → 다음 에이전트 준비
    """
    plan: list[OrchestratorTask] | None = state.get("orchestrator_plan")
    idx:  int                            = state.get("orchestrator_task_idx") or 0

    # ── Phase 1: 계획 수립 ───────────────────────────────────────────────────
    if plan is None:
        user_query  = _last_user_message(state)
        session_id  = state.get("session_id")
        doc_ctx     = _doc_context(session_id)
        system_text = ORCHESTRATOR_PLAN_PROMPT + doc_ctx + today_context()

        plan = []
        try:
            response = _get_model().invoke(
                [SystemMessage(content=system_text), HumanMessage(content=user_query)]
            )
            plan = _parse_plan(response.content)
        except Exception as e:
            _log.warning("Orchestrator plan creation failed: %s", e)

        if not plan:
            # 폴백: 키워드 기반 단일 작업
            _log.info("Orchestrator: using keyword fallback for %r", user_query)
            plan = [{
                "id": 1,
                "agent": _fallback_agent(user_query),
                "description": user_query,
                "status": "pending",
                "result": None,
            }]

        _log.info("Orchestrator plan: %s", [(t["id"], t["agent"]) for t in plan])
        return {"orchestrator_plan": plan, "orchestrator_task_idx": 0}

    # ── Phase 2+: 결과 평가 + 계획 업데이트 ─────────────────────────────────
    plan = [dict(t) for t in plan]   # 불변 → 가변 복사
    extra_state: dict = {}

    # 완료된 작업 상태 업데이트
    if idx < len(plan):
        result = _last_ai_result(state)
        task_status = "done" if not _is_failed(result) else "failed"
        plan[idx]["status"] = task_status
        plan[idx]["result"] = (result or "")[:300]

        if task_status == "done":
            # ── 성공한 작업에만 LLM follow-up 판단 ──────────────────────────
            # 실패한 경우는 자동 폴백(web_search)이 이미 처리하므로 Y/N 버튼 불필요
            user_q   = _last_user_message(state)
            followup = _llm_derive_followup(plan[idx]["agent"], user_q, result or "")
            if followup:
                extra_state["pending_followup"] = followup
                _log.info("Orchestrator: LLM follow-up → %r", followup[:60])

        else:
            # ── 실패한 작업: 자동 폴백 삽입 + 상태 알림 (Y/N 버튼 없음) ───────
            # RAG 실패 → "관련 정보가 없어서 웹 검색합니다" 로 자동 진행
            if (
                plan[idx]["agent"] == "rag_agent"
                and not any(t["agent"] == "web_search_agent" and t["status"] == "pending" for t in plan)
            ):
                fallback_task: OrchestratorTask = {
                    "id": max(t["id"] for t in plan) + 1,
                    "agent": "web_search_agent",
                    "description": plan[idx]["description"],
                    "status": "pending",
                    "result": None,
                }
                plan.insert(idx + 1, fallback_task)
                _log.info("Orchestrator: RAG 실패 → web_search_agent 자동 폴백 삽입")

    next_idx = idx + 1
    _log.info("Orchestrator: task %d done → advancing to %d/%d", idx, next_idx, len(plan))
    return {"orchestrator_plan": plan, "orchestrator_task_idx": next_idx, **extra_state}


# ── 라우팅 함수 (graph.py에서 사용) ─────────────────────────────────────────

def route_from_orchestrator(state: ChatState) -> str:
    """
    orchestrator_agent 완료 후 다음 노드를 결정한다.
    - 계획의 현재 작업 에이전트로 라우팅
    - 모든 작업 완료 → END
    """
    from langgraph.graph import END

    plan: list[OrchestratorTask] | None = state.get("orchestrator_plan")
    idx:  int                            = state.get("orchestrator_task_idx") or 0

    if not plan:
        return END

    # pending 상태인 다음 작업 찾기
    for task in plan[idx:]:
        if task.get("status") == "pending":
            agent = task["agent"]
            _log.info("Orchestrator routing → %s (%s)", agent, task["description"][:50])
            return agent

    _log.info("Orchestrator: all tasks done → END")
    return END
