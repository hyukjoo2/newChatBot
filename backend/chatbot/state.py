from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TaskItem(TypedDict):
    """task_agent가 관리하는 개별 작업 단위."""
    id: int
    description: str
    status: str          # "pending" | "in_progress" | "done" | "failed"
    result: Optional[str]


class OrchestratorTask(TypedDict):
    """orchestrator_agent가 관리하는 에이전트 실행 단위."""
    id: int
    agent: str           # 실행할 에이전트 이름
    description: str     # 작업 설명 / 검색 쿼리
    status: str          # "pending" | "done" | "failed"
    result: Optional[str]


class ChatState(TypedDict):
    """LangGraph 상태 정의."""

    # 대화 메시지 히스토리
    messages: Annotated[list[BaseMessage], add_messages]

    # 대화 모드: CHAT | RAG
    mode: str

    # 현재 세션 ID (PostgreSQL chat_sessions.id)
    session_id: str

    # RAG 검색 결과 (retrieve 노드 → generate 노드로 전달)
    retrieved_chunks: Optional[list[dict]]

    # supervisor 가 결정한 다음 에이전트
    next: Optional[str]

    # task_agent가 관리하는 TODO 목록 (None이면 일반 대화)
    task_list: Optional[list[TaskItem]]

    # Phase 1 완료 → Phase 2 시작 신호 (문자열 매칭 대신 상태 필드로 안전하게 전환)
    task_plan_ready: Optional[bool]

    # ── Corrective RAG ──────────────────────────────────────────────────────
    # 답변 품질 평가 결과: "relevant" | "not_relevant"
    answer_grade: Optional[str]

    # RAG 재시도 횟수 (최대 2회)
    rag_retry_count: Optional[int]

    # reasoning_agent가 제안하는 후속 작업 (UI Yes/No 버튼에 사용)
    # 형식: "사용자에게 보여줄 질문:::실제 실행할 작업 설명"
    pending_followup: Optional[str]

    # ── Orchestrator ────────────────────────────────────────────────────────
    # orchestrator_agent가 수립·관리하는 실행 계획
    orchestrator_plan: Optional[list[OrchestratorTask]]

    # 현재 실행 중인 작업 인덱스
    orchestrator_task_idx: Optional[int]
