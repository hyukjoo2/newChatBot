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
