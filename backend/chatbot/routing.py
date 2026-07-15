from __future__ import annotations

from backend.chatbot.state import ChatState


def select_mode(state: ChatState) -> str:
    """항상 retrieve를 먼저 시도한다 (자동 모드)."""
    return "retrieve"


def select_after_retrieve(state: ChatState) -> str:
    """
    retrieve 이후 청크가 존재하면 RAG, 없으면 일반 대화로 분기한다.
    문서가 없거나 관련 내용이 없는 질문은 LLM이 직접 답변한다.
    """
    chunks = state.get("retrieved_chunks") or []
    if chunks:
        return "generate_rag"
    return "chat"
