from __future__ import annotations

import logging
from typing import Optional

from backend.database.repositories.session_repository import (
    ChatSession,
    SessionRepository,
)
from backend.database.repositories.document_repository import DocumentRepository
from backend.database.repositories.chunk_repository import ChunkRepository

_log = logging.getLogger(__name__)
_repo = SessionRepository()
_doc_repo = DocumentRepository()
_chunk_repo = ChunkRepository()


def get_or_create_admin_user_id() -> str:
    return _repo.ensure_admin_user()


def create_session(
    user_id: str,
    title: str = "새 대화",
    mode: str = "CHAT",
) -> str:
    """새 대화 세션을 생성하고 세션 ID를 반환한다."""
    return _repo.create(user_id=user_id, title=title, default_mode=mode)


def list_sessions(user_id: str) -> list[ChatSession]:
    return _repo.list_sessions(user_id)


def get_session(session_id: str) -> Optional[ChatSession]:
    return _repo.get_by_id(session_id)


def rename_session(session_id: str, title: str) -> None:
    _repo.update_title(session_id, title)


def delete_session(session_id: str) -> None:
    """세션을 삭제하고 SESSION-scoped 문서·벡터도 함께 정리한다."""
    # SESSION-scoped 문서 cascade 삭제
    try:
        session_docs = _doc_repo.list_by_session(session_id)
        for doc in session_docs:
            try:
                _chunk_repo.delete_by_document(doc.id)
                _doc_repo.soft_delete(doc.id)
            except Exception as e:
                _log.warning("Failed to delete document %s for session %s: %s", doc.id, session_id, e)
    except Exception as e:
        _log.warning("Failed to list session documents for %s: %s", session_id, e)

    _repo.soft_delete(session_id)


def update_session_last_message(session_id: str) -> None:
    _repo.update_last_message_at(session_id)
