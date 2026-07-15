from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from psycopg.rows import dict_row

from backend.database.connection import get_connection


@dataclass
class ChatMessage:
    id: str
    session_id: str
    role: str
    content: str
    mode: Optional[str]
    status: str
    model_name: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    metadata: dict
    created_at: datetime


class MessageRepository:

    def save(
        self,
        session_id: str,
        role: str,
        content: str,
        mode: Optional[str] = None,
        status: str = "COMPLETE",
        model_name: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        sql = """
            INSERT INTO chat_messages (
                session_id, role, content, mode, status,
                model_name, prompt_tokens, completion_tokens, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        meta_json = json.dumps(metadata or {})
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql,
                    (
                        session_id,
                        role,
                        content,
                        mode,
                        status,
                        model_name,
                        prompt_tokens,
                        completion_tokens,
                        meta_json,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])

    def list_by_session(self, session_id: str) -> list[ChatMessage]:
        sql = """
            SELECT
                id, session_id, role, content, mode, status,
                model_name, prompt_tokens, completion_tokens,
                metadata, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (session_id,))
                rows = cur.fetchall()
                return [self._row_to_msg(r) for r in rows]

    def list_recent(self, session_id: str, limit: int = 10) -> list[ChatMessage]:
        sql = """
            SELECT
                id, session_id, role, content, mode, status,
                model_name, prompt_tokens, completion_tokens,
                metadata, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (session_id, limit))
                rows = cur.fetchall()
                return list(reversed([self._row_to_msg(r) for r in rows]))

    @staticmethod
    def _row_to_msg(r: dict) -> ChatMessage:
        meta = r["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return ChatMessage(
            id=str(r["id"]),
            session_id=str(r["session_id"]),
            role=r["role"],
            content=r["content"],
            mode=r["mode"],
            status=r["status"],
            model_name=r["model_name"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            metadata=meta or {},
            created_at=r["created_at"],
        )
