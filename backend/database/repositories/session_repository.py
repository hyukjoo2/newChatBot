from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from backend.database.connection import get_connection


@dataclass
class ChatSession:
    id: str
    user_id: str
    title: str
    default_mode: str
    summary: Optional[str]
    is_pinned: bool
    is_archived: bool
    last_message_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionRepository:

    def create(
        self,
        user_id: str,
        title: str = "새 대화",
        default_mode: str = "CHAT",
    ) -> str:
        sql = """
            INSERT INTO chat_sessions (
                user_id, title, default_mode, last_message_at
            )
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            RETURNING id
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (user_id, title, default_mode))
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])

    def list_sessions(self, user_id: str) -> list[ChatSession]:
        sql = """
            SELECT
                s.id,
                s.user_id,
                s.title,
                s.default_mode,
                s.summary,
                s.is_pinned,
                s.is_archived,
                s.last_message_at,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.user_id = %s
              AND s.deleted_at IS NULL
              AND s.is_archived = FALSE
            GROUP BY s.id
            ORDER BY
                s.is_pinned DESC,
                COALESCE(s.last_message_at, s.created_at) DESC
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (user_id,))
                rows = cur.fetchall()
                return [
                    ChatSession(
                        id=str(r["id"]),
                        user_id=str(r["user_id"]),
                        title=r["title"],
                        default_mode=r["default_mode"],
                        summary=r["summary"],
                        is_pinned=r["is_pinned"],
                        is_archived=r["is_archived"],
                        last_message_at=r["last_message_at"],
                        created_at=r["created_at"],
                        updated_at=r["updated_at"],
                        message_count=r["message_count"],
                    )
                    for r in rows
                ]

    def get_by_id(self, session_id: str) -> Optional[ChatSession]:
        sql = """
            SELECT
                s.id, s.user_id, s.title, s.default_mode, s.summary,
                s.is_pinned, s.is_archived, s.last_message_at,
                s.created_at, s.updated_at,
                COUNT(m.id) AS message_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.id = %s AND s.deleted_at IS NULL
            GROUP BY s.id
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (session_id,))
                r = cur.fetchone()
                if r is None:
                    return None
                return ChatSession(
                    id=str(r["id"]),
                    user_id=str(r["user_id"]),
                    title=r["title"],
                    default_mode=r["default_mode"],
                    summary=r["summary"],
                    is_pinned=r["is_pinned"],
                    is_archived=r["is_archived"],
                    last_message_at=r["last_message_at"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    message_count=r["message_count"],
                )

    def update_title(self, session_id: str, title: str) -> None:
        sql = """
            UPDATE chat_sessions
            SET title = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND deleted_at IS NULL
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (title, session_id))
                conn.commit()

    def update_last_message_at(self, session_id: str) -> None:
        sql = """
            UPDATE chat_sessions
            SET last_message_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id,))
                conn.commit()

    def update_summary(self, session_id: str, summary: str) -> None:
        sql = """
            UPDATE chat_sessions
            SET summary = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (summary, session_id))
                conn.commit()

    def soft_delete(self, session_id: str) -> None:
        """세션을 소프트 삭제한다. SESSION 범위 문서는 GLOBAL로 승격해 검색 가능하게 유지."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # SESSION-scoped 문서를 GLOBAL로 승격 (파일/벡터 보존)
                cur.execute(
                    """
                    UPDATE documents
                    SET scope = 'GLOBAL',
                        session_id = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                      AND scope = 'SESSION'
                      AND deleted_at IS NULL
                    """,
                    (session_id,),
                )
                # 세션 소프트 삭제
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET deleted_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND deleted_at IS NULL
                    """,
                    (session_id,),
                )
                conn.commit()

    def ensure_admin_user(self) -> str:
        """관리자 사용자 ID를 반환한다. 없으면 삽입 후 반환."""
        sql_select = "SELECT id FROM app_users WHERE username = 'admin'"
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql_select)
                row = cur.fetchone()
                if row:
                    return str(row["id"])
                cur.execute(
                    """
                    INSERT INTO app_users (username, display_name, role)
                    VALUES ('admin', 'Administrator', 'ADMIN')
                    ON CONFLICT (username) DO NOTHING
                    RETURNING id
                    """
                )
                row = cur.fetchone()
                conn.commit()
                if row:
                    return str(row["id"])
                # ON CONFLICT path
                cur.execute(sql_select)
                row = cur.fetchone()
                return str(row["id"])
