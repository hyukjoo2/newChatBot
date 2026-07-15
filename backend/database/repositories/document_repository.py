from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from psycopg.rows import dict_row

from backend.database.connection import get_connection


@dataclass
class Document:
    id: str
    uploaded_by: str
    file_name: str
    original_file_name: str
    original_path: str
    extracted_text_path: Optional[str]
    mime_type: Optional[str]
    file_extension: Optional[str]
    file_size_bytes: Optional[int]
    file_hash_sha256: Optional[str]
    category: Optional[str]
    scope: str
    session_id: Optional[str]
    status: str
    is_active: bool
    chunk_count: int
    page_count: Optional[int]
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    embedding_model: Optional[str]
    embedding_dimension: Optional[int]
    error_message: Optional[str]
    metadata: dict
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]
    deleted_at: Optional[datetime]


class DocumentRepository:

    def create(
        self,
        uploaded_by: str,
        file_name: str,
        original_file_name: str,
        original_path: str,
        mime_type: Optional[str] = None,
        file_extension: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        file_hash_sha256: Optional[str] = None,
        category: Optional[str] = None,
        scope: str = "GLOBAL",
        session_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        sql = """
            INSERT INTO documents (
                uploaded_by, file_name, original_file_name, original_path,
                mime_type, file_extension, file_size_bytes, file_hash_sha256,
                category, scope, session_id, status,
                chunk_size, chunk_overlap, metadata
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, 'UPLOADED',
                %s, %s, %s
            )
            RETURNING id
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql,
                    (
                        uploaded_by, file_name, original_file_name, original_path,
                        mime_type, file_extension, file_size_bytes, file_hash_sha256,
                        category, scope, session_id,
                        chunk_size, chunk_overlap,
                        json.dumps(metadata or {}),
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])

    def list_all(self) -> list[Document]:
        sql = """
            SELECT * FROM documents
            WHERE deleted_at IS NULL
            ORDER BY created_at DESC
        """
        return self._fetch_many(sql)

    def list_active(self) -> list[Document]:
        sql = """
            SELECT * FROM documents
            WHERE is_active = TRUE
              AND deleted_at IS NULL
              AND status = 'READY'
            ORDER BY updated_at DESC
        """
        return self._fetch_many(sql)

    def get_by_id(self, document_id: str) -> Optional[Document]:
        sql = "SELECT * FROM documents WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (document_id,))
                r = cur.fetchone()
                return self._row_to_doc(r) if r else None

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """SHA-256 해시로 문서를 조회한다 (삭제되지 않은 것만)."""
        sql = "SELECT * FROM documents WHERE file_hash_sha256 = %s AND deleted_at IS NULL"
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (file_hash,))
                r = cur.fetchone()
                return self._row_to_doc(r) if r else None

    def update_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        sql = """
            UPDATE documents
            SET status = %s, error_message = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, error_message, document_id))
                conn.commit()

    def update_extracted_path(self, document_id: str, path: str) -> None:
        sql = """
            UPDATE documents
            SET extracted_text_path = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (path, document_id))
                conn.commit()

    def mark_ready(
        self,
        document_id: str,
        chunk_count: int,
        page_count: Optional[int],
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        sql = """
            UPDATE documents
            SET status = 'READY',
                is_active = TRUE,
                chunk_count = %s,
                page_count = %s,
                embedding_model = %s,
                embedding_dimension = %s,
                processed_at = CURRENT_TIMESTAMP,
                error_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (chunk_count, page_count, embedding_model, embedding_dimension, document_id),
                )
                conn.commit()

    def mark_failed(self, document_id: str, error_message: str) -> None:
        sql = """
            UPDATE documents
            SET status = 'FAILED',
                is_active = FALSE,
                error_message = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (error_message, document_id))
                conn.commit()

    def deactivate(self, document_id: str) -> None:
        sql = """
            UPDATE documents
            SET is_active = FALSE,
                status = 'INACTIVE',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND deleted_at IS NULL
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                conn.commit()

    def reactivate(self, document_id: str) -> None:
        sql = """
            UPDATE documents
            SET is_active = TRUE,
                status = 'READY',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND deleted_at IS NULL AND chunk_count > 0
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                conn.commit()

    def soft_delete(self, document_id: str) -> None:
        sql = """
            UPDATE documents
            SET is_active = FALSE,
                status = 'DELETED',
                deleted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND deleted_at IS NULL
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                conn.commit()

    def list_by_session(self, session_id: str) -> list[Document]:
        """session_id 가 일치하는 비삭제 문서 목록을 반환한다."""
        sql = "SELECT * FROM documents WHERE session_id = %s AND deleted_at IS NULL"
        return self._fetch_many(sql, (session_id,))

    def hard_delete(self, document_id: str) -> None:
        sql = "DELETE FROM documents WHERE id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                conn.commit()

    def list_file_names(self) -> set[str]:
        """삭제되지 않은 문서의 file_name 집합을 반환한다 (중복 이름 방지용)."""
        sql = "SELECT file_name FROM documents WHERE deleted_at IS NULL"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return {r[0] for r in cur.fetchall()}

    def _fetch_many(self, sql: str, params: tuple = ()) -> list[Document]:
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [self._row_to_doc(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_doc(r: dict) -> Document:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return Document(
            id=str(r["id"]),
            uploaded_by=str(r["uploaded_by"]),
            file_name=r["file_name"],
            original_file_name=r["original_file_name"],
            original_path=r["original_path"],
            extracted_text_path=r.get("extracted_text_path"),
            mime_type=r.get("mime_type"),
            file_extension=r.get("file_extension"),
            file_size_bytes=r.get("file_size_bytes"),
            file_hash_sha256=r.get("file_hash_sha256"),
            category=r.get("category"),
            scope=r["scope"],
            session_id=str(r["session_id"]) if r.get("session_id") else None,
            status=r["status"],
            is_active=r["is_active"],
            chunk_count=r["chunk_count"],
            page_count=r.get("page_count"),
            chunk_size=r.get("chunk_size"),
            chunk_overlap=r.get("chunk_overlap"),
            embedding_model=r.get("embedding_model"),
            embedding_dimension=r.get("embedding_dimension"),
            error_message=r.get("error_message"),
            metadata=meta,
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            processed_at=r.get("processed_at"),
            deleted_at=r.get("deleted_at"),
        )
