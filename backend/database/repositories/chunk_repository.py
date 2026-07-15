from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from psycopg.rows import dict_row

from backend.database.connection import get_connection


@dataclass
class DocumentChunk:
    id: str
    document_id: str
    chunk_index: int
    page_number: Optional[int]
    page_chunk_index: Optional[int]
    start_offset: Optional[int]
    end_offset: Optional[int]
    content: str
    token_count: Optional[int]
    metadata: dict
    created_at: datetime


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    file_name: str
    category: Optional[str]
    chunk_index: int
    page_number: Optional[int]
    content: str
    metadata: dict
    score: float
    vector_distance: Optional[float] = None  # 코사인 거리 (0=동일, 1=무관)


class ChunkRepository:

    def save(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        embedding: Optional[list[float]] = None,
        page_number: Optional[int] = None,
        page_chunk_index: Optional[int] = None,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        token_count: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        sql = """
            INSERT INTO document_chunks (
                document_id, chunk_index, page_number, page_chunk_index,
                start_offset, end_offset, content, token_count,
                embedding, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
            RETURNING id
        """
        emb_str = None
        if embedding is not None:
            emb_str = "[" + ",".join(str(v) for v in embedding) + "]"

        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql,
                    (
                        document_id, chunk_index, page_number, page_chunk_index,
                        start_offset, end_offset, content, token_count,
                        emb_str, json.dumps(metadata or {}),
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row["id"])

    def save_batch(self, chunks: list[dict]) -> None:
        """여러 청크를 한 번에 저장한다."""
        sql = """
            INSERT INTO document_chunks (
                document_id, chunk_index, page_number, page_chunk_index,
                start_offset, end_offset, content, token_count,
                embedding, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
        """
        rows = []
        for c in chunks:
            emb = c.get("embedding")
            emb_str = None
            if emb is not None:
                emb_str = "[" + ",".join(str(v) for v in emb) + "]"
            rows.append((
                c["document_id"], c["chunk_index"], c.get("page_number"),
                c.get("page_chunk_index"), c.get("start_offset"), c.get("end_offset"),
                c["content"], c.get("token_count"),
                emb_str, json.dumps(c.get("metadata") or {}),
            ))

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
                conn.commit()

    def delete_by_document(self, document_id: str) -> None:
        sql = "DELETE FROM document_chunks WHERE document_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                conn.commit()

    def get_summary_chunk(self, document_id: str) -> Optional[str]:
        """chunk_index=-1 요약 쫐크의 콘텐츠를 반환한다."""
        sql = """
            SELECT content FROM document_chunks
            WHERE document_id = %s AND chunk_index = -1
            LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
                return row[0] if row else None

    def list_by_document(
        self,
        document_id: str,
        page_size: int = 20,
        offset: int = 0,
    ) -> list[DocumentChunk]:
        sql = """
            SELECT
                id, document_id, chunk_index, page_number, page_chunk_index,
                start_offset, end_offset, content, token_count, metadata, created_at
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index ASC
            LIMIT %s OFFSET %s
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, (document_id, page_size, offset))
                return [self._row_to_chunk(r) for r in cur.fetchall()]

    def vector_search(
        self,
        query_embedding: list[float],
        session_id: Optional[str],
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> list[SearchResult]:
        emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        params: list = [emb_str, emb_str]
        scope_clause = ""

        if session_id:
            scope_clause = """
                AND (
                    d.scope = 'GLOBAL'
                    OR (d.scope = 'SESSION' AND d.session_id = %s)
                )
            """
            params.append(session_id)
        else:
            scope_clause = "AND d.scope = 'GLOBAL'"

        category_clause = ""
        if category:
            category_clause = "AND d.category = %s"
            params.append(category)

        params.append(top_k)

        sql = f"""
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.file_name,
                d.category,
                dc.chunk_index,
                dc.page_number,
                dc.content,
                dc.metadata,
                1 - (dc.embedding <=> %s::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.embedding IS NOT NULL
              AND d.is_active = TRUE
              AND d.deleted_at IS NULL
              AND d.status = 'READY'
              {scope_clause}
              {category_clause}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [self._row_to_search(r, "similarity") for r in cur.fetchall()]

    def keyword_search(
        self,
        query_text: str,
        session_id: Optional[str],
        top_k: int = 5,
    ) -> list[SearchResult]:
        params: list = [query_text, query_text]
        scope_clause = ""

        if session_id:
            scope_clause = """
                AND (
                    d.scope = 'GLOBAL'
                    OR (d.scope = 'SESSION' AND d.session_id = %s)
                )
            """
            params.append(session_id)
        else:
            scope_clause = "AND d.scope = 'GLOBAL'"

        params.append(top_k)

        sql = f"""
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.file_name,
                d.category,
                dc.chunk_index,
                dc.page_number,
                dc.content,
                dc.metadata,
                ts_rank_cd(
                    dc.search_vector,
                    plainto_tsquery('simple', %s)
                ) AS keyword_score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.search_vector @@ plainto_tsquery('simple', %s)
              AND d.is_active = TRUE
              AND d.deleted_at IS NULL
              AND d.status = 'READY'
              {scope_clause}
            ORDER BY keyword_score DESC
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [self._row_to_search(r, "keyword_score") for r in cur.fetchall()]

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        session_id: Optional[str],
        top_k: int = 5,
        candidate_limit: int = 20,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        use_keyword = bool(query_text and query_text.strip())

        scope_clause = ""
        scope_params: list = []
        if session_id:
            scope_clause = """
                AND (
                    d.scope = 'GLOBAL'
                    OR (d.scope = 'SESSION' AND d.session_id = %(session_id)s)
                )
            """
            scope_params = [session_id]
        else:
            scope_clause = "AND d.scope = 'GLOBAL'"

        if use_keyword:
            keyword_cte = f"""
            keyword_results AS (
                SELECT
                    dc.id,
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(
                            dc.search_vector,
                            plainto_tsquery('simple', %(query_text)s)
                        ) DESC
                    ) AS keyword_rank
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.search_vector @@ plainto_tsquery('simple', %(query_text)s)
                  AND dc.chunk_index >= 0
                  AND d.is_active = TRUE
                  AND d.deleted_at IS NULL
                  AND d.status = 'READY'
                  {scope_clause}
                ORDER BY ts_rank_cd(
                    dc.search_vector,
                    plainto_tsquery('simple', %(query_text)s)
                ) DESC
                LIMIT %(candidate_limit)s
            ),"""
            keyword_join = "LEFT JOIN keyword_results kr ON kr.id = dc.id"
            keyword_score = "COALESCE(1.0 / (%(rrf_k)s + kr.keyword_rank), 0)"
            where_clause = "WHERE vr.id IS NOT NULL OR kr.id IS NOT NULL"
        else:
            # 빈 쿼리: 순수 벡터 검색만 사용
            keyword_cte = ""
            keyword_join = ""
            keyword_score = "0"
            where_clause = "WHERE vr.id IS NOT NULL"

        sql = f"""
            WITH vector_results AS (
                SELECT
                    dc.id,
                    ROW_NUMBER() OVER (
                        ORDER BY dc.embedding <=> %(emb)s::vector
                    ) AS vector_rank
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.embedding IS NOT NULL
                  AND dc.chunk_index >= 0
                  AND d.is_active = TRUE
                  AND d.deleted_at IS NULL
                  AND d.status = 'READY'
                  {scope_clause}
                ORDER BY dc.embedding <=> %(emb)s::vector
                LIMIT %(candidate_limit)s
            ),
            {keyword_cte}
            _dummy AS (SELECT 1)
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.file_name,
                d.category,
                dc.page_number,
                dc.chunk_index,
                dc.content,
                dc.metadata,
                COALESCE(1.0 / (%(rrf_k)s + vr.vector_rank), 0)
                + {keyword_score} AS combined_score,
                (dc.embedding <=> %(emb)s::vector) AS vector_distance
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            LEFT JOIN vector_results vr ON vr.id = dc.id
            {keyword_join}
            {where_clause}
            ORDER BY combined_score DESC
            LIMIT %(top_k)s
        """
        bind: dict = {
            "emb": emb_str,
            "query_text": query_text,
            "candidate_limit": candidate_limit,
            "rrf_k": rrf_k,
            "top_k": top_k,
        }
        if session_id:
            bind["session_id"] = session_id

        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, bind)
                return [self._row_to_search(r, "combined_score") for r in cur.fetchall()]

    def search_by_filename(
        self,
        query: str,
        query_embedding: list[float],
        session_id: Optional[str],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        파일명에 query 가 포함된 문서의 청크를 밴터 유사도 순으로 반환한다.
        콘텐츠에 특정 단어가 없어도 파일명으로 문서를 특정할 수 있는 fallback 경로.
        """
        emb_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
        scope_clause = ""
        scope_params: list = []
        if session_id:
            scope_clause = """
                AND (
                    d.scope = 'GLOBAL'
                    OR (d.scope = 'SESSION' AND d.session_id = %s)
                )
            """
            scope_params = [session_id]
        else:
            scope_clause = "AND d.scope = 'GLOBAL'"

        sql = f"""
            SELECT
                dc.id AS chunk_id,
                dc.document_id,
                d.file_name,
                d.category,
                dc.chunk_index,
                dc.page_number,
                dc.content,
                dc.metadata,
                1 - (dc.embedding <=> %s::vector) AS score
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE LOWER(d.file_name) LIKE LOWER(%s)
              AND dc.embedding IS NOT NULL
              AND d.is_active = TRUE
              AND d.deleted_at IS NULL
              AND d.status = 'READY'
              {scope_clause}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """
        like_pat = f"%{query}%"
        params = [emb_str, like_pat, *scope_params, emb_str, top_k]

        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                return [self._row_to_search(r, "score") for r in cur.fetchall()]

    @staticmethod
    def _row_to_chunk(r: dict) -> DocumentChunk:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        return DocumentChunk(
            id=str(r["id"]),
            document_id=str(r["document_id"]),
            chunk_index=r["chunk_index"],
            page_number=r.get("page_number"),
            page_chunk_index=r.get("page_chunk_index"),
            start_offset=r.get("start_offset"),
            end_offset=r.get("end_offset"),
            content=r["content"],
            token_count=r.get("token_count"),
            metadata=meta,
            created_at=r["created_at"],
        )

    @staticmethod
    def _row_to_search(r: dict, score_col: str) -> SearchResult:
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        vd = r.get("vector_distance")
        return SearchResult(
            chunk_id=str(r["chunk_id"]),
            document_id=str(r["document_id"]),
            file_name=r["file_name"],
            category=r.get("category"),
            chunk_index=r["chunk_index"],
            page_number=r.get("page_number"),
            content=r["content"],
            metadata=meta,
            score=float(r[score_col] or 0),
            vector_distance=float(vd) if vd is not None else None,
        )
