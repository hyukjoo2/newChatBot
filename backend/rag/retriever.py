from __future__ import annotations

from typing import Optional

from backend.config import settings
from backend.database.repositories.chunk_repository import ChunkRepository, SearchResult
from backend.rag.embeddings import get_embeddings


_chunk_repo = ChunkRepository()


def vector_search(
    query: str,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
    category: Optional[str] = None,
) -> list[SearchResult]:
    """벡터 유사도 검색."""
    embedding = get_embeddings(query)
    return _chunk_repo.vector_search(
        query_embedding=embedding,
        session_id=session_id,
        top_k=top_k or settings.default_retrieval_top_k,
        category=category,
    )


def keyword_search(
    query: str,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """PostgreSQL 전문 검색."""
    return _chunk_repo.keyword_search(
        query_text=query,
        session_id=session_id,
        top_k=top_k or settings.default_retrieval_top_k,
    )


def hybrid_search(
    query: str,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
    candidate_limit: int = 20,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """하이브리드 검색 (벡터 + 키워드, RRF 결합)."""
    embedding = get_embeddings(query)
    return _chunk_repo.hybrid_search(
        query_embedding=embedding,
        query_text=query,
        session_id=session_id,
        top_k=top_k or settings.default_retrieval_top_k,
        candidate_limit=candidate_limit,
        rrf_k=rrf_k,
    )
