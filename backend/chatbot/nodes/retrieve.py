from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from backend.chatbot.state import ChatState
from backend.config import settings
from backend.database.repositories.chunk_repository import ChunkRepository
from backend.rag.embeddings import get_embeddings


_chunk_repo = ChunkRepository()

# 검색/설명 요청 한국어 메타 표현 제거 패턴
_META_RE = re.compile(
    r"(에\s*대해서?|에\s*대한|와\s*관련|관련해서?|관해서?)?\s*"
    r"(rag에서|문서에서|지식베이스에서|로컬에서|db에서|자료에서)?\s*"
    r"(검색해\s*(줘|주세요|줄래요?)?|"
    r"찾아\s*(줘|주세요|줄래요?)?|"
    r"알려\s*(줘|주세요|줄래요?)?|"
    r"설명해\s*(줘|주세요|줄래요?)?|"
    r"정리해\s*(줘|주세요|줄래요?)?|"
    r"요약해\s*(줘|주세요|줄래요?)?|"
    r"뭐야\??|뭔가요\??|무엇인가요\??)$",
    re.IGNORECASE,
)

# rag에서 검색해줘 처럼 검색어 없이 메타 명령만 있는 경우 앞부분도 제거
_META_PREFIX_RE = re.compile(
    r"^(rag에서|문서에서|지식베이스에서|로컬에서|db에서|자료에서)\s*",
    re.IGNORECASE,
)

# 최소 유사도 — 이 값 미만이면 관련 문서 없음으로 간주
_MIN_SCORE = 0.02


def _extract_search_query(text: str) -> str:
    """사용자 메시지에서 한국어 메타 명령어를 제거하고 핵심 검색어만 반환한다."""
    cleaned = _META_RE.sub("", text.strip()).strip(" ,.")
    cleaned = _META_PREFIX_RE.sub("", cleaned).strip(" ,.")
    return cleaned if len(cleaned) >= 2 else text


def retrieve_node(state: ChatState) -> dict:
    """RAG 검색 노드: 사용자 질문을 임베딩하여 관련 청크를 조회한다."""
    messages = state["messages"]
    session_id = state.get("session_id")

    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                user_query = content
            break

    if not user_query:
        return {"retrieved_chunks": []}

    try:
        search_query = _extract_search_query(user_query)
        embedding = get_embeddings(search_query)
        results = _chunk_repo.hybrid_search(
            query_embedding=embedding,
            query_text=search_query,
            session_id=session_id,
            top_k=settings.default_retrieval_top_k,
        )
        # 최소 점수 미만 결과 제거
        results = [r for r in results if r.score >= _MIN_SCORE]

        chunks = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "file_name": r.file_name,
                "category": r.category,
                "chunk_index": r.chunk_index,
                "page_number": r.page_number,
                "content": r.content,
                "score": r.score,
            }
            for r in results
        ]
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("retrieve_node error: %s", e, exc_info=True)
        chunks = []

    return {"retrieved_chunks": chunks}
