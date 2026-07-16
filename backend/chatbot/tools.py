"""
LangGraph tool: 로컬 지식베이스 문서 검색.
LLM이 search_documents 를 직접 호출해 언제 검색할지 판단한다.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from backend.chatbot.state import ChatState
from backend.config import settings
from backend.database.repositories.chunk_repository import ChunkRepository, SearchResult
from backend.rag.embeddings import get_embeddings

_chunk_repo = ChunkRepository()
_MIN_SCORE = 0.013        # RRF 최소 점수 (rank ~8위 이내)
_MAX_VECTOR_DISTANCE = 0.50  # 절대 상한 (코사인 거리) — 0.50 = 유사도 50% 이상만 허용
_LOW_RELEVANCE_THRESHOLD = 0.38  # 이 이상이면 ⚠️ 저관련성 경고 첨부
_ADAPTIVE_WINDOW = 0.12   # 베스트 매치 기준 상대 거리 폭

_log = logging.getLogger(__name__)
_HAS_KOREAN = re.compile(r"[\uac00-\ud7a3]")


def _hybrid_and_filter(query: str, session_id, top_k: int) -> list[SearchResult]:
    """단일 쿼리 hybrid 검색 + 3단계 필터링."""
    emb = get_embeddings(query)
    raw = _chunk_repo.hybrid_search(
        query_embedding=emb,
        query_text=query,
        session_id=session_id,
        top_k=top_k,
    )
    r = [x for x in raw if x.score >= _MIN_SCORE]
    r = [x for x in r if x.vector_distance is None or x.vector_distance <= _MAX_VECTOR_DISTANCE]
    if r:
        min_vd = min((x.vector_distance for x in r if x.vector_distance is not None), default=None)
        if min_vd is not None:
            r = [x for x in r if x.vector_distance is None or x.vector_distance <= min_vd + _ADAPTIVE_WINDOW]
    return r, emb


@tool
def search_documents(
    query: str,
    state: Annotated[ChatState, InjectedState],
) -> str:
    """
    로컬 지식베이스(업로드된 PDF·문서·이미지)에서 query 와 관련된 내용을 검색합니다.
    사용자가 문서·자료·파일 기반 정보를 물어볼 때 반드시 이 도구를 사용하세요.

    Args:
        query: 문서에 실제로 등장할 단어나 표현을 그대로 사용하세요.
               - 고유명사(작품명·인명·브랜드)는 절대 번역하지 말고 원본 그대로: '미러코드', 'SAP', '하린'
               - 기술 개념은 영어 전문 용어가 정확도 높음: 'patent', 'embedding', 'failure detection'
               - ❌ 절대 금지: '미러코드' → 'novel',  '이혁주 소설' → 'fiction'  (엉뚱한 추상어)
    """
    session_id = state.get("session_id") or None
    _log.debug("search_documents query=%r session_id=%r", query, session_id)

    try:
        top_k = settings.default_retrieval_top_k

        # 1차 검색
        results, embedding = _hybrid_and_filter(query, session_id, top_k)

        # 한국어 쿼리인 경우: 순수 벡터(의미) 검색으로 2차 보완
        # → 청크에 영문 표기만 있어도 임베딩 유사도로 매칭 가능
        if _HAS_KOREAN.search(query):
            seen_ids = {r.chunk_id for r in results}
            extra_raw = _chunk_repo.hybrid_search(
                query_embedding=embedding,  # 같은 임베딩, keyword 비중 낮춤
                query_text="",             # 키워드 검색 비활성화 (빈 문자열)
                session_id=session_id,
                top_k=top_k,
            )
            # 1차와 동일한 거리 기준 적용, 중복 제거
            extra = [x for x in extra_raw
                     if x.chunk_id not in seen_ids
                     and x.vector_distance is not None
                     and x.vector_distance <= _MAX_VECTOR_DISTANCE]
            if extra:
                all_vds = [r.vector_distance for r in results if r.vector_distance is not None] + \
                          [r.vector_distance for r in extra if r.vector_distance is not None]
                min_vd = min(all_vds) if all_vds else 1.0
                extra = [x for x in extra if x.vector_distance <= min_vd + _ADAPTIVE_WINDOW]
                results = results + extra
                # 최종 vd 정렬
                results.sort(key=lambda x: x.vector_distance if x.vector_distance is not None else 1.0)
                results = results[:top_k]

        _log.debug("search_documents final: %d results", len(results))

        # 아무것도 없으면 파일명 기반 fallback
        if not results:
            results = _chunk_repo.search_by_filename(
                query=query,
                query_embedding=embedding,
                session_id=session_id,
                top_k=top_k,
            )

        if not results:
            return "관련 문서를 찾을 수 없습니다."

        # 최상위 결과의 벡터 거리로 전반적인 관련성 판단
        best_vd = min(
            (r.vector_distance for r in results if r.vector_distance is not None),
            default=None,
        )
        low_relevance = best_vd is not None and best_vd > _LOW_RELEVANCE_THRESHOLD

        # 사람이 읽기 쉬운 형식으로 결과 포맷
        parts = []
        refs = []
        for i, r in enumerate(results, 1):
            page_info = f", {r.page_number}페이지" if r.page_number else ""
            vd_str = f" [거리:{r.vector_distance:.2f}]" if r.vector_distance is not None else ""
            parts.append(f"[{i}] {r.file_name}{page_info}{vd_str}\n{r.content}")
            refs.append({
                "document_id": r.document_id,
                "file_name": r.file_name,
                "page_number": r.page_number,
                "score": round(r.score, 4),
            })

        # refs 를 메타데이터 블록으로 첨부 (chat_service 가 파싱해 출처 UI 표시)
        body = "\n\n---\n\n".join(parts)
        meta = f"\n\n[document_refs:{json.dumps(refs, ensure_ascii=False)}]"

        # 저관련성 경고: 에이전트가 할루시네이션 하지 않도록 명시적으로 알림
        if low_relevance:
            warning = (
                "⚠️ 관련성 낮음: 검색 결과가 질의와 의미적으로 멀리 떨어져 있습니다 "
                f"(최소 거리 {best_vd:.2f}). "
                "아래 내용이 질의와 실제로 무관하다면 '관련 문서를 찾지 못했습니다'라고 답하세요.\n\n"
            )
            return warning + body + meta

        return body + meta
    except Exception as e:
        _log.warning("search_documents error: %s", e, exc_info=True)
        return "문서 검색 중 오류가 발생했습니다."


@tool
def web_search(query: str) -> str:
    """
    인터넷에서 최신 정보를 검색합니다.
    특정 장소, 현재 이슈, 인물, 뉴스, 최신 정보 등 모델의 학습 데이터에 없을 수 있는 내용을 찾을 때 사용하세요.

    Args:
        query: 검색어. 구체적이고 명확한 한국어 또는 영어로 작성하세요.
               예: '헤이리 못난이유원지 정보', 'Heyri Art Valley amusement park'
    """
    import requests as _req

    client_id     = settings.naver_client_id
    client_secret = settings.naver_client_secret

    if not client_id or not client_secret:
        return "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 설정되지 않았습니다."

    try:
        url = "https://openapi.naver.com/v1/search/webkr.json"
        headers = {
            "X-Naver-Client-Id":     client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        params = {"query": query, "display": 5, "sort": "sim"}
        resp = _req.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            return "검색 결과를 찾을 수 없습니다."

        import re as _re
        _tag = _re.compile(r"<[^>]+>")
        parts = []
        for i, item in enumerate(items, 1):
            title       = _tag.sub("", item.get("title", ""))
            description = _tag.sub("", item.get("description", ""))
            link        = item.get("link", "")
            parts.append(f"[{i}] {title}\n{description}\n출처: {link}")
        return "\n\n---\n\n".join(parts)

    except Exception as exc:
        _log.error("web_search (Naver) failed: %s", exc)
        return f"웹 검색 중 오류가 발생했습니다: {exc}"
