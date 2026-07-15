from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.config import settings
from backend.database.repositories.chunk_repository import ChunkRepository
from backend.database.repositories.document_repository import DocumentRepository
from backend.documents.loaders import (
    extract_text_from_docx_bytes,
    extract_text_from_md_bytes,
    extract_text_from_txt_bytes,
)
from backend.documents.pdf_loader import extract_text_from_pdf_bytes
from backend.documents.image_loader import ocr_image_bytes
from backend.rag.chunker import chunk_pages, chunk_text
from backend.rag.embeddings import get_embeddings, get_embeddings_batch
from backend.services.document_service import _save_extracted_text

_log = logging.getLogger(__name__)

_doc_repo = DocumentRepository()
_chunk_repo = ChunkRepository()

_EMBEDDING_MODEL = settings.ollama_embedding_model
_EMBEDDING_DIM = 768  # nomic-embed-text 기준

# summary 청크는 chunk_index = -1 로 식별
_SUMMARY_CHUNK_INDEX = -1

_SUMMARY_SYSTEM = (
    "당신은 문서를 간결하고 풍부하게 요약하는 전문가입니다. "
    "검색 가능성을 높이기 위해 한국어와 영어 키워드를 모두 포함하세요."
)

_SUMMARY_PROMPT = """\
다음 문서를 읽고, 의미 검색에 최적화된 요약을 작성하세요.

파일명: {file_name}

문서 내용:
{sample}

아래 항목을 반드시 포함하세요. 영어 용어는 한국어 뒤에 괄호로 병기하세요. (예: 특허 (patent), 상태 불일치 (state inconsistency)):
- 문서 유형: (예: 특허 (patent), 논문 (paper), 발표자료 (slides) 등)
- 주제 및 핵심 목적:
- 주요 기술 용어: 한국어 (영어) 형식으로 나열
- 영어 키워드 (검색용): 영어 단어만 공백으로 구분하여 나열
- 저자 / 기관 (있으면):
- 핵심 내용 3~5줄 요약:
"""


def _generate_summary(file_name: str, full_text: str) -> str | None:
    """LLM을 사용해 문서 semantic summary를 생성한다.

    반환값은 chunk_index=-1 로 저장되어 hybrid search 에 자동 포함된다.
    실패 시 None 반환 (ingestion을 중단하지 않음).
    """
    try:
        # 입력이 너무 길면 앞뒤만 샘플링
        if len(full_text) > 5000:
            sample = full_text[:3000] + "\n\n[...중략...]\n\n" + full_text[-2000:]
        else:
            sample = full_text

        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            num_predict=700,
        )
        messages = [
            SystemMessage(content=_SUMMARY_SYSTEM),
            HumanMessage(content=_SUMMARY_PROMPT.format(file_name=file_name, sample=sample)),
        ]
        response = model.invoke(messages)
        summary = response.content.strip()
        if not summary:
            return None
        # 파일명을 맨 앞에 명시해 파일명 기반 검색 품질도 높임
        return f"[{file_name}] 문서 요약\n\n{summary}"
    except Exception as e:
        _log.warning("Summary generation failed for '%s': %s", file_name, e)
        return None


def _extract_text(document_id: str, mime_type: str, data: bytes, file_extension: str = "") -> list[tuple[int, str]]:
    """파일 형식에 따라 텍스트를 추출한다. MIME 불명확 시 확장자로 fallback."""
    mime = (mime_type or "").lower()
    ext = (file_extension or "").lower().lstrip(".")

    if "pdf" in mime or ext == "pdf":
        return extract_text_from_pdf_bytes(data)

    if "image" in mime or ext in ("jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"):
        text = ocr_image_bytes(data)
        return [(1, text)]

    if "word" in mime or "docx" in mime or \
            mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or \
            ext == "docx":
        text = extract_text_from_docx_bytes(data)
        return [(1, text)]

    if "markdown" in mime or mime in ("text/markdown", "text/x-markdown") or ext in ("md", "markdown"):
        text = extract_text_from_md_bytes(data)
        return [(1, text)]

    # TXT, 그 외
    text = extract_text_from_txt_bytes(data)
    return [(1, text)]


def ingest_document(document_id: str) -> None:
    """
    문서를 처음부터 끝까지 처리한다.

    1. 원본 파일 읽기
    2. 텍스트 추출
    3. 청크 생성
    4. 임베딩 생성
    5. pgvector 저장
    6. documents 상태 → READY
    """
    doc = _doc_repo.get_by_id(document_id)
    if doc is None:
        raise ValueError(f"문서를 찾을 수 없습니다: {document_id}")

    try:
        # 1. 원본 파일 읽기
        _doc_repo.update_status(document_id, "EXTRACTING")
        with open(doc.original_path, "rb") as f:
            data = f.read()

        # 2. 텍스트 추출
        pages = _extract_text(document_id, doc.mime_type or "", data, doc.file_extension or "")
        full_text = "\n\n".join(text for _, text in pages)

        extracted_path = _save_extracted_text(document_id, full_text)
        _doc_repo.update_extracted_path(document_id, extracted_path)
        _doc_repo.update_status(document_id, "EXTRACTED")

        # 3. 청크 생성
        _doc_repo.update_status(document_id, "CHUNKING")
        chunks = chunk_pages(
            pages=pages,
            chunk_size=doc.chunk_size,
            chunk_overlap=doc.chunk_overlap,
        )

        if not chunks:
            raise ValueError("청크를 생성할 수 없습니다. 추출된 텍스트가 없습니다.")

        _doc_repo.update_status(document_id, "CHUNKED")

        # 4. 임베딩 생성
        _doc_repo.update_status(document_id, "EMBEDDING")
        texts = [c.content for c in chunks]
        embeddings = get_embeddings_batch(texts)

        # 5. 기존 청크 삭제 후 저장
        _chunk_repo.delete_by_document(document_id)
        chunk_rows = [
            {
                "document_id": document_id,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "content": c.content,
                "token_count": c.token_count,
                "embedding": emb,
                "metadata": {},
            }
            for c, emb in zip(chunks, embeddings)
        ]
        _chunk_repo.save_batch(chunk_rows)

        # 6. 완료 처리
        page_count = max((p for p, _ in pages), default=1)
        _doc_repo.mark_ready(
            document_id=document_id,
            chunk_count=len(chunks),
            page_count=page_count,
            embedding_model=_EMBEDDING_MODEL,
            embedding_dimension=len(embeddings[0]) if embeddings else _EMBEDDING_DIM,
        )

        # 7. 문서 semantic summary 생성 및 저장 (chunk_index = -1)
        #    실패해도 ingestion 결과에 영향 없음
        _doc_repo.update_status(document_id, "READY")  # mark_ready가 이미 설정하지만 명시
        summary_text = _generate_summary(doc.file_name, full_text)
        if summary_text:
            try:
                summary_emb = get_embeddings(summary_text)
                _chunk_repo.save(
                    document_id=document_id,
                    chunk_index=_SUMMARY_CHUNK_INDEX,
                    content=summary_text,
                    embedding=summary_emb,
                    metadata={"is_summary": True},
                )
                _log.info("Summary chunk saved for document %s", document_id)
            except Exception as e:
                _log.warning("Failed to save summary chunk for %s: %s", document_id, e)

    except Exception as e:
        _doc_repo.mark_failed(document_id, str(e))
        raise


def reprocess_document(document_id: str) -> None:
    """문서를 재처리한다 (기존 청크 삭제 후 다시 임베딩)."""
    _doc_repo.update_status(document_id, "UPLOADED")
    ingest_document(document_id)


def reindex_summaries(document_ids: list[str] | None = None) -> dict[str, str]:
    """
    기존 문서의 summary 청크를 (재)생성한다.

    Args:
        document_ids: 처리할 문서 ID 목록. None 이면 summary가 없는 모든 활성 문서를 처리.

    Returns:
        {document_id: "ok" | "skipped" | "failed"} 결과 맵
    """
    import psycopg
    from backend.database.connection import get_connection

    if document_ids is None:
        # summary 청크(chunk_index = -1)가 없는 활성 문서 찾기
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.id FROM documents d
                    WHERE d.is_active = TRUE
                      AND d.deleted_at IS NULL
                      AND d.status = 'READY'
                      AND NOT EXISTS (
                          SELECT 1 FROM document_chunks dc
                          WHERE dc.document_id = d.id
                            AND dc.chunk_index = %s
                      )
                    """,
                    (_SUMMARY_CHUNK_INDEX,),
                )
                document_ids = [str(r[0]) for r in cur.fetchall()]

    results: dict[str, str] = {}
    for doc_id in document_ids:
        doc = _doc_repo.get_by_id(doc_id)
        if doc is None:
            results[doc_id] = "skipped"
            continue

        try:
            # 추출된 텍스트 파일 읽기 (재추출 없이 기존 텍스트 재사용)
            extracted_path = doc.extracted_text_path
            if extracted_path and Path(extracted_path).exists():
                full_text = Path(extracted_path).read_text(encoding="utf-8")
            else:
                # 추출 텍스트 없으면 원본 재추출
                with open(doc.original_path, "rb") as f:
                    data = f.read()
                pages = _extract_text(doc_id, doc.mime_type or "", data)
                full_text = "\n\n".join(text for _, text in pages)

            # 기존 summary 청크 삭제 후 재생성
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM document_chunks WHERE document_id = %s AND chunk_index = %s",
                        (doc_id, _SUMMARY_CHUNK_INDEX),
                    )
                    conn.commit()

            summary_text = _generate_summary(doc.file_name, full_text)
            if summary_text:
                summary_emb = get_embeddings(summary_text)
                _chunk_repo.save(
                    document_id=doc_id,
                    chunk_index=_SUMMARY_CHUNK_INDEX,
                    content=summary_text,
                    embedding=summary_emb,
                    metadata={"is_summary": True},
                )
                results[doc_id] = "ok"
                _log.info("Summary reindexed: %s (%s)", doc.file_name, doc_id)
            else:
                results[doc_id] = "failed"
        except Exception as e:
            _log.error("reindex_summaries error for %s: %s", doc_id, e)
            results[doc_id] = "failed"

    return results
