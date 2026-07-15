from __future__ import annotations

import hashlib
import io
import os
import shutil
from pathlib import Path
from typing import Optional

from backend.config import settings
from backend.database.repositories.document_repository import Document, DocumentRepository


_doc_repo = DocumentRepository()


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_filename(filename: str) -> str:
    """
    동일한 file_name이 이미 존재하면 Windows 탐색기 방식으로
    '이름 (1).ext', '이름 (2).ext' 순서로 고유 이름을 반환한다.
    """
    existing = _doc_repo.list_file_names()
    if filename not in existing:
        return filename

    stem = Path(filename).stem
    ext = Path(filename).suffix
    seq = 1
    while True:
        candidate = f"{stem} ({seq}){ext}"
        if candidate not in existing:
            return candidate
        seq += 1


def _save_original_file(document_id: str, filename: str, data: bytes) -> str:
    """원본 파일을 data/uploads/{document_id}/ 에 저장하고 경로를 반환한다."""
    dest_dir = Path(settings.upload_dir) / document_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    dest_path.write_bytes(data)
    return str(dest_path)


def _save_extracted_text(document_id: str, text: str) -> str:
    """추출된 텍스트를 data/extracted/{document_id}/extracted.txt 에 저장하고 경로를 반환한다."""
    dest_dir = Path(settings.extracted_dir) / document_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "extracted.txt"
    dest_path.write_text(text, encoding="utf-8")
    return str(dest_path)


def register_document(
    uploaded_by: str,
    filename: str,
    data: bytes,
    mime_type: Optional[str] = None,
    category: Optional[str] = None,
    scope: str = "GLOBAL",
    session_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> str:
    """
    파일을 저장하고 documents 테이블에 레코드를 생성한다.

    Returns:
        document_id
    """
    file_hash = _compute_sha256(data)
    ext = Path(filename).suffix.lower()
    file_size = len(data)

    # 동일한 파일(해시 기준)이 이미 존재하면 처리
    existing = _doc_repo.get_by_hash(file_hash)
    if existing is not None:
        if scope == "SESSION":
            # SESSION 범위 업로드면 재인제스트 없이 기존 문서 id 반환 (컨텍스트 주입에 재사용)
            return existing.id
        raise ValueError(
            f"동일한 파일이 이미 업로드되어 있습니다: '{existing.file_name}'"
        )

    # 동일한 파일명이 이미 존재하면 (1), (2), ... 시퀀스 부여
    file_name = _unique_filename(filename)

    # 임시 document_id 없이 먼저 DB에 생성 (id는 DB가 자동 부여)
    document_id = _doc_repo.create(
        uploaded_by=uploaded_by,
        file_name=file_name,
        original_file_name=filename,
        original_path="",  # 경로는 파일 저장 후 업데이트
        mime_type=mime_type,
        file_extension=ext,
        file_size_bytes=file_size,
        file_hash_sha256=file_hash,
        category=category,
        scope=scope,
        session_id=session_id,
        chunk_size=chunk_size or settings.default_chunk_size,
        chunk_overlap=chunk_overlap or settings.default_chunk_overlap,
    )

    original_path = _save_original_file(document_id, file_name, data)

    # 경로 업데이트
    from backend.database.connection import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET original_path = %s WHERE id = %s",
                (original_path, document_id),
            )
            conn.commit()

    return document_id


def list_documents() -> list[Document]:
    return _doc_repo.list_all()


def list_active_documents() -> list[Document]:
    return _doc_repo.list_active()


def get_document(document_id: str) -> Optional[Document]:
    return _doc_repo.get_by_id(document_id)


def deactivate_document(document_id: str) -> None:
    _doc_repo.deactivate(document_id)


def reactivate_document(document_id: str) -> None:
    _doc_repo.reactivate(document_id)


def delete_document(document_id: str) -> None:
    """DB에서 소프트 삭제하고 파일 시스템의 원본 파일도 삭제한다."""
    doc = _doc_repo.get_by_id(document_id)
    _doc_repo.soft_delete(document_id)

    if doc:
        # 원본 파일 디렉토리 삭제
        upload_dir = Path(settings.upload_dir) / document_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)

        extracted_dir = Path(settings.extracted_dir) / document_id
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir, ignore_errors=True)
