from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings


@dataclass
class Chunk:
    content: str
    chunk_index: int
    page_number: Optional[int]
    start_offset: int
    end_offset: int
    token_count: int


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    page_number: Optional[int] = None,
) -> list[Chunk]:
    """텍스트를 청크로 분할한다."""
    size = chunk_size or settings.default_chunk_size
    overlap = chunk_overlap or settings.default_chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    splits = splitter.create_documents([text])
    chunks = []
    for i, doc in enumerate(splits):
        content = doc.page_content.strip()
        if not content:
            continue
        chunks.append(
            Chunk(
                content=content,
                chunk_index=i,
                page_number=page_number,
                start_offset=0,
                end_offset=len(content),
                token_count=len(content.split()),
            )
        )
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Chunk]:
    """
    페이지별 텍스트를 청크로 분할한다.

    pages: [(page_number, text), ...]
    """
    all_chunks: list[Chunk] = []
    global_index = 0
    for page_number, text in pages:
        page_chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page_number,
        )
        for chunk in page_chunks:
            chunk.chunk_index = global_index
            global_index += 1
            all_chunks.append(chunk)
    return all_chunks
