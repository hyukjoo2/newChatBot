from __future__ import annotations

import io
from typing import Optional


def extract_text_from_txt(file_path: str, encoding: str = "utf-8") -> str:
    """TXT 파일에서 텍스트를 읽는다."""
    with open(file_path, encoding=encoding, errors="replace") as f:
        return f.read()


def extract_text_from_txt_bytes(data: bytes, encoding: str = "utf-8") -> str:
    return data.decode(encoding, errors="replace")


def extract_text_from_md(file_path: str, encoding: str = "utf-8") -> str:
    """Markdown 파일에서 텍스트를 읽는다."""
    with open(file_path, encoding=encoding, errors="replace") as f:
        return f.read()


def extract_text_from_md_bytes(data: bytes, encoding: str = "utf-8") -> str:
    return data.decode(encoding, errors="replace")


def _docx_to_text(doc) -> str:
    """단락 + 표 셀 텍스트를 모두 추출한다."""
    parts = []
    for block in doc.element.body:
        tag = block.tag.split('}')[-1] if '}' in block.tag else block.tag
        if tag == 'p':
            from docx.text.paragraph import Paragraph
            text = Paragraph(block, doc).text
            if text.strip():
                parts.append(text)
        elif tag == 'tbl':
            from docx.table import Table
            for row in Table(block, doc).rows:
                row_text = "\t".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    parts.append(row_text)
    return "\n".join(parts)


def extract_text_from_docx(file_path: str) -> str:
    """DOCX 파일에서 텍스트를 추출한다."""
    try:
        import docx
        doc = docx.Document(file_path)
        return _docx_to_text(doc)
    except ImportError:
        raise RuntimeError("python-docx 패키지가 필요합니다: pip install python-docx")


def extract_text_from_docx_bytes(data: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(data))
        return _docx_to_text(doc)
    except ImportError:
        raise RuntimeError("python-docx 패키지가 필요합니다: pip install python-docx")
