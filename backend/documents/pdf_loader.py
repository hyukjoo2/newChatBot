from __future__ import annotations

import io
import logging
import unicodedata

from pypdf import PdfReader

_log = logging.getLogger(__name__)


def _is_garbled(text: str, threshold: float = 0.3) -> bool:
    """
    추출된 텍스트가 깨진 인코딩인지 판단한다.
    제어문자·Private Use 문자 비율이 threshold 이상이면 garbled로 간주.
    """
    if not text:
        return True
    bad = sum(
        1 for c in text
        if unicodedata.category(c) in ("Cc", "Cs", "Co")  # 제어문자, surrogate, private use
        or (ord(c) < 32 and c not in "\n\r\t")
    )
    return bad / len(text) >= threshold


def _ocr_pdf_bytes(data: bytes) -> list[tuple[int, str]]:
    """PyMuPDF로 PDF 페이지를 이미지로 변환한 뒤 pytesseract로 OCR한다."""
    try:
        import fitz  # pymupdf
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(f"OCR PDF 처리에 필요한 패키지가 없습니다: {e}. pip install pymupdf pytesseract 필요") from e

    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(doc, 1):
        mat = fitz.Matrix(2, 2)  # 2× 배율로 OCR 정확도 향상
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="kor+eng")
        pages.append((i, text.strip()))
    return pages


def _extract_with_pypdf(reader: PdfReader) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append((i, text.strip()))
    return pages


def _needs_ocr(pages: list[tuple[int, str]]) -> bool:
    """전체 페이지의 텍스트를 합쳐서 garbled 여부 판단."""
    total_text = "\n".join(t for _, t in pages)
    if not total_text.strip():
        return True
    return _is_garbled(total_text)


def extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]:
    """
    PDF 파일에서 페이지별 텍스트를 추출한다.
    pypdf 추출 결과가 깨진 경우 OCR로 fallback한다.

    Returns:
        [(page_number, text), ...]  (1-based page number)
    """
    with open(file_path, "rb") as f:
        data = f.read()
    return extract_text_from_pdf_bytes(data)


def extract_text_from_pdf_bytes(data: bytes) -> list[tuple[int, str]]:
    """바이트 데이터에서 페이지별 텍스트를 추출한다. 필요시 OCR fallback."""
    reader = PdfReader(io.BytesIO(data))
    pages = _extract_with_pypdf(reader)

    if _needs_ocr(pages):
        _log.info("PDF 텍스트 추출 품질 불량 → OCR fallback 시작 (%d pages)", len(pages))
        pages = _ocr_pdf_bytes(data)
        _log.info("OCR 완료: 총 %d 페이지", len(pages))

    return pages
