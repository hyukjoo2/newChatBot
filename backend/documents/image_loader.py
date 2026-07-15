from __future__ import annotations

import io
from typing import Optional

from PIL import Image
import pytesseract


def ocr_image_file(file_path: str, lang: str = "kor+eng") -> str:
    """이미지 파일에서 OCR로 텍스트를 추출한다."""
    image = Image.open(file_path)
    return pytesseract.image_to_string(image, lang=lang)


def ocr_image_bytes(data: bytes, lang: str = "kor+eng") -> str:
    """바이트 데이터 이미지에서 OCR로 텍스트를 추출한다."""
    image = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(image, lang=lang)
