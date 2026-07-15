"""중국어 감지·필터 및 프롬프트 누수 제거 유틸."""
from __future__ import annotations

import re


# ── 프롬프트 누수 패턴 ──────────────────────────────────────────────────────
# 시스템 프롬프트에서 흘러나온 내용을 감지·제거한다.
_LEAKED_LINE_RE = re.compile(
    r"^\s*("
    r"\[언어\s*규칙"           # [언어 규칙...]
    r"|\[도구\s*사용\s*규칙"   # [도구 사용 규칙]
    r"|\[답변\s*규칙"          # [답변 규칙]
    r"|\[검색\s*쿼리"          # [검색 쿼리 작성 규칙]
    r"|\[지식베이스"           # [지식베이스...]
    r"|\[현재\s*지식베이스"    # [현재 지식베이스...]
    r"|\[라우팅"               # [라우팅...]
    r"|▶\s+"                  # ▶ 불릿 (프롬프트 지시문)
    r"|✅\s+"                  # ✅ (프롬프트 예시)
    r"|❌\s+"                  # ❌ (프롬프트 예시)
    r")",
    re.MULTILINE,
)

_LEAKED_BLOCK_START_RE = re.compile(
    r"\[언어\s*규칙|당신은\s+.*?(전문가|비서|관리자)입니다",
)


def strip_leaked_prompt(text: str) -> str:
    """
    LLM 응답에서 시스템 프롬프트가 누수된 부분을 제거한다.

    - 프롬프트 지시문 패턴으로 시작하는 줄 제거
    - 프롬프트 블록 시작이 감지되면 해당 지점 이후 잘라냄
    """
    if not text:
        return text

    # 프롬프트 블록 시작 위치 감지 → 그 앞부분만 사용
    m = _LEAKED_BLOCK_START_RE.search(text)
    if m and m.start() > 0:
        text = text[: m.start()]

    # 개별 누수 줄 제거
    lines = text.splitlines()
    clean = [ln for ln in lines if not _LEAKED_LINE_RE.match(ln)]
    return "\n".join(clean).strip()


def has_chinese(text: str, threshold: float = 0.15) -> bool:
    """
    텍스트에 중국어가 일정 비율 이상 포함되어 있는지 감지한다.

    한국어 한자(漢字)와 구별하기 위해 전체 비한글 문자 중 CJK 비율을 사용한다.
    threshold: CJK 문자 비율 기준 (기본 15% 초과 시 중국어로 판단)
    """
    if not text:
        return False

    cjk_count = 0
    non_space_count = 0
    hangul_count = 0

    for ch in text:
        cp = ord(ch)
        if ch.isspace():
            continue
        non_space_count += 1
        # 한글 (AC00–D7A3, 1100–11FF, 3130–318F)
        if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            hangul_count += 1
        # CJK Unified Ideographs (4E00–9FFF) — 중국어/일본어 한자 영역
        elif 0x4E00 <= cp <= 0x9FFF:
            cjk_count += 1

    if non_space_count == 0:
        return False

    # 한글이 많은 텍스트에서 CJK가 소수 섞인 경우는 무시
    cjk_ratio = cjk_count / non_space_count
    return cjk_ratio > threshold
