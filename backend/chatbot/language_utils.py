"""중국어 감지·필터 및 프롬프트 누수 제거 유틸."""
from __future__ import annotations

import re
from datetime import datetime


_WEEKDAY_KO = {
    "Monday": "월", "Tuesday": "화", "Wednesday": "수", "Thursday": "목",
    "Friday": "금", "Saturday": "토", "Sunday": "일",
}


def today_context() -> str:
    """오늘 날짜·시간을 LLM이 무시할 수 없는 명시적 지시문으로 반환한다.
    시스템 프롬프트 끝에 붙여서 마지막 지시로 작동하게 한다."""
    now = datetime.now()
    wd = _WEEKDAY_KO.get(now.strftime("%A"), "")
    date_str = now.strftime(f"%Y년 %m월 %d일 ({wd}요일)")
    time_str = now.strftime("%H:%M")
    return (
        f"\n[현재 날짜/시각 — 반드시 준수]\n"
        f"오늘: {date_str}  현재 시각: {time_str}\n"
        f"날짜·연도를 언급할 때는 반드시 위 날짜를 사용하세요. 다른 연도를 추측하지 마세요."
    )



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


# ── Chain-of-Thought: <think> 블록 분리 ─────────────────────────────────────

def split_think_content(
    text: str,
    in_think: bool = False,
) -> tuple[list[tuple[bool, str]], bool]:
    """
    스트리밍 텍스트에서 <think>...</think> 블록을 분리한다.

    Returns:
        segments: (is_think, content) 튜플 목록
                  is_think=True  → 추론 내용 (사용자에게 별도 표시, DB 미저장)
                  is_think=False → 실제 답변 내용
        new_in_think: 이 텍스트 처리 후의 thinking 상태 (다음 청크에 전달)

    Example:
        "안녕<think>잠깐 생각</think>하세요" → [
            (False, "안녕"),
            (True,  "잠깐 생각"),
            (False, "하세요"),
        ], False
    """
    segments: list[tuple[bool, str]] = []
    pos = 0

    while pos < len(text):
        if in_think:
            end = text.find("</think>", pos)
            if end == -1:
                # </think> 없음 — 나머지 전체가 thinking
                segments.append((True, text[pos:]))
                pos = len(text)
            else:
                if end > pos:
                    segments.append((True, text[pos:end]))
                in_think = False
                pos = end + 8  # len("</think>")
        else:
            start = text.find("<think>", pos)
            if start == -1:
                # <think> 없음 — 나머지 전체가 실제 내용
                if pos < len(text):
                    segments.append((False, text[pos:]))
                pos = len(text)
            else:
                if start > pos:
                    segments.append((False, text[pos:start]))
                in_think = True
                pos = start + 7  # len("<think>")

    return segments, in_think


# ── 마크다운 볼드/이탤릭 공백 정규화 ──────────────────────────────────────────
# LLM이 ** text ** 처럼 마커 안쪽에 공백을 넣는 경우
# 표준 마크다운 파서가 볼드로 인식하지 못해 ** 가 그대로 노출된다.
_BOLD_SPACE_RE = re.compile(r'\*\*[ \t]+([^*\n]+?)[ \t]+\*\*')
_ITALIC_SPACE_RE = re.compile(r'(?<!\*)\*[ \t]+([^*\n]+?)[ \t]+\*(?!\*)')


def fix_markdown_spacing(text: str) -> str:
    """** text ** → **text**, * text * → *text* 로 정규화한다."""
    text = _BOLD_SPACE_RE.sub(r'**\1**', text)
    text = _ITALIC_SPACE_RE.sub(r'*\1*', text)
    return text


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
