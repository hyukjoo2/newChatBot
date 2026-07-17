"""
Weather Agent 노드: 날씨 전용 에이전트.

- get_weather 도구를 직접 호출 (LLM 도구 결정 불필요)
- 최소 프롬프트로 날씨 데이터를 자연스럽게 포맷
- /no_think, thinking 없음 → 빠르고 예측 가능
- 복잡한 추론/[FOLLOWUP] 로직 없음 → reasoning_agent 컨텍스트 오염 방지
"""
from __future__ import annotations

import re
import logging

from langchain_core.messages import HumanMessage, AIMessage

from backend.chatbot.state import ChatState
from backend.chatbot.tools import get_weather

_log = logging.getLogger(__name__)

# 쿼리에서 지역명 추출 패턴
_LOCATION_STRIP_RE = re.compile(
    r"\s*(날씨|기온|강수|미세먼지|우산|비|눈|폭염|태풍|weather|forecast)"
    r"[\s가-힣a-zA-Z]*$",
    re.IGNORECASE,
)
_META_STRIP_RE = re.compile(
    r"\s*(어때\??|어때요\??|알려줘|알려주세요|어떄\??)\s*$",
    re.IGNORECASE,
)


def _extract_location(query: str) -> str:
    """사용자 쿼리에서 지역명을 추출한다."""
    loc = _META_STRIP_RE.sub("", query.strip())
    loc = _LOCATION_STRIP_RE.sub("", loc).strip()
    return loc or query.strip()


def weather_agent_node(state: ChatState) -> dict:
    """
    날씨 에이전트:
    1. 쿼리에서 지역명 추출 (결정론적)
    2. get_weather 직접 호출 (LLM 포맷팅 없음 → 빠름)
    3. get_weather가 이미 잘 포맷된 텍스트를 반환하므로 그대로 사용
    """
    # 1. 지역명 추출
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            c = msg.content
            user_query = c if isinstance(c, str) else str(c)
            break

    location = _extract_location(user_query)
    _log.info("weather_agent: location=%r from query=%r", location, user_query)

    # 2. 날씨 데이터 직접 조회
    weather_data = get_weather.invoke({"location": location})

    # 조회 실패 시 즉시 반환 (hallucination 방지)
    if weather_data.startswith("날씨 정보 조회 중 오류"):
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y년 %m월 %d일")
        return {"messages": [AIMessage(content=(
            f"죄송합니다, {today} 현재 날씨 데이터를 가져올 수 없습니다.\n"
            f"잠시 후 다시 시도해 주세요."
        ))]}

    # 3. get_weather 결과를 그대로 반환 (LLM 호출 없음)
    return {"messages": [AIMessage(content=weather_data)]}
