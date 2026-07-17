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


@tool
def get_weather(location: str) -> str:
    """
    특정 지역의 현재 날씨와 오늘·내일 예보를 조회합니다.
    기온, 체감온도, 습도, 바람, 강수확률 등 실제 수치를 반환합니다.

    Args:
        location: 날씨를 확인할 지역명 (한국어 또는 영어)
                  예: '서울', '광명 하안1동', '부산 해운대', '제주도', 'Tokyo'
    """
    import requests as _req
    from datetime import datetime as _dt

    # 풍향 코드 → 한국어
    _WIND_DIR = {
        "N":"북", "NNE":"북북동", "NE":"북동", "ENE":"동북동",
        "E":"동", "ESE":"동남동", "SE":"남동", "SSE":"남남동",
        "S":"남", "SSW":"남남서", "SW":"남서", "WSW":"서남서",
        "W":"서", "WNW":"서북서", "NW":"북서", "NNW":"북북서",
    }
    # 날씨 설명 영→한 번역
    _WEATHER_KO = {
        "Sunny": "맑음", "Clear": "맑음", "Partly cloudy": "구름 조금",
        "Partly Cloudy": "구름 조금", "Cloudy": "흐림", "Overcast": "흐림",
        "Mist": "안개", "Fog": "안개", "Freezing fog": "결빙 안개",
        "Patchy rain possible": "비 올 수 있음", "Patchy snow possible": "눈 올 수 있음",
        "Blowing snow": "눈보라", "Blizzard": "눈보라",
        "Thundery outbreaks possible": "천둥 올 수 있음",
        "Patchy light drizzle": "약한 이슬비", "Light drizzle": "이슬비",
        "Freezing drizzle": "결빙 이슬비", "Heavy freezing drizzle": "강한 결빙 이슬비",
        "Patchy light rain": "가끔 약한 비", "Light rain": "가벼운 비",
        "Moderate rain at times": "때때로 보통 비", "Moderate rain": "보통 비",
        "Heavy rain at times": "때때로 강한 비", "Heavy rain": "강한 비",
        "Light snow": "약한 눈", "Moderate snow": "보통 눈", "Heavy snow": "폭설",
        "Ice pellets": "진눈깨비", "Light rain shower": "가벼운 소나기",
        "Moderate or heavy rain shower": "강한 소나기", "Torrential rain shower": "폭우",
        "Thunderstorm": "뇌우", "Thunder": "천둥",
    }

    try:
        url = f"https://wttr.in/{_req.utils.quote(location)}?format=j1&lang=ko"
        headers = {"User-Agent": "curl/7.68.0", "Accept-Language": "ko-KR"}
        resp = _req.get(url, headers=headers, timeout=10)
        # wttr.in은 간헐적으로 500을 반환함 — 한 번 재시도
        if resp.status_code >= 500:
            import time as _time
            _time.sleep(1)
            resp = _req.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        cur = data["current_condition"][0]
        weather_days = data.get("weather", [])
        today = weather_days[0] if weather_days else {}
        tomorrow = weather_days[1] if len(weather_days) > 1 else {}

        # 현재 시각 한 번만 가져와 재사용
        _now = _dt.now()
        now_h = _now.hour

        # 현재 상태
        temp      = cur.get("temp_C", "?")
        feels     = cur.get("FeelsLikeC", "?")
        humidity  = cur.get("humidity", "?")
        wind_kmph = cur.get("windspeedKmph", "?")
        wind_dir  = _WIND_DIR.get(cur.get("winddir16Point", ""), cur.get("winddir16Point", ""))
        uv        = cur.get("uvIndex", "?")
        precip_mm = cur.get("precipMM", "0")
        raw_desc = (
            (cur.get("lang_ko") or [{}])[0].get("value", "").strip()
            or (cur.get("weatherDesc") or [{}])[0].get("value", "").strip()
        )
        desc_ko = _WEATHER_KO.get(raw_desc, raw_desc)

        # 오늘 예보
        t_max  = today.get("maxtempC", "?")
        t_min  = today.get("mintempC", "?")
        hourly = today.get("hourly", [])

        future_rain = [
            (int(h.get("time", "0")) // 100, int(h.get("chanceofrain", 0)))
            for h in hourly
            if int(h.get("time", "0")) // 100 >= now_h
        ]
        rain_pct = 0
        rain_time_label = ""
        if future_rain:
            peak_h, rain_pct = max(future_rain, key=lambda x: x[1])
            if rain_pct > 0:
                if peak_h < 12:   rain_time_label = f"오전 {peak_h}시경"
                elif peak_h < 18: rain_time_label = f"오후 {peak_h - 12}시경"
                else:             rain_time_label = f"저녁 {peak_h - 12}시경"

        # 내일 예보
        tmr_max = tomorrow.get("maxtempC", "?")
        tmr_min = tomorrow.get("mintempC", "?")
        tmr_rain = max(
            int(h.get("chanceofrain", 0)) for h in tomorrow.get("hourly", [{"chanceofrain": 0}])
        )

        # 오늘 날짜 (사용자에게 날짜 기준 명확히 제공)
        weekday_ko = {"Monday":"월","Tuesday":"화","Wednesday":"수","Thursday":"목",
                      "Friday":"금","Saturday":"토","Sunday":"일"}
        today_str_ko = _now.strftime(f"%Y년 %m월 %d일 ({weekday_ko.get(_now.strftime('%A'), '')}요일)")

        # 강수확률 표기: 현재 비 없어도 오후에 높을 수 있으므로 시간 컨텍스트 포함
        rain_label = (f"{rain_pct}% ({rain_time_label} 최대)" if rain_time_label
                      else f"{rain_pct}%")

        lines = [
            f"📍 **{location}** 현재 날씨  ({today_str_ko} 기준)",
            f"날씨 상태: {desc_ko}",
            f"🌡 기온: {temp}°C (체감 {feels}°C)",
            f"💧 습도: {humidity}%",
            f"🌬 바람: {wind_kmph}km/h ({wind_dir}방향)" if wind_dir else f"🌬 바람: {wind_kmph}km/h",
            f"🌧 강수량: {precip_mm}mm  |  오늘 남은 시간 최대 강수확률: {rain_label}",
            f"☀️ 자외선 지수: {uv}",
            "",
            f"📅 **오늘 예보 ({today_str_ko})**: 최저 {t_min}°C / 최고 {t_max}°C",
            f"📅 **내일 예보**: 최저 {tmr_min}°C / 최고 {tmr_max}°C  (강수확률 {tmr_rain}%)",
        ]

        # ── 결정론적 권장사항 (수치 임계값 기반, LLM 판단 없음) ───────────────
        advices: list[str] = []
        try:
            temp_i = int(temp) if temp != "?" else 0
            uv_i   = int(uv)   if uv != "?"   else 0
            wind_i = int(wind_kmph) if wind_kmph != "?" else 0
            hum_i  = int(humidity)  if humidity != "?"  else 0
        except (ValueError, TypeError):
            temp_i = uv_i = wind_i = hum_i = 0

        if rain_pct >= 60:
            advices.append(f"🌂 오늘 강수확률 {rain_pct}% — 우산을 챙기세요")
        if tmr_rain >= 70:
            advices.append(f"☔ 내일 강수확률 {tmr_rain}% — 내일도 우산 필요")
        if temp_i >= 33:
            advices.append(f"🔥 기온 {temp}°C — 폭염 주의, 수분 보충 필수")
        if uv_i >= 6:
            advices.append(f"🕶️ 자외선 지수 {uv} (높음) — 자외선 차단제·선글라스 착용")
        if hum_i >= 80:
            advices.append(f"💦 습도 {humidity}% — 불쾌지수 높음, 실내 냉방 활용")
        if wind_i >= 30:
            advices.append(f"💨 풍속 {wind_kmph}km/h — 강풍 주의")

        if advices:
            lines += ["", "[날씨 권장사항]"] + [f"  • {a}" for a in advices]

        return "\n".join(lines)

    except Exception as exc:
        _log.error("get_weather failed for %r: %s", location, exc)
        return f"날씨 정보 조회 중 오류가 발생했습니다: {exc}"
