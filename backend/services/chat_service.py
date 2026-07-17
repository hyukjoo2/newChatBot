from __future__ import annotations

import json
import re
from typing import Generator, Optional

from langchain_core.messages import HumanMessage, ToolMessage

from backend.chatbot.graph import get_graph
from backend.chatbot.language_utils import strip_leaked_prompt, split_think_content
from backend.config import settings
from backend.database.repositories.message_repository import (
    ChatMessage,
    MessageRepository,
)
from backend.services.session_service import get_session, rename_session, update_session_last_message

_REFS_RE = re.compile(r"\[document_refs:(.+?)\]$", re.DOTALL)

# 최종 응답에서 [FOLLOWUP] 잔여물 제거 (스트리밍 필터가 누락한 경우 대비)
_FOLLOWUP_CLEANUP_RE = re.compile(r"\[FOLLOWUP\].*?\[/FOLLOWUP\]", re.DOTALL)
_FOLLOWUP_PARTIAL_RE = re.compile(r"\[/?FOLLOWUP[^\]]*\]?")


def _strip_followup_stream(text: str, in_block: bool, buf: str) -> tuple[str, bool, str]:
    """
    스트리밍 중 [FOLLOWUP]...[/FOLLOWUP] 블록을 제거한다.
    토큰 경계에서 태그가 분할되는 경우(예: '[' → 'FOLLOWUP' → ']')도 처리한다.
    Returns: (visible_text, in_block, remainder_buf)
    """
    _OPEN  = "[FOLLOWUP]"
    _CLOSE = "[/FOLLOWUP]"

    def _partial_prefix(s: str, tag: str) -> int:
        """s의 끝이 tag의 접두사와 일치하는 최대 길이 반환 (0 = 부분 일치 없음)."""
        for n in range(min(len(tag) - 1, len(s)), 0, -1):
            if s.endswith(tag[:n]):
                return n
        return 0

    result = ""
    current = buf + text
    while current:
        if in_block:
            end = current.find(_CLOSE)
            if end == -1:
                return result, True, current   # 블록 안, 전체 버퍼 유지
            current = current[end + len(_CLOSE):]
            in_block = False
        else:
            start = current.find(_OPEN)
            if start == -1:
                # 끝 부분이 [FOLLOWUP] 시작의 일부일 수 있으면 버퍼에 남김
                partial = _partial_prefix(current, _OPEN)
                if partial:
                    result += current[:-partial]
                    return result, False, current[-partial:]
                result += current
                return result, False, ""
            result += current[:start]
            current = current[start + len(_OPEN):]
            in_block = True
    return result, in_block, ""

# worker 에이전트 노드 이름 — 이 노드들의 응답만 사용자에게 스트리밍한다
_WORKER_NODES = {"rag_agent", "summary_agent", "task_agent", "email_agent", "image_agent", "reasoning_agent", "web_search_agent", "weather_agent"}
# orchestrator_agent는 구조 업데이트를 통해 상태 메시지를 방출

# ── 진행 상황 상태 마커 ──────────────────────────────────────────────────────
STATUS_PREFIX   = "\x00STATUS:"
THINK_PREFIX    = "\x00THINK:"
FOLLOWUP_PREFIX = "\x00FOLLOWUP:"

# orchestrator가 수립한 작업 계획을 UI에 표시한다.
# 형식: PLAN_PREFIX + "1|agent|설명\n2|agent|설명\n..."
PLAN_PREFIX = "\x00PLAN:"

# 에이전트 이름 → 사용자 친화적 레이블
_AGENT_DISPLAY: dict[str, str] = {
    "rag_agent":        "📄 로컬 문서 검색",
    "web_search_agent": "🌐 웹 검색",
    "weather_agent":    "🌤️ 날씨 조회",
    "email_agent":      "✉️ 이메일 작성",
    "summary_agent":    "📝 문서 요약",
    "image_agent":      "🖼️ 이미지 분석",
    "reasoning_agent":  "💬 추론·답변",
    "task_agent":       "⚙️ 다중 작업",
}

# 각 노드의 사용자 친화적 상태 메시지
_NODE_LABELS: dict[str, str] = {
    "rag_agent":          "📄 문서를 검색해 답변을 작성하고 있습니다",
    "summary_agent":      "📝 문서를 요약하고 있습니다",
    "task_agent":         "⚙️ 작업을 처리하고 있습니다",
    "email_agent":        "✉️ 이메일을 작성하고 있습니다",
    "image_agent":        "🖼️ 이미지를 분석하고 있습니다",
    "reasoning_agent":    "💬 답변을 작성하고 있습니다",
    "web_search_agent":   "🌐 웹에서 최신 정보를 검색하고 있습니다",
    "weather_agent":      "🌤️ 날씨 정보를 조회하고 있습니다",
}

_msg_repo = MessageRepository()


def get_history(session_id: str) -> list[ChatMessage]:
    """세션의 전체 대화 기록을 반환한다."""
    return _msg_repo.list_by_session(session_id)


def chat(
    session_id: str,
    user_message: str,
    mode: str = "CHAT",
) -> str:
    """
    사용자 메시지에 대한 AI 응답을 생성하고 저장한다.

    Returns:
        AI 응답 텍스트
    """
    # 사용자 메시지 저장
    _msg_repo.save(
        session_id=session_id,
        role="USER",
        content=user_message,
        mode=mode,
    )

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "mode": mode,
            "session_id": session_id,
            "retrieved_chunks": None,
            "next": None,
            "task_list": None,
            "task_plan_ready": None,
            "answer_grade": None,
            "rag_retry_count": None,
            "pending_followup": None,
            "orchestrator_plan": None,
            "orchestrator_task_idx": None,
        },
        config=config,
    )

    ai_messages = result.get("messages", [])
    ai_response = ai_messages[-1] if ai_messages else None
    response_text = ""
    sources = []

    if ai_response:
        content = ai_response.content
        response_text = content if isinstance(content, str) else str(content)
        meta = getattr(ai_response, "response_metadata", {}) or {}
        sources = meta.get("sources", [])

    metadata = {"mode": mode.lower()}
    if sources:
        metadata["sources"] = sources

    _msg_repo.save(
        session_id=session_id,
        role="ASSISTANT",
        content=response_text,
        mode=mode,
        model_name=settings.ollama_model,
        metadata=metadata,
    )

    update_session_last_message(session_id)
    return response_text


def stream_chat(
    session_id: str,
    user_message: str,
    mode: str = "CHAT",
) -> Generator[str, None, None]:
    """
    사용자 메시지에 대한 AI 응답을 스트리밍으로 생성한다.

    Yields:
        응답 텍스트 토큰
    """
    _msg_repo.save(
        session_id=session_id,
        role="USER",
        content=user_message,
        mode=mode,
    )

    # 첫 메시지일 때 세션 제목 자동 업데이트
    session = get_session(session_id)
    if session and session.title == "새 대화":
        title = user_message.strip().splitlines()[0]
        if len(title) > 30:
            title = title[:30] + "…"
        rename_session(session_id, title)

    graph = get_graph()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 25,  # 무한루프 방지 (task_agent 최대 반복 횟수)
    }

    full_response = []
    sources = []
    _last_msg_id: str | None = None
    _node_visits: dict[str, int] = {}
    _in_think: bool = False
    _in_followup_block: bool = False   # [FOLLOWUP]...[/FOLLOWUP] 블록 스트리밍 중 숨김
    _followup_buf: str = ""             # 불완전한 [FOLLOWUP] 토큰 버퍼
    _followup_emitted: bool = False     # FOLLOWUP 토큰 중복 방출 방지

    for event_type, event_data in graph.stream(
        {
            "messages": [HumanMessage(content=user_message)],
            "mode": mode,
            "session_id": session_id,
            "retrieved_chunks": None,
            "next": None,
            "task_list": None,
            "task_plan_ready": None,
            "answer_grade": None,
            "rag_retry_count": None,
            "pending_followup": None,
            "orchestrator_plan": None,
            "orchestrator_task_idx": None,
        },
        config=config,
        stream_mode=["messages", "updates"],
    ):
        # ── updates: 노드 완료 이벤트 → 진행 상황 마커 방출 ─────────────────
        if event_type == "updates":
            node_name = next(iter(event_data), "")
            _node_visits[node_name] = _node_visits.get(node_name, 0) + 1

            if node_name == "orchestrator_agent":
                o_delta = event_data.get("orchestrator_agent") or {}
                plan = o_delta.get("orchestrator_plan") or []
                idx  = o_delta.get("orchestrator_task_idx") or 0

                if plan and idx == 0:
                    # ── Phase 1 완료: 작업 계획 방출 ─────────────────────────
                    # 형식: "1|agent|설명\n2|agent|설명"
                    lines = []
                    for t in plan:
                        lines.append(f"{t['id']}|{t['agent']}|{t['description']}")
                    yield f"{PLAN_PREFIX}" + "\n".join(lines)

                elif plan and idx < len(plan):
                    # ── Phase 2+ : 다음 작업 진행 상태 ──────────────────────
                    task  = plan[idx]
                    label = _AGENT_DISPLAY.get(task["agent"], "💡 작업")
                    desc  = task.get("description", "")[:40]
                    # RAG 실패 → 자동 web_search 폴백: 사용자에게 이유 알림 (Y/N 버튼 없음)
                    prev_failed_rag = (
                        idx > 0
                        and task["agent"] == "web_search_agent"
                        and plan[idx - 1].get("status") == "failed"
                        and plan[idx - 1].get("agent") == "rag_agent"
                    )
                    if prev_failed_rag:
                        yield f"{STATUS_PREFIX}🌐 로컬 문서에 관련 정보가 없어 웹에서 검색합니다..."
                    else:
                        yield f"{STATUS_PREFIX}▶ [{idx}/{len(plan)}] {label}: {desc}..."

                # orchestrator가 결정론적으로 도출한 후속 작업 제안
                # _followup_emitted: 동일 세션에서 다중 방출 방지
                pf = o_delta.get("pending_followup")
                if pf and not _followup_emitted:
                    yield f"{FOLLOWUP_PREFIX}{pf}"
                    _followup_emitted = True

            elif node_name == "rag_agent":
                # rag_agent 완료 → tool_calls 없으면 grade_answer 가 다음 실행
                try:
                    state_delta = event_data.get("rag_agent") or {}
                    msgs = state_delta.get("messages") or []
                    last_msg = msgs[-1] if msgs else None
                    has_tool_call = bool(getattr(last_msg, "tool_calls", None))
                    if not has_tool_call:
                        yield f"{STATUS_PREFIX}🔍 답변 적절성을 검토하고 있습니다..."
                except Exception:
                    pass

            elif node_name == "grade_answer":
                # grade_answer 완료 → 재검색 여부 알림
                grade_delta = event_data.get("grade_answer") or {}
                grade = grade_delta.get("answer_grade", "relevant")
                retry = grade_delta.get("rag_retry_count", 0)
                if grade == "not_relevant":
                    if retry >= 2:  # MAX_RETRIES
                        yield f"{STATUS_PREFIX}🌐 문서에서 찾지 못했습니다. 웹 검색으로 전환합니다..."
                    else:
                        yield f"{STATUS_PREFIX}🔄 답변이 충분하지 않습니다. 다시 검색합니다... (재시도 {retry}/2)"

            # reasoning_agent/web_search_agent FOLLOWUP은 orchestrator에서 일괄 처리
            # (두 곣에서 방출하면 동일 질문이 두 번 표시됨)

            continue  # updates 이벤트는 응답 내용이 아님

        # ── messages: 실제 응답 토큰 ─────────────────────────────────────────
        chunk, metadata = event_data
        node_name = metadata.get("langgraph_node", "")

        # ToolMessage 에서 출처 파싱 (search_documents 결과)
        if isinstance(chunk, ToolMessage):
            m = _REFS_RE.search(chunk.content or "")
            if m:
                try:
                    sources = json.loads(m.group(1))
                except Exception:
                    pass
            continue

        # worker 에이전트의 최종 텍스트 응답만 스트리밍 (supervisor는 라우팅만 하므로 제외)
        if node_name not in _WORKER_NODES:
            continue

        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_calls:
            # 도구 호출 시 진행 상황 메시지를 스트리밍해 사용자에게 피드백 제공
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tool_name = tc.get("name", "")
                args = tc.get("args") or {}
                if tool_name == "search_documents":
                    query = args.get("query", "")
                    status = f"\n> 🔍 **검색 중:** `{query}`\n\n" if query else "\n> 🔍 **문서 검색 중...**\n\n"
                    full_response.append(status)
                    yield status
            continue

        content = chunk.content
        if not content:
            continue

        # task_agent 가 새 메시지를 시작할 때 줄바꿈 구분자 주입
        # (여러 작업 결과가 이어붙여질 때 시각적 구분)
        if node_name == "task_agent":
            msg_id = getattr(chunk, "id", None)
            if msg_id and msg_id != _last_msg_id:
                if _last_msg_id is not None:   # 첫 번째 메시지는 구분자 불필요
                    sep = "\n\n"
                    full_response.append(sep)
                    yield sep
                _last_msg_id = msg_id

        if isinstance(content, str):
            # [FOLLOWUP]...[/FOLLOWUP] 블록 제거 (스트리밍 중 사용자에게 노출되면 안 됨)
            visible, _in_followup_block, _followup_buf = \
                _strip_followup_stream(content, _in_followup_block, _followup_buf)
            if not visible:
                continue
            segments, _in_think = split_think_content(visible, _in_think)
            for is_think, text in segments:
                if is_think:
                    yield f"{THINK_PREFIX}{text}"
                else:
                    full_response.append(text)
                    yield text
        elif isinstance(content, list):
            for block in content:
                raw = block if isinstance(block, str) else (block.get("text") if isinstance(block, dict) else "")
                if raw:
                    visible, _in_followup_block, _followup_buf = \
                        _strip_followup_stream(raw, _in_followup_block, _followup_buf)
                    if not visible:
                        continue
                    segments, _in_think = split_think_content(visible, _in_think)
                    for is_think, text in segments:
                        if is_think:
                            yield f"{THINK_PREFIX}{text}"
                        else:
                            full_response.append(text)
                            yield text

    response_text = strip_leaked_prompt("".join(full_response))
    # 스트리밍 필터가 놓친 [FOLLOWUP] 잔여물 최종 제거
    response_text = _FOLLOWUP_CLEANUP_RE.sub("", response_text)
    response_text = _FOLLOWUP_PARTIAL_RE.sub("", response_text).strip()
    msg_meta = {"mode": mode.lower()}
    if sources:
        msg_meta["sources"] = sources

    _msg_repo.save(
        session_id=session_id,
        role="ASSISTANT",
        content=response_text,
        mode=mode,
        model_name=settings.ollama_model,
        metadata=msg_meta,
    )

    update_session_last_message(session_id)


def inject_session_document_context(
    session_id: str,
    file_name: str,
    document_id: str,
) -> None:
    """
    SESSION-scoped 문서 업로드 후 요약을 LangGraph 체크포인터 상태에 주입한다.

    이후 대화에서 LLM이 해당 문서 내용을 컨텍스트로 인식할 수 있게 한다.
    """
    from langchain_core.messages import AIMessage
    from backend.database.repositories.chunk_repository import ChunkRepository

    chunk_repo = ChunkRepository()
    summary = chunk_repo.get_summary_chunk(document_id)

    if not summary:
        summary_text = f"📎 **{file_name}** 파일이 이 대화에 추가되었습니다."
    else:
        summary_text = f"📎 **{file_name}** 파일이 이 대화에 추가되었습니다.\n\n{summary}"

    graph = _get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # LangGraph 체크포인터에 AI 메시지로 추가 (이후 대화 컨텍스트에 포함됨)
    graph.update_state(
        config,
        {"messages": [AIMessage(content=summary_text)]},
    )

    # UI 표시용 DB 저장
    _msg_repo.save(
        session_id=session_id,
        role="ASSISTANT",
        content=summary_text,
        mode="CHAT",
        model_name="system",
        metadata={"type": "document_context", "document_id": document_id},
    )
