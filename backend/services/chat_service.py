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

# worker 에이전트 노드 이름 — 이 노드들의 응답만 사용자에게 스트리밍한다
_WORKER_NODES = {"rag_agent", "summary_agent", "task_agent", "email_agent", "image_agent", "direct_agent", "web_search_agent"}

# ── 진행 상황 상태 마커 ──────────────────────────────────────────────────────
# 이 접두사로 시작하는 토큰은 UI가 별도 진행 표시로 처리한다.
# full_response 에 포함되지 않으므로 DB에 저장되지 않는다.
STATUS_PREFIX = "\x00STATUS:"

# 모델이 <think>...</think> 블록을 생성할 때 추론 내용을 별도 채널로 스트리밍한다.
# full_response 에 포함되지 않으므로 DB에 저장되지 않는다.
THINK_PREFIX = "\x00THINK:"

# 각 노드의 사용자 친화적 상태 메시지
_NODE_LABELS: dict[str, str] = {
    "rag_agent":          "📄 문서를 검색해 답변을 작성하고 있습니다",
    "summary_agent":      "📝 문서를 요약하고 있습니다",
    "task_agent":         "⚙️ 작업을 처리하고 있습니다",
    "email_agent":        "✉️ 이메일을 작성하고 있습니다",
    "image_agent":        "🖼️ 이미지를 분석하고 있습니다",
    "direct_agent":       "💬 답변을 작성하고 있습니다",
    "web_search_agent":   "🌐 웹에서 최신 정보를 검색하고 있습니다",
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
    _last_msg_id: str | None = None  # task_agent 메시지 경계 감지용
    _node_visits: dict[str, int] = {}  # 노드별 updates 이벤트 횟수 추적
    _in_think: bool = False            # <think> 블록 파싱 상태

    for event_type, event_data in graph.stream(
        {
            "messages": [HumanMessage(content=user_message)],
            "mode": mode,
            "session_id": session_id,
            "retrieved_chunks": None,
        },
        config=config,
        stream_mode=["messages", "updates"],
    ):
        # ── updates: 노드 완료 이벤트 → 진행 상황 마커 방출 ─────────────────
        if event_type == "updates":
            node_name = next(iter(event_data), "")
            _node_visits[node_name] = _node_visits.get(node_name, 0) + 1

            if node_name == "supervisor":
                # supervisor 완료 → 어떤 에이전트가 다음에 실행될지 알림
                state_delta = event_data.get("supervisor") or {}
                next_agent = (state_delta.get("next") or "").strip().lower()
                label = _NODE_LABELS.get(next_agent, "")
                yield f"{STATUS_PREFIX}{label or '🧭 에이전트를 선택하고 있습니다'}"

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
            segments, _in_think = split_think_content(content, _in_think)
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
                    segments, _in_think = split_think_content(raw, _in_think)
                    for is_think, text in segments:
                        if is_think:
                            yield f"{THINK_PREFIX}{text}"
                        else:
                            full_response.append(text)
                            yield text

    response_text = strip_leaked_prompt("".join(full_response))
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
