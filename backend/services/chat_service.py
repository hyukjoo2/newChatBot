from __future__ import annotations

import json
import re
from typing import Generator, Optional

from langchain_core.messages import HumanMessage, ToolMessage

from backend.chatbot.graph import get_graph
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings
from backend.database.repositories.message_repository import (
    ChatMessage,
    MessageRepository,
)
from backend.services.session_service import get_session, rename_session, update_session_last_message

_REFS_RE = re.compile(r"\[document_refs:(.+?)\]$", re.DOTALL)

# worker 에이전트 노드 이름 — 이 노드들의 응답만 사용자에게 스트리밍한다
_WORKER_NODES = {"rag_agent", "summary_agent", "task_agent", "email_agent", "image_agent", "direct_agent"}

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

    for chunk, metadata in graph.stream(
        {
            "messages": [HumanMessage(content=user_message)],
            "mode": mode,
            "session_id": session_id,
            "retrieved_chunks": None,
        },
        config=config,
        stream_mode="messages",
    ):
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
            full_response.append(content)
            yield content
        elif isinstance(content, list):
            for block in content:
                text = block if isinstance(block, str) else (block.get("text") if isinstance(block, dict) else "")
                if text:
                    full_response.append(text)
                    yield text

    response_text = strip_leaked_prompt("\n".join(full_response))
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
