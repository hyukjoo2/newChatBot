from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import RAG_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.config import settings

_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    return _INTRO_RE.sub("", text).strip()


@lru_cache(maxsize=1)
def _get_model() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "관련 문서를 찾을 수 없습니다."
    parts = []
    for i, c in enumerate(chunks, 1):
        page_info = f", {c['page_number']}페이지" if c.get("page_number") else ""
        parts.append(
            f"[{i}] {c['file_name']}{page_info}\n{c['content']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_rag_node(state: ChatState) -> dict:
    """RAG 답변 생성 노드: 검색된 청크를 컨텍스트로 포함하여 답변을 생성한다."""
    model = _get_model()
    chunks = state.get("retrieved_chunks") or []
    context = _build_context(chunks)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    response = model.invoke(messages)
    response.content = _strip_intro(response.content)

    # 출처 정보를 response metadata에 첨부
    sources = [
        {
            "document_id": c["document_id"],
            "file_name": c["file_name"],
            "page_number": c.get("page_number"),
            "score": c.get("score", 0),
        }
        for c in chunks
    ]
    if not hasattr(response, "response_metadata"):
        response.response_metadata = {}
    response.response_metadata["sources"] = sources

    return {"messages": [response], "retrieved_chunks": chunks}
