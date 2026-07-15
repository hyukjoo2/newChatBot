"""
이미지 에이전트 노드: 업로드된 이미지의 시각적 내용을 moondream 등 vision 모델로 분석한다.
"""
from __future__ import annotations

import base64
import logging
from functools import lru_cache
from typing import Annotated, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState, ToolNode

from backend.chatbot.prompts import IMAGE_AGENT_SYSTEM_PROMPT
from backend.chatbot.state import ChatState
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings
from backend.database.repositories.document_repository import DocumentRepository
from backend.rag.embeddings import get_embeddings

_log = logging.getLogger(__name__)
_doc_repo = DocumentRepository()

# vision 모델은 인제스천/분석 시에만 로드되므로 캐시 없이 매번 생성
_VISION_DESCRIBE_PROMPT = (
    "Describe this image in detail. "
    "Include: objects, colors, scene/setting, any text visible, "
    "people (if present), and notable features. "
    "Be specific and thorough."
)


def _load_image_as_base64(path: str) -> Optional[str]:
    """이미지 파일을 base64 문자열로 읽는다."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        _log.warning("Failed to load image '%s': %s", path, e)
        return None


def _call_vision_model(image_b64: str, mime_type: str = "image/jpeg") -> str:
    """moondream(또는 설정된 vision 모델)으로 이미지를 설명한다."""
    vision_llm = ChatOllama(
        model=settings.ollama_vision_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_predict=512,
    )
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        {"type": "text", "text": _VISION_DESCRIBE_PROMPT},
    ])
    response = vision_llm.invoke([message])
    return response.content.strip() if response.content else ""


@tool
def describe_image(
    query: str,
    state: Annotated[ChatState, InjectedState],
) -> str:
    """
    업로드된 이미지를 vision 모델로 분석해 시각적 내용을 설명합니다.
    이미지에 포함된 객체, 색상, 장면, 텍스트 등을 상세히 설명합니다.

    Args:
        query: 찾을 이미지 파일명 또는 관련 키워드.
               특정 파일을 지정하려면 파일명을 포함하세요.
               예: 'sunset.jpg', '특허 도면', 'slide_01'
    """
    try:
        # 1. 이미지 문서 목록 조회 (mime_type이 image/* 인 것)
        all_docs = _doc_repo.list_active()
        image_docs = [
            d for d in all_docs
            if d.mime_type and d.mime_type.startswith("image/")
        ]

        if not image_docs:
            return "업로드된 이미지가 없습니다."

        # 2. 쿼리와 가장 일치하는 이미지 선택 (파일명 포함 여부)
        q_lower = query.lower()
        matched = None
        for doc in image_docs:
            if q_lower in doc.file_name.lower() or doc.file_name.lower() in q_lower:
                matched = doc
                break

        # 파일명 매칭 실패 시 첫 번째 이미지 사용
        if matched is None:
            matched = image_docs[0]

        _log.debug("Image agent analyzing: %s (%s)", matched.file_name, matched.original_path)

        # 3. 이미지 로드
        image_b64 = _load_image_as_base64(matched.original_path)
        if image_b64 is None:
            return f"이미지 파일을 읽을 수 없습니다: {matched.file_name}"

        # 4. Vision 모델로 설명 생성
        mime = matched.mime_type or "image/jpeg"
        try:
            description = _call_vision_model(image_b64, mime)
        except Exception as e:
            _log.warning("Vision model failed for '%s': %s", matched.file_name, e)
            description = ""

        # 5. 결과 포맷
        parts = [f"📷 **{matched.file_name}**"]
        if description:
            parts.append(f"\n[이미지 설명]\n{description}")
        else:
            parts.append("\n(Vision 모델 분석 실패 — vision 모델이 설치되어 있는지 확인하세요)")

        return "\n".join(parts)

    except Exception as e:
        _log.error("describe_image error: %s", e, exc_info=True)
        return f"이미지 분석 중 오류가 발생했습니다: {e}"


_IMAGE_TOOLS = [describe_image]


@lru_cache(maxsize=1)
def _get_image_agent_model() -> ChatOllama:
    base = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )
    return base.bind_tools(_IMAGE_TOOLS)


def image_agent_node(state: ChatState) -> dict:
    """이미지 에이전트 노드: describe_image 도구를 사용해 이미지를 분석하고 설명한다."""
    model = _get_image_agent_model()
    messages = [SystemMessage(content=IMAGE_AGENT_SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)
    if response.content and not getattr(response, "tool_calls", None):
        response.content = strip_leaked_prompt(response.content)
    return {"messages": [response]}
