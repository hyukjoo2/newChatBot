"""
Task Agent 노드: Plan → Execute 2단계로 다중 작업을 처리한다.

Phase 1 (Plan):  state["task_list"]가 없을 때
  - 도구 없는 모델로 📋 작업 목록만 생성 (항상 텍스트 출력 보장)
  - task_list 파싱 후 state에 저장

Phase 2 (Execute): state["task_list"]가 있을 때
  - 도구 바인딩 모델로 각 작업 실행 (ReAct 루프)
  - 검색·분석·작성을 순서대로 수행
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from backend.chatbot.prompts import TASK_PLAN_PROMPT, TASK_EXECUTE_PROMPT
from backend.chatbot.state import ChatState, TaskItem
from backend.chatbot.tools import search_documents
from backend.chatbot.language_utils import strip_leaked_prompt
from backend.config import settings

_log = logging.getLogger(__name__)

_TOOLS = [search_documents]


def _normalize_query(text: str) -> str:
    """llama3.1:8b 등이 한글 음절 사이에 삽입하는 공백을 제거해 검색 쿼리를 정규화한다.

    예: 'R AG 에서 미 러 코드 를 열 고' → 'RAG에서 미러코드를 열고'
    """
    # 1. 인접 한글 음절(가-힣) 사이 단일 공백 제거 (반복 적용)
    prev = ""
    while prev != text:
        prev = text
        text = re.sub(r"([\uAC00-\uD7A3]) ([\uAC00-\uD7A3])", r"\1\2", text)
    # 2. 대문자 약어 내 공백 제거 (예: R AG → RAG, G PT → GPT)
    text = re.sub(r"\b([A-Z]{1,3}) ([A-Z])", r"\1\2", text)
    return text.strip()
_INTRO_RE = re.compile(
    r"(안녕하세요[,!.]?\s*)?"
    r"저는\s*(셀마|Selma)(\s*\([^)]+\))?(\s*(AI\s*)?비서)?\s*입니다[!.]?\s*",
    re.IGNORECASE,
)


def _strip_intro(text: str) -> str:
    return _INTRO_RE.sub("", text).strip()


def _parse_plan_revisions(text: str, current_tasks: list[TaskItem]) -> list[TaskItem] | None:
    """🔄 계획 수정/추가 마커를 파싱해 업데이트된 task_list를 반환한다.
    변경이 없으면 None을 반환한다."""
    modified = False
    updated: list[dict] = [dict(t) for t in current_tasks]

    # 수정: 🔄 계획 수정: 작업 N → "새 내용"
    for m in re.finditer(
        r'🔄\s*계획\s*수정\s*:\s*작업\s*(\d+)\s*[→\-]+\s*[""""]?([^""""\n(]+)',
        text,
    ):
        tid = int(m.group(1))
        new_desc = m.group(2).strip().rstrip('"""".').strip()
        for task in updated:
            if task["id"] == tid:
                task["description"] = new_desc
                task["status"] = "revised"
                modified = True
                _log.info("task_agent: plan revised task %d → %s", tid, new_desc[:60])
                break

    # 추가: 🔄 계획 추가: 작업 N → "새 작업"
    for m in re.finditer(
        r'🔄\s*계획\s*추가\s*:\s*작업\s*(\d+)\s*[→\-]+\s*[""""]?([^""""\n(]+)',
        text,
    ):
        tid = int(m.group(1))
        desc = m.group(2).strip().rstrip('"""".').strip()
        if not any(t["id"] == tid for t in updated):
            updated.append(TaskItem(id=tid, description=desc, status="pending", result=None))
            modified = True
            _log.info("task_agent: plan added task %d — %s", tid, desc[:60])

    if not modified:
        return None
    return sorted(updated, key=lambda t: t["id"])


def _parse_task_list(text: str) -> list[TaskItem]:
    """LLM 출력에서 번호 목록을 파싱해 TaskItem 리스트로 반환한다."""
    tasks: list[TaskItem] = []
    for m in re.finditer(r"^\s*(\d+)\.\s*(?:\[[ x]\]\s*)?(.+)$", text, re.MULTILINE):
        tasks.append(TaskItem(
            id=int(m.group(1)),
            description=m.group(2).strip(),
            status="pending",
            result=None,
        ))
    return tasks


@lru_cache(maxsize=1)
def _get_plan_model() -> ChatOllama:
    """도구 없는 모델 — 계획 단계에서 반드시 텍스트만 출력하도록 강제한다."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,          # 계획은 결정론적으로
        num_ctx=settings.num_ctx,
        num_predict=512,           # 목록만 생성하므로 짧게 제한
    )


@lru_cache(maxsize=1)
def _get_execute_model():
    """도구 바인딩 실행 모델 — search_documents 도구를 사용해 각 작업을 처리한다."""
    base = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
    )
    return base.bind_tools(_TOOLS)


def task_agent_node(state: ChatState) -> dict:
    """
    Task Agent 노드.

    첫 호출: Plan 단계 — 도구 없이 작업 목록을 텍스트로 출력한다.
    이후 호출: Execute 단계 — 도구를 사용해 각 작업을 순서대로 실행한다.
    """
    task_list = state.get("task_list")

    if task_list is None:
        # ── Phase 1: Plan ─────────────────────────────────────────────────
        _log.debug("task_agent: Phase 1 — generating plan")
        model = _get_plan_model()
        messages = [SystemMessage(content=TASK_PLAN_PROMPT), *state["messages"]]
        response = model.invoke(messages)

        content = response.content if isinstance(response.content, str) else ""
        parsed = _parse_task_list(content)
        _log.debug("task_agent: parsed %d tasks", len(parsed))
        updates: dict = {"messages": [response]}
        if parsed:
            updates["task_list"] = parsed
            updates["task_plan_ready"] = True   # Phase 2 시작 신호 (문자열 매칭 불필요)
        return updates

    else:
        # ── Phase 2: 도구 호출(ReAct)로 각 작업 실행 ─────────────────────
        _log.debug("task_agent: Phase 2 — executing with tool calls")

        # 최근 30개 메시지만 사용 (과거 세션 누적 방지)
        all_msgs = state["messages"]
        history = all_msgs[-30:] if len(all_msgs) > 30 else all_msgs

        # 다음 pending 작업 탐색
        pending = [t for t in task_list if t.get("status", "pending") == "pending"]
        if not pending:
            return {"task_plan_ready": False}

        next_task = pending[0]
        task_id: int = next_task["id"]
        task_desc: str = next_task["description"]

        # 마지막 메시지가 ToolMessage(도구 결과)이면 검색 완료 → 그대로 호출해 최종 답변 생성
        last_msg = history[-1] if history else None
        if isinstance(last_msg, ToolMessage):
            messages = [SystemMessage(content=TASK_EXECUTE_PROMPT), *history]
        else:
            task_msg = HumanMessage(content=(
                f"▶ 작업 {task_id}: {task_desc}\n"
                "search_documents 도구를 사용해 관련 내용을 검색한 후 작업 결과를 한국어로 작성하세요."
            ))
            messages = [SystemMessage(content=TASK_EXECUTE_PROMPT), *history, task_msg]

        model = _get_execute_model()
        response = model.invoke(messages)

        # 도구 호출이 있으면 반환 — ToolNode가 실행 후 task_agent로 재진입
        if getattr(response, "tool_calls", None):
            return {"messages": [response]}

        # 최종 답변: 작업 완료 처리
        content = response.content if isinstance(response.content, str) else ""
        if not content:
            content = "관련 문서를 찾지 못했습니다."
        # 각 작업 앞 줄바꿈 — 여러 작업이 하나의 스트림으로 이어질 때 구분선 역할
        response.content = "\n\n" + _strip_intro(strip_leaked_prompt(content)).lstrip()

        updated_tasks: list = [dict(t) for t in task_list]
        for t in updated_tasks:
            if t["id"] == task_id:
                t["status"] = "done"
                t["result"] = content[:300]

        still_pending = [t for t in updated_tasks if t.get("status", "pending") == "pending"]
        return {
            "messages": [response],
            "task_list": updated_tasks,
            "task_plan_ready": bool(still_pending),
        }
