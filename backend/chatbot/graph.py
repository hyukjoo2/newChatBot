from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition

from backend.chatbot.nodes.orchestrator_agent import (
    orchestrator_planner_node, route_from_planner,
    orchestrator_evaluator_node, route_from_evaluator,
)
from backend.chatbot.nodes.rag_agent import rag_agent_node, rag_tools_node, _TOOLS as _RAG_TOOLS
from backend.chatbot.nodes.image_agent import image_agent_node, _IMAGE_TOOLS
from backend.chatbot.nodes.reasoning_agent import reasoning_agent_node, _TOOLS as _DIRECT_TOOLS
from backend.chatbot.nodes.weather_agent import weather_agent_node
from backend.chatbot.nodes.email_agent import email_agent_node
from backend.chatbot.nodes.summary_agent import summary_agent_node, _TOOLS as _SUMMARY_TOOLS
from backend.chatbot.nodes.task_agent import task_agent_node, task_tools_node, _TOOLS as _TASK_TOOLS
from backend.chatbot.nodes.grade_answer import grade_answer_node, MAX_RETRIES
from backend.chatbot.nodes.web_search_agent import web_search_agent_node
from backend.chatbot.state import ChatState
from backend.config import settings

_log = logging.getLogger(__name__)


def build_graph(checkpointer=None) -> any:
    """
    멀티 에이전트 그래프 (planner/evaluator 분리 구조).

    흐름:
        START → orchestrator_planner (질문 분해 + 계획 수립, 1회)
                 ↓ [첫 작업 에이전트로 라우팅]
                 ├─ rag_agent        → rag_tools (ReAct) → rag_agent
                 │                   → grade_answer → orchestrator_evaluator
                 ├─ web_search_agent → orchestrator_evaluator
                 ├─ reasoning_agent  → direct_tools ↺ → orchestrator_evaluator
                 ├─ summary_agent    → summary_tools ↺ → orchestrator_evaluator
                 ├─ task_agent       → task_tools ↺ → orchestrator_evaluator
                 ├─ email_agent      → orchestrator_evaluator
                 ├─ image_agent      → image_tools ↺ → orchestrator_evaluator
                 └─ weather_agent    → orchestrator_evaluator
                 ↓ orchestrator_evaluator (결과 평가, 작업마다 반복)
                 → 다음 작업 에이전트 or END
    """
    builder = StateGraph(ChatState)
    _WORKERS = {
        "rag_agent", "summary_agent", "task_agent", "email_agent",
        "image_agent", "reasoning_agent", "web_search_agent", "weather_agent",
    }

    # ── 노드 등록 ──────────────────────────────────────────────────────────
    builder.add_node("orchestrator_planner",   orchestrator_planner_node)
    builder.add_node("orchestrator_evaluator", orchestrator_evaluator_node)
    builder.add_node("rag_agent",          rag_agent_node)
    builder.add_node("rag_tools",          rag_tools_node)
    builder.add_node("summary_agent",      summary_agent_node)
    builder.add_node("summary_tools",      ToolNode(_SUMMARY_TOOLS))
    builder.add_node("task_agent",         task_agent_node)
    builder.add_node("task_tools",         task_tools_node)
    builder.add_node("email_agent",        email_agent_node)
    builder.add_node("image_agent",        image_agent_node)
    builder.add_node("image_tools",        ToolNode(_IMAGE_TOOLS))
    builder.add_node("reasoning_agent",    reasoning_agent_node)
    builder.add_node("direct_tools",       ToolNode(_DIRECT_TOOLS))
    builder.add_node("weather_agent",      weather_agent_node)
    builder.add_node("grade_answer",       grade_answer_node)
    builder.add_node("web_search_agent",   web_search_agent_node)

    _WORKER_EDGES = {w: w for w in _WORKERS} | {END: END}

    # ── 진입점 ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator_planner")

    # planner → 첫 번째 작업 에이전트
    builder.add_conditional_edges("orchestrator_planner", route_from_planner, _WORKER_EDGES)

    # evaluator → 다음 작업 에이전트 or END
    builder.add_conditional_edges("orchestrator_evaluator", route_from_evaluator, _WORKER_EDGES)

    # ── Worker 완료 → evaluator 복귀 ─────────────────────────────────────
    builder.add_edge("email_agent",      "orchestrator_evaluator")
    builder.add_edge("web_search_agent", "orchestrator_evaluator")
    builder.add_edge("weather_agent",    "orchestrator_evaluator")

    builder.add_conditional_edges("reasoning_agent", tools_condition, {
        "tools": "direct_tools",
        END: "orchestrator_evaluator",
    })
    builder.add_edge("direct_tools", "reasoning_agent")

    builder.add_conditional_edges("rag_agent", tools_condition, {
        "tools": "rag_tools",
        END: "grade_answer",
    })
    builder.add_edge("rag_tools", "rag_agent")

    def _route_after_grade(state: ChatState) -> str:
        if state.get("answer_grade") == "not_relevant":
            retry = state.get("rag_retry_count", 0)
            if retry < MAX_RETRIES:
                return "rag_agent"
            # 재시도 소진 → 평가자가 Yes/No 버튼 제안 (웹 직접 실행 안 함)
            return "orchestrator_evaluator"
        return "orchestrator_evaluator"

    builder.add_conditional_edges("grade_answer", _route_after_grade, {
        "rag_agent":               "rag_agent",
        "orchestrator_evaluator":  "orchestrator_evaluator",
    })

    builder.add_conditional_edges("summary_agent", tools_condition, {
        "tools": "summary_tools",
        END: "orchestrator_evaluator",
    })
    builder.add_edge("summary_tools", "summary_agent")

    def _task_agent_condition(state: ChatState) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if last and getattr(last, "tool_calls", None):
            return "tools"
        if state.get("task_plan_ready") is True:
            return "task_agent"
        return "orchestrator_evaluator"

    builder.add_conditional_edges("task_agent", _task_agent_condition, {
        "tools":                  "task_tools",
        "task_agent":             "task_agent",
        "orchestrator_evaluator": "orchestrator_evaluator",
    })
    builder.add_edge("task_tools", "task_agent")

    # image_agent ReAct 루프 → evaluator 복귀
    builder.add_conditional_edges("image_agent", tools_condition, {
        "tools": "image_tools",
        END: "orchestrator_evaluator",
    })
    builder.add_edge("image_tools", "image_agent")

    return builder.compile(checkpointer=checkpointer)



# ── 앱 시작 시 한 번만 초기화되는 그래프 싱글턴 ────────────────────────────
_graph = None


def get_graph():
    """앱 수명 동안 그래프를 한 번만 생성하고 재사용한다."""
    global _graph
    if _graph is not None:
        return _graph

    try:
        import psycopg
        conn = psycopg.connect(settings.dsn, autocommit=True)
        saver = PostgresSaver(conn)
        saver.setup()
        _graph = build_graph(checkpointer=saver)
        _log.info("Graph initialized with PostgresSaver")
    except Exception:
        _log.exception(
            "Failed to initialize PostgresSaver — falling back to InMemorySaver. "
            "Conversation history will NOT be persisted across restarts."
        )
        from langgraph.checkpoint.memory import InMemorySaver
        _graph = build_graph(checkpointer=InMemorySaver())

    return _graph

