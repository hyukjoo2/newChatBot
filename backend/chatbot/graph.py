from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition

from backend.chatbot.nodes.supervisor import supervisor_node, _AGENT_ALIASES
from backend.chatbot.nodes.rag_agent import rag_agent_node, _TOOLS as _RAG_TOOLS
from backend.chatbot.nodes.image_agent import image_agent_node, _IMAGE_TOOLS
from backend.chatbot.nodes.direct_agent import direct_agent_node
from backend.chatbot.nodes.email_agent import email_agent_node
from backend.chatbot.nodes.summary_agent import summary_agent_node, _TOOLS as _SUMMARY_TOOLS
from backend.chatbot.nodes.task_agent import task_agent_node, _TOOLS as _TASK_TOOLS
from backend.chatbot.state import ChatState
from backend.config import settings

_log = logging.getLogger(__name__)


def _route_from_supervisor(state: ChatState) -> str:
    """supervisor가 설정한 next 값을 정규화·검증하고, 알 수 없는 값이면 direct_agent로 폴백한다."""
    raw = state.get("next", "direct_agent") or "direct_agent"
    normalized = _AGENT_ALIASES.get(raw.strip().lower())
    if not normalized:
        _log.warning("Unknown next agent '%s', falling back to direct_agent", raw)
        return "direct_agent"
    return normalized


def build_graph(checkpointer=None) -> any:
    """
    멀티 에이전트 그래프.

    흐름:
        START → supervisor
                 ├─ rag_agent     → rag_tools (ReAct 루프) → rag_agent → END
                 ├─ summary_agent → summary_tools (ReAct 루프) → summary_agent → END
                 ├─ task_agent    → task_tools (ReAct 루프) → task_agent → END
                 ├─ email_agent   → END
                 ├─ image_agent   → image_tools → image_agent → END
                 └─ direct_agent  → END
    """
    builder = StateGraph(ChatState)

    # ── 노드 등록 ──────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("rag_tools", ToolNode(_RAG_TOOLS))
    builder.add_node("summary_agent", summary_agent_node)
    builder.add_node("summary_tools", ToolNode(_SUMMARY_TOOLS))
    builder.add_node("task_agent", task_agent_node)
    builder.add_node("task_tools", ToolNode(_TASK_TOOLS))
    builder.add_node("email_agent", email_agent_node)
    builder.add_node("image_agent", image_agent_node)
    builder.add_node("image_tools", ToolNode(_IMAGE_TOOLS))
    builder.add_node("direct_agent", direct_agent_node)

    # ── 엣지 ────────────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "summary_agent": "summary_agent",
            "task_agent": "task_agent",
            "email_agent": "email_agent",
            "image_agent": "image_agent",
            "direct_agent": "direct_agent",
        },
    )

    builder.add_edge("direct_agent", END)
    builder.add_edge("email_agent", END)

    # rag_agent ReAct 루프
    builder.add_conditional_edges("rag_agent", tools_condition, {
        "tools": "rag_tools",
        END: END,
    })
    builder.add_edge("rag_tools", "rag_agent")

    # summary_agent ReAct 루프
    builder.add_conditional_edges("summary_agent", tools_condition, {
        "tools": "summary_tools",
        END: END,
    })
    builder.add_edge("summary_tools", "summary_agent")

    # task_agent: Plan → Execute 2단계 라우팅
    # Phase 1 후 task_list가 세팅되면 다시 task_agent(Execute)로, 도구 호출이면 task_tools로
    def _task_agent_condition(state: ChatState) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if last and getattr(last, "tool_calls", None):
            return "tools"
        if state.get("task_plan_ready") is True:
            return "task_agent"
        return END

    builder.add_conditional_edges("task_agent", _task_agent_condition, {
        "tools": "task_tools",
        "task_agent": "task_agent",
        END: END,
    })
    builder.add_edge("task_tools", "task_agent")

    # image_agent ReAct 루프
    builder.add_conditional_edges("image_agent", tools_condition, {
        "tools": "image_tools",
        END: END,
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

