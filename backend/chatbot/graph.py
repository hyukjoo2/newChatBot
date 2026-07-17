from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition

from backend.chatbot.nodes.orchestrator_agent import orchestrator_agent_node, route_from_orchestrator
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
    멀티 에이전트 그래프 (orchestrator 중심).

    흐름:
        START → orchestrator_agent (의도 파악 + 계획 수립)
                 ↓ [계획 기반 결정론적 라우팅]
                 ├─ rag_agent     → rag_tools (ReAct) → rag_agent
                 │                → grade_answer (품질 평가)
                 │                    ├─ relevant     → orchestrator_agent
                 │                    ├─ not_relevant (retry<2) → rag_agent
                 │                    └─ not_relevant (retry≥2) → web_search_agent
                 ├─ web_search_agent → orchestrator_agent
                 ├─ reasoning_agent  → direct_tools (ReAct) → reasoning_agent → orchestrator_agent
                 ├─ summary_agent    → summary_tools (ReAct) → summary_agent → orchestrator_agent
                 ├─ task_agent       → task_tools (ReAct) → task_agent → orchestrator_agent
                 ├─ email_agent      → orchestrator_agent
                 └─ image_agent      → image_tools → image_agent → orchestrator_agent
                 ↓ [모든 작업 완료]
                 END
    """
    builder = StateGraph(ChatState)

    # ── 노드 등록 ──────────────────────────────────────────────────────────
    builder.add_node("orchestrator_agent", orchestrator_agent_node)
    builder.add_node("rag_agent",          rag_agent_node)
    builder.add_node("rag_tools",          rag_tools_node)       # custom: 중복 쿼리 차단
    builder.add_node("summary_agent",      summary_agent_node)
    builder.add_node("summary_tools",      ToolNode(_SUMMARY_TOOLS))
    builder.add_node("task_agent",         task_agent_node)
    builder.add_node("task_tools",         task_tools_node)       # custom: search→web 폴백
    builder.add_node("email_agent",        email_agent_node)
    builder.add_node("image_agent",        image_agent_node)
    builder.add_node("image_tools",        ToolNode(_IMAGE_TOOLS))
    builder.add_node("reasoning_agent",    reasoning_agent_node)
    builder.add_node("direct_tools",       ToolNode(_DIRECT_TOOLS))
    builder.add_node("weather_agent",      weather_agent_node)
    builder.add_node("grade_answer",       grade_answer_node)
    builder.add_node("web_search_agent",   web_search_agent_node)

    # ── 진입점 ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator_agent")

    # orchestrator → 계획의 다음 에이전트 (결정론적)
    builder.add_conditional_edges(
        "orchestrator_agent",
        route_from_orchestrator,
        {
            "rag_agent":        "rag_agent",
            "summary_agent":    "summary_agent",
            "task_agent":       "task_agent",
            "email_agent":      "email_agent",
            "image_agent":      "image_agent",
            "web_search_agent": "web_search_agent",
            "reasoning_agent":  "reasoning_agent",
            "weather_agent":    "weather_agent",
            END: END,
        },
    )

    # ── Worker 완료 → orchestrator 복귀 ─────────────────────────────────────
    builder.add_edge("email_agent", "orchestrator_agent")
    builder.add_edge("web_search_agent", "orchestrator_agent")
    builder.add_edge("weather_agent", "orchestrator_agent")

    # reasoning_agent ReAct 루프 → 최종 답변 시 orchestrator 복귀
    builder.add_conditional_edges("reasoning_agent", tools_condition, {
        "tools": "direct_tools",
        END: "orchestrator_agent",
    })
    builder.add_edge("direct_tools", "reasoning_agent")

    # rag_agent ReAct 루프 + Corrective RAG
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
            else:
                return "web_search_agent"
        return "orchestrator_agent"  # 품질 통과 → orchestrator 복귀

    builder.add_conditional_edges("grade_answer", _route_after_grade, {
        "rag_agent":        "rag_agent",
        "web_search_agent": "web_search_agent",
        "orchestrator_agent": "orchestrator_agent",
    })

    # summary_agent ReAct 루프 → orchestrator 복귀
    builder.add_conditional_edges("summary_agent", tools_condition, {
        "tools": "summary_tools",
        END: "orchestrator_agent",
    })
    builder.add_edge("summary_tools", "summary_agent")

    # task_agent: Plan → Execute 2단계 → orchestrator 복귀
    def _task_agent_condition(state: ChatState) -> str:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if last and getattr(last, "tool_calls", None):
            return "tools"
        if state.get("task_plan_ready") is True:
            return "task_agent"
        return "orchestrator_agent"  # 완료 → orchestrator 복귀

    builder.add_conditional_edges("task_agent", _task_agent_condition, {
        "tools":             "task_tools",
        "task_agent":        "task_agent",
        "orchestrator_agent": "orchestrator_agent",
    })
    builder.add_edge("task_tools", "task_agent")

    # image_agent ReAct 루프 → orchestrator 복귀
    builder.add_conditional_edges("image_agent", tools_condition, {
        "tools": "image_tools",
        END: "orchestrator_agent",
    })
    builder.add_edge("image_tools", "image_agent")

    return builder.compile(checkpointer=checkpointer)
    """
    멀티 에이전트 그래프.

    흐름:
        START → supervisor
                 ├─ rag_agent     → rag_tools (ReAct 루프) → rag_agent
                 │                → grade_answer (답변 품질 평가)
                 │                    ├─ relevant              → END
                 │                    ├─ not_relevant (retry<2) → rag_agent
                 │                    └─ not_relevant (retry≥2) → web_search_agent
                 ├─ web_search_agent → END (항상 먼저 검색 후 답변)
                 ├─ summary_agent → summary_tools (ReAct 루프) → summary_agent → END
                 ├─ task_agent    → task_tools (ReAct 루프) → task_agent → END
                 ├─ email_agent   → END
                 ├─ image_agent   → image_tools → image_agent → END
                 └─ reasoning_agent  → END
    """
    builder = StateGraph(ChatState)

    # ── 노드 등록 ──────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("rag_tools", rag_tools_node)  # custom: 중복 쿼리 차단 ToolNode
    builder.add_node("summary_agent", summary_agent_node)
    builder.add_node("summary_tools", ToolNode(_SUMMARY_TOOLS))
    builder.add_node("task_agent", task_agent_node)
    builder.add_node("task_tools", task_tools_node)   # custom: search→web 자동 폴백
    builder.add_node("email_agent", email_agent_node)
    builder.add_node("image_agent", image_agent_node)
    builder.add_node("image_tools", ToolNode(_IMAGE_TOOLS))
    builder.add_node("reasoning_agent", reasoning_agent_node)
    builder.add_node("direct_tools", ToolNode(_DIRECT_TOOLS))
    builder.add_node("grade_answer", grade_answer_node)
    builder.add_node("web_search_agent", web_search_agent_node)

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
            "direct_agent": "reasoning_agent",
            "web_search_agent": "web_search_agent",
        },
    )

    builder.add_edge("email_agent", END)

    # reasoning_agent ReAct 루프 (모를 때 web_search/get_weather 도구 호출, 답변 후 후속 판단)
    builder.add_conditional_edges("reasoning_agent", tools_condition, {
        "tools": "direct_tools",
        END: END,
    })
    builder.add_edge("direct_tools", "reasoning_agent")

    # rag_agent ReAct 루프 + Corrective RAG
    # 도구 호출이 없으면 grade_answer로 가서 답변 품질을 평가한다.
    builder.add_conditional_edges("rag_agent", tools_condition, {
        "tools": "rag_tools",
        END: "grade_answer",
    })
    builder.add_edge("rag_tools", "rag_agent")

    def _route_after_grade(state: ChatState) -> str:
        """grade_answer 결과에 따라 종료 / rag 재시도 / 웹 검색 폴백을 결정한다."""
        if state.get("answer_grade") == "not_relevant":
            retry = state.get("rag_retry_count", 0)
            if retry < MAX_RETRIES:   # < (이전 <= 로 무한루프 버그)
                _log.info("Answer graded not_relevant — retrying rag_agent (retry=%d)", retry)
                return "rag_agent"
            else:
                _log.info("MAX_RETRIES reached — falling back to web_search_agent")
                return "web_search_agent"
        return END

    builder.add_conditional_edges("grade_answer", _route_after_grade, {
        "rag_agent": "rag_agent",
        "web_search_agent": "web_search_agent",
        END: END,
    })

    # web_search_agent: 항상 먼저 검색 후 답변 (ReAct 루프 불필요)
    builder.add_edge("web_search_agent", END)

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

