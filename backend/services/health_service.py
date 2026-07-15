"""
애플리케이션 헬스체크 서비스.

앱 시작 시 또는 사이드바 진단 버튼에서 PostgresSaver, Ollama LLM, 임베딩 모델 연결을 점검한다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)


@dataclass
class ComponentStatus:
    name: str
    ok: bool
    latency_ms: Optional[float] = None
    detail: str = ""


@dataclass
class HealthReport:
    components: list[ComponentStatus] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(c.ok for c in self.components)

    def summary(self) -> str:
        lines = []
        for c in self.components:
            icon = "✅" if c.ok else "❌"
            lat = f" ({c.latency_ms:.0f}ms)" if c.latency_ms is not None else ""
            detail = f" — {c.detail}" if c.detail else ""
            lines.append(f"{icon} {c.name}{lat}{detail}")
        return "\n".join(lines)


def _check_postgres(dsn: str) -> ComponentStatus:
    name = "PostgreSQL (checkpointer)"
    t0 = time.perf_counter()
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(dsn) as saver:
            saver.setup()
        latency = (time.perf_counter() - t0) * 1000
        return ComponentStatus(name=name, ok=True, latency_ms=latency, detail="connected")
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        _log.warning("Health check failed — %s: %s", name, e)
        return ComponentStatus(name=name, ok=False, latency_ms=latency, detail=str(e)[:120])


def _check_ollama_llm(base_url: str, model: str) -> ComponentStatus:
    name = f"Ollama LLM ({model})"
    t0 = time.perf_counter()
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, base_url=base_url, num_predict=1, temperature=0.0)
        llm.invoke("ping")
        latency = (time.perf_counter() - t0) * 1000
        return ComponentStatus(name=name, ok=True, latency_ms=latency, detail="responding")
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        _log.warning("Health check failed — %s: %s", name, e)
        return ComponentStatus(name=name, ok=False, latency_ms=latency, detail=str(e)[:120])


def _check_embeddings(base_url: str, embed_model: str) -> ComponentStatus:
    name = f"Embeddings ({embed_model})"
    t0 = time.perf_counter()
    try:
        from backend.rag.embeddings import get_embeddings
        vec = get_embeddings("health check")
        latency = (time.perf_counter() - t0) * 1000
        if not vec or len(vec) == 0:
            return ComponentStatus(name=name, ok=False, latency_ms=latency, detail="empty vector returned")
        return ComponentStatus(name=name, ok=True, latency_ms=latency, detail=f"dim={len(vec)}")
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        _log.warning("Health check failed — %s: %s", name, e)
        return ComponentStatus(name=name, ok=False, latency_ms=latency, detail=str(e)[:120])


def run_health_check() -> HealthReport:
    """PostgreSQL, Ollama LLM, 임베딩 모델 연결 상태를 점검하고 HealthReport를 반환한다."""
    from backend.config import settings

    report = HealthReport()
    report.components.append(_check_postgres(settings.dsn))
    report.components.append(_check_ollama_llm(settings.ollama_base_url, settings.ollama_model))
    report.components.append(_check_embeddings(settings.ollama_base_url, settings.ollama_embedding_model))
    return report
