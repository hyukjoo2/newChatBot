from __future__ import annotations

import logging
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from backend.config import settings

_log = logging.getLogger(__name__)

# Ollama 내부 llama-server에 한 번에 보낼 최대 청크 수.
# 너무 크면 단일 HTTP 요청이 수백 KB가 돼서 EOF/400 에러 발생.
_EMBED_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def _get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_embeddings(text: str) -> list[float]:
    """텍스트를 임베딩 벡터로 변환한다."""
    model = _get_embedding_model()
    return model.embed_query(text)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 _EMBED_BATCH_SIZE 단위로 나눠 임베딩한다.

    대용량 문서(수백~수천 청크)를 한 번에 보내면 Ollama 내부 llama-server가
    EOF/400 에러를 반환하므로 배치로 분할해 전송한다.
    """
    model = _get_embedding_model()
    results: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        _log.debug("embedding batch %d-%d / %d", start + 1, start + len(batch), total)
        results.extend(model.embed_documents(batch))
    return results
