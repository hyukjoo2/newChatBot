from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from backend.config import settings


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
    """여러 텍스트를 일괄 임베딩한다."""
    model = _get_embedding_model()
    return model.embed_documents(texts)
