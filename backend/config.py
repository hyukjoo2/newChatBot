from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} 환경변수는 숫자여야 합니다. 현재 값: {raw}") from e


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} 환경변수는 정수여야 합니다. 현재 값: {raw}") from e


@dataclass(frozen=True)
class Settings:
    # Ollama / LLM
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    ollama_vision_model: str
    temperature: float
    num_ctx: int
    num_predict: int

    # PostgreSQL
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    database_url: str

    # File storage
    upload_dir: str
    extracted_dir: str

    # RAG defaults
    default_chunk_size: int
    default_chunk_overlap: int
    default_retrieval_top_k: int

    # Naver Search API
    naver_client_id: str
    naver_client_secret: str

    @property
    def dsn(self) -> str:
        """psycopg DSN 문자열."""
        return (
            f"host={self.postgres_host} "
            f"port={self.postgres_port} "
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password}"
        )


def load_settings() -> Settings:
    return Settings(
        ollama_base_url=_get_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_get_str("OLLAMA_MODEL", "gemma3:4b"),
        ollama_embedding_model=_get_str("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_vision_model=_get_str("OLLAMA_VISION_MODEL", "moondream"),
        temperature=_get_float("LLM_TEMPERATURE", 0.2),
        num_ctx=_get_int("LLM_NUM_CTX", 8192),
        num_predict=_get_int("LLM_NUM_PREDICT", 1024),
        postgres_host=_get_str("POSTGRES_HOST", "localhost"),
        postgres_port=_get_int("POSTGRES_PORT", 5432),
        postgres_db=_get_str("POSTGRES_DB", "local_assistant"),
        postgres_user=_get_str("POSTGRES_USER", "assistant"),
        postgres_password=_get_str("POSTGRES_PASSWORD", "assistant_password"),
        database_url=_get_str(
            "DATABASE_URL",
            "postgresql+psycopg://assistant:assistant_password@localhost:5432/local_assistant",
        ),
        upload_dir=_get_str("UPLOAD_DIR", "data/uploads"),
        extracted_dir=_get_str("EXTRACTED_DIR", "data/extracted"),
        default_chunk_size=_get_int("DEFAULT_CHUNK_SIZE", 300),
        default_chunk_overlap=_get_int("DEFAULT_CHUNK_OVERLAP", 80),
        default_retrieval_top_k=_get_int("DEFAULT_RETRIEVAL_TOP_K", 5),
        naver_client_id=_get_str("NAVER_CLIENT_ID", ""),
        naver_client_secret=_get_str("NAVER_CLIENT_SECRET", ""),
    )


settings = load_settings()
