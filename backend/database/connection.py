from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg_pool import ConnectionPool

from backend.config import settings


_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """애플리케이션 전역 커넥션 풀을 반환한다."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.dsn,
            min_size=1,
            max_size=10,
            open=True,
        )
    return _pool


def close_pool() -> None:
    """커넥션 풀을 닫는다."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """커넥션 풀에서 커넥션을 빌려 반환한다."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn
