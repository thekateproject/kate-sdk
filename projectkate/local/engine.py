"""Lazy SQLite engine at ~/.kate/evals.db."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from projectkate.local.models import SDKBase

_init_lock = asyncio.Lock()
_engine = None
_session_factory = None


def _db_path() -> Path:
    kate_dir = Path.home() / ".kate"
    kate_dir.mkdir(parents=True, exist_ok=True)
    return kate_dir / "evals.db"


async def get_engine():
    global _engine
    async with _init_lock:
        if _engine is None:
            url = f"sqlite+aiosqlite:///{_db_path()}"
            _engine = create_async_engine(url, echo=False)
            async with _engine.begin() as conn:
                await conn.run_sync(SDKBase.metadata.create_all)
    return _engine


async def get_session() -> AsyncSession:
    global _session_factory
    engine = await get_engine()
    async with _init_lock:
        if _session_factory is None:
            _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory()
