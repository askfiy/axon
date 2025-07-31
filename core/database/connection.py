from typing import TypeAlias
from contextlib import asynccontextmanager

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from core.config import env_helper

# ----- redis

pool: redis.ConnectionPool = redis.ConnectionPool.from_url(  # pyright: ignore[reportUnknownMemberType]
    url=env_helper.ASYNC_REDIS_URL, decode_responses=True
)


def get_redis_client() -> redis.Redis:
    return redis.Redis(connection_pool=pool)


# ----- sqlalchemy

engine = create_async_engine(
    env_helper.ASYNC_DB_URL,
    # echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,  # 明确指定使用 AsyncSession
)


async def get_async_session():
    async with AsyncSessionLocal(bind=engine) as session:
        yield session


async def get_async_tx_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            raise exc


get_async_session_direct = asynccontextmanager(get_async_session)
get_async_tx_session_direct = asynccontextmanager(get_async_tx_session)


AsyncTxSession: TypeAlias = AsyncSession

__all__ = [
    "engine",
    "get_async_session",
    "get_async_tx_session",
    "get_async_session_direct",
    "get_async_tx_session_direct",
    "AsyncTxSession",
    "AsyncSession",
    "get_redis_client",
]
