from fastapi import Header
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.connection import get_async_session


async def global_headers(
    x_trace_id: str | None = Header(
        default=None,
        alias="X-Trace-Id",
        description="用于分布式追踪的唯一 ID. 若未提供. 则 Axon 将自动生成一个 uuid.",
    ),
):
    pass


__all__ = ["get_async_session", "AsyncSession", "global_headers"]
