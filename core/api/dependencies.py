from fastapi import Header

from core.database.connection import (
    get_async_session,
    get_async_tx_session,
    AsyncSession,
    AsyncTxSession,
)


async def global_headers(
    x_trace_id: str | None = Header(
        default=None,
        alias="X-Trace-Id",
        description="用于分布式追踪的唯一 ID. 若未提供. 则 Taxonsk 将自动生成一个 uuid.",
    ),
):
    pass


__all__ = [
    "get_async_session",
    "get_async_tx_session",
    "AsyncSession",
    "AsyncTxSession",
    "global_headers",
]
