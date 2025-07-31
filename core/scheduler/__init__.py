import asyncio

from .dispatch import Dispatch


async def open_scheduler():
    await Dispatch.forever()


async def stop_scheduler():
    await Dispatch.shutdown()


__all__ = ["open_scheduler", "stop_scheduler"]
