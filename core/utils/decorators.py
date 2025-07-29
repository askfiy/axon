import functools
from typing import Callable, TypeVar, ParamSpec
from collections.abc import Awaitable
from pyinstrument import Profiler

from sqlalchemy.ext.asyncio import AsyncSession

P = ParamSpec("P")
R = TypeVar("R")


def transactional(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """
    安全的自动提交回滚事务。
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        session: AsyncSession | None = kwargs.get("session")
        if not isinstance(session, AsyncSession):
            for arg in args:
                if isinstance(arg, AsyncSession):
                    session = arg
                    break

        if not session:
            raise TypeError(
                "Decorated function must have an 'AsyncSession' instance as an argument."
            )

        try:
            result = await func(*args, **kwargs)
            await session.commit()
            return result
        except Exception as exc:
            await session.rollback()
            raise exc

    return wrapper


def profiled(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """
    打印性能报告到控制台。
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        profiler = Profiler()

        profiler.start()

        response = await func(*args, **kwargs)

        profiler.stop()

        report_text = profiler.output_text(unicode=True, color=True)
        print("\n" + "=" * 80)
        print(f"📊 PyInstrument Profile Report for Endpoint: '{func.__name__}'")
        print("=" * 80)
        print(report_text)
        print("=" * 80 + "\n")

        return response

    return wrapper
