import logging
import functools
from typing import TypeVar, ParamSpec
from collections.abc import Awaitable, Callable

from pyinstrument import Profiler

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger()


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
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 PyInstrument Profile Report for Endpoint: '{func.__name__}'")
        logger.info("=" * 80)
        logger.info(report_text)
        logger.info("=" * 80 + "\n")

        return response

    return wrapper
