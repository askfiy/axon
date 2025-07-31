from datetime import datetime
import sys
import logging
from typing import override

from colorlog import ColoredFormatter

from core.utils.context import g


class Formatter(ColoredFormatter):
    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        super_time = super().formatTime(record, datefmt)
        return f"{super_time}.{int(record.msecs):03d}"

    @override
    def format(self, record: logging.LogRecord) -> str:
        record.space = " "
        record.trace_id = g.get("trace_id", "X-Trace-ID")

        record.timestamp = self.formatTime(record, self.datefmt)
        return super().format(record)


formatter = (
    "%(log_color)s%(levelname)s%(reset)s:"
    "%(white)s%(space)-5s%(reset)s"
    "[%(light_green)s%(timestamp)s%(reset)s] "
    "[%(light_blue)s%(name)s%(reset)s] - "
    "[%(light_yellow)s%(funcName)s:%(lineno)s]%(reset)s - "
    "[%(cyan)s%(trace_id)s%(reset)s] "
    "%(bold_white)s%(message)s%(reset)s"
)

console_formatter = Formatter(
    formatter,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    datefmt="%Y-%m-%d %H:%M:%S",
    secondary_log_colors={},
    style="%",
)


def setup_logging(level: int | str = logging.INFO) -> None:
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    root_logger.addHandler(console_handler)

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_errors_logger = logging.getLogger("uvicorn.error")

    uvicorn_errors_logger.handlers.clear()
    uvicorn_access_logger.handlers.clear()

    uvicorn_errors_logger.propagate = True
    uvicorn_access_logger.propagate = False
