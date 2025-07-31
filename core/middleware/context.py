"""
ASGI 层面的中间件. 旨在提供类似于 Flask 的 g 对象.
"""

from typing import Any, override
from contextvars import ContextVar, copy_context
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send


class GlobalContextException(Exception):
    pass


@dataclass
class Globals:
    _context_data = ContextVar("context_data", default={})

    def clear(self) -> None:
        self._context_data.set({})

    def get(self, name: str, default: Any = None) -> Any:
        return self._context_data.get().get(name, default)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._context_data.get()[name]
        except KeyError:
            raise GlobalContextException(f"'{name}' is not found from global context.")

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        self._context_data.get()[name] = value


class GlobalContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        g.clear()

        ctx = copy_context()
        await ctx.run(self.app, scope, receive, send)


g = Globals()
