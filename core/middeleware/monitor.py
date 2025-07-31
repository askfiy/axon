import time
import json
import logging
import typing
from typing import override

from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
    _StreamingResponse,  # pyright: ignore[reportPrivateUsage]  # noqa: F401
)

logger = logging.getLogger("monitor-middleware")


class GlobalMonitorMiddleware(BaseHTTPMiddleware):
    FILTER_API_PATH = ["/docs", "/openapi.json"]

    def get_request_info(self, request: Request) -> str:
        method = request.method
        path = request.url.path

        query = request.url.query
        http_version = request.scope.get("http_version", "unknown")

        full_path = path

        if query:
            full_path += f"?{query}"

        return f"{method} {full_path} HTTP/{http_version}"

    async def get_response_body(self, response: _StreamingResponse):
        response_body_chunks: list[bytes] = []

        async for chunk in response.body_iterator:
            response_body_chunks.append(typing.cast("bytes", chunk))

        return b"".join(response_body_chunks)

    async def get_request_log(self, request: Request) -> str:
        logger_info = ""

        request_body = await request.body()

        if request_body:
            try:
                log_data = json.loads(request_body)
                logger_info = f", JSON: {log_data}"
            except json.JSONDecodeError:
                logger_info = (
                    f", (Non-JSON): {request_body.decode(errors='ignore')[:500]}..."
                )

        return logger_info

    async def get_response_log(self, response_body: bytes) -> str:
        logger_info = ""

        if response_body:
            try:
                log_data = json.loads(response_body)
                logger_info = f", JSON: {log_data}"
            except json.JSONDecodeError:
                logger_info = (
                    f", (Non-JSON): {response_body.decode(errors='ignore')[:500]}..."
                )

        return logger_info

    @override
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self.FILTER_API_PATH:
            return await call_next(request)

        request_info = self.get_request_info(request)

        await self.get_request_log(request)

        start_time = time.perf_counter()
        request_loger = await self.get_request_log(request)

        logger.info(f"Request:  '{request_info}'{request_loger}")

        response: _StreamingResponse = await call_next(request)

        response_body = await self.get_response_body(response)

        process_time = (time.perf_counter() - start_time) * 1000
        response_loger = await self.get_response_log(response_body)

        logger.info(
            f"Response: '{request_info} {response.status_code}' ({process_time:.2f}ms){response_loger}"
        )

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
