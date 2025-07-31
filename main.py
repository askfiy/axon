import uuid
import logging
from contextlib import asynccontextmanager
from collections.abc import Awaitable, Callable

import uvicorn
import fastapi
from fastapi import Request, Response, Depends
from fastapi.responses import JSONResponse

from core.context import g
from core.api.routes import api_router
from core.utils.logger import setup_logging
from core.api.dependencies import global_headers
from core.scheduler import open_scheduler, stop_scheduler
from core.models.http import ResponseModel
from core.middleware import GlobalContextMiddleware, GlobalMonitorMiddleware
from core.exceptions import (
    ServiceException,
    ServiceNotFoundException,
    ServiceMissMessageException,
)

logger = logging.getLogger("Axon")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    setup_logging()

    # 开始任务调度
    await open_scheduler()
    yield
    # 结束任务调度
    await stop_scheduler()


app = fastapi.FastAPI(
    title="Axon", lifespan=lifespan, dependencies=[Depends(global_headers)]
)

app.add_middleware(GlobalContextMiddleware)
app.add_middleware(GlobalMonitorMiddleware)


@app.middleware("http")
async def trace(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    g.trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Trace-Id"] = g.trace_id
    return response


@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    status_code = fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, ServiceNotFoundException):
        status_code = fastapi.status.HTTP_404_NOT_FOUND
    elif isinstance(exc, ServiceMissMessageException):
        status_code = fastapi.status.HTTP_400_BAD_REQUEST

    raise fastapi.HTTPException(status_code=status_code, detail=str(exc))


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    status_code = fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR
    message = str(exc)

    return JSONResponse(
        status_code=status_code,
        content=ResponseModel(
            code=status_code,
            message=message,
            is_failed=True,
            result=None,
        ).model_dump(by_alias=True),
    )


@app.get(
    path="/heart",
    name="心跳检测",
    status_code=fastapi.status.HTTP_200_OK,
)
async def heart():
    return {"success": True}


app.include_router(api_router, prefix="/api/v1")


def main():
    uvicorn.run(app="main:app", host="0.0.0.0", port=7699)


if __name__ == "__main__":
    main()
