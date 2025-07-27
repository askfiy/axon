from contextlib import asynccontextmanager

import uvicorn
import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from core.api.routes import api_router
from core.models.http import ResponseModel


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    yield


app = fastapi.FastAPI(title="Axon", lifespan=lifespan)


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    status_code = fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR
    message = str(exc)

    if isinstance(exc, fastapi.HTTPException):
        status_code = exc.status_code
        message = exc.detail

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
