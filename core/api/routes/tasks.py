from typing import Annotated

import fastapi
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.session import get_async_session
from core.models.http import ResponseModel
from core.models.http.tasks import (
    TaskInCRUDResponse,
    TaskCreateRequestModel,
    TaskUpdateRequestModel,
)
from ..controller import tasks as tasks_controller

tasks_route = fastapi.APIRouter(prefix="/tasks", tags=["Tasks"])


@tasks_route.post(
    path="",
    name="创建任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def create(
    request_model: TaskCreateRequestModel,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_controller.create_task(
        session=session, request_model=request_model
    )
    return ResponseModel(result=result)


@tasks_route.get(
    path="",
    name="获取全部任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[list[TaskInCRUDResponse]],
)
async def get(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ResponseModel[list[TaskInCRUDResponse]]:
    result = await tasks_controller.get_tasks(session=session)
    return ResponseModel(result=result)


@tasks_route.get(
    path="/{task_id}",
    name="获取某个任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def get_by_id(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_controller.get_task_by_id(session=session, task_id=task_id)
    return ResponseModel(result=result)


@tasks_route.put(
    path="/{task_id}",
    name="更新某个任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def update(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    request_model: TaskUpdateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_controller.update_task(
        session=session, task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=result)


@tasks_route.delete(
    path="/{task_id}",
    name="删除某个任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[bool],
)
async def delete(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[bool]:
    result = await tasks_controller.delete_task_by_id(session=session, task_id=task_id)
    return ResponseModel(result=result)
