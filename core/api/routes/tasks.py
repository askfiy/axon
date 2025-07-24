from typing import Annotated

import fastapi
from fastapi import Depends

from core.services import tasks as tasks_services
from core.models.http import (
    ResponseModel,
    TaskInCRUDResponse,
    TaskCreateRequestModel,
    TaskUpdateRequestModel,
    TaskChatCreateRequestModel,
)
from core.api.dependencies import get_async_session, AsyncSession

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
    result = await tasks_services.create_task(
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
    result = await tasks_services.get_tasks(session=session)
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
    result = await tasks_services.get_task_by_id(session=session, task_id=task_id)
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
    result = await tasks_services.update_task(
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
    result = await tasks_services.delete_task_by_id(session=session, task_id=task_id)
    return ResponseModel(result=result)


# @tasks_route.get(
#     path="/{task_id}/chats",
#     name="获取任务的聊天历史记录",
#     status_code=fastapi.status.HTTP_200_OK,
#     responses=ResponseModel,
# )
# async def get_chats(
#     session: Annotated[AsyncSession, Depends(get_async_session)],
#     task_id: int = fastapi.Path(description="任务 ID"),
# ) -> Response:
#     pass


@tasks_route.post(
    path="/{task_id}/control/chat",
    name="插入聊天历史记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def insert_task_chat(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    request_model: TaskChatCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_services.insert_task_chat(
        session=session, task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=result)
