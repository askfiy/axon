from typing import Annotated

import fastapi
from fastapi import Depends

from core.services import tasks as tasks_services
from core.models.http import (
    ResponseModel,
    PageinationRequest,
    PageinationResponse,
    TaskInCRUDResponse,
    TaskCreateRequestModel,
    TaskUpdateRequestModel,
)
from core.api.dependencies import (
    AsyncSession,
    AsyncTxSession,
    get_async_session,
    get_async_tx_session,
)

tasks_route = fastapi.APIRouter(prefix="/tasks", tags=["Tasks"])


@tasks_route.post(
    path="",
    name="创建任务",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def create(
    request_model: TaskCreateRequestModel,
    session: Annotated[AsyncTxSession, Depends(get_async_tx_session)],
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_services.create_task(
        session=session, request_model=request_model
    )
    return ResponseModel(result=result)


@tasks_route.get(
    path="",
    name="获取全部任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PageinationResponse[TaskInCRUDResponse],
)
async def get(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    pageination: PageinationRequest = Depends(PageinationRequest),
) -> PageinationResponse[TaskInCRUDResponse]:
    result = await tasks_services.get_tasks(session=session, pageination=pageination)
    return result


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
    session: Annotated[AsyncTxSession, Depends(get_async_tx_session)],
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
    session: Annotated[AsyncTxSession, Depends(get_async_tx_session)],
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[bool]:
    result = await tasks_services.delete_task_by_id(session=session, task_id=task_id)
    return ResponseModel(result=result)
