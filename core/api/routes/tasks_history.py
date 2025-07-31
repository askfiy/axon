from typing import Annotated

import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
    PageinationRequest,
    PageinationResponse,
    TaskInCRUDResponse,
    TaskHistoryInCRUDResponse,
    TaskHistoryCreateRequestModel,
)
from core.services import tasks_history as tasks_history_services
from core.api.dependencies import (
    get_async_session,
    get_async_tx_session,
    AsyncSession,
    AsyncTxSession,
)


tasks_history_route = fastapi.APIRouter(
    prefix="/{task_id}/history", tags=["Tasks-history"]
)


@tasks_history_route.get(
    path="",
    name="获取执行记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PageinationResponse[TaskHistoryInCRUDResponse],
)
async def get(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    task_id: int = fastapi.Path(description="任务 ID"),
    pageination: PageinationRequest = Depends(PageinationRequest),
) -> PageinationResponse[TaskHistoryInCRUDResponse]:
    result = await tasks_history_services.get_histories(
        task_id=task_id, session=session, pageination=pageination
    )
    return result


@tasks_history_route.post(
    path="",
    name="插入执行记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def insert_task_history(
    session: Annotated[AsyncTxSession, Depends(get_async_tx_session)],
    request_model: TaskHistoryCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_history_services.insert_task_history(
        session=session, task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=result)
