from typing import Annotated

import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
    PageinationRequest,
    PageinationResponse,
    TaskInCRUDResponse,
    TaskChatCreateRequestModel,
    TaskChatInCRUDResponse,
)
from core.services import tasks_chat as tasks_chat_services
from core.api.dependencies import get_async_session, AsyncSession


tasks_chat_route = fastapi.APIRouter(prefix="/{task_id}/chat", tags=["Tasks-chat"])


@tasks_chat_route.get(
    path="",
    name="获取聊天记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PageinationResponse[TaskChatInCRUDResponse],
)
async def get(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    task_id: int = fastapi.Path(description="任务 ID"),
    pageination: PageinationRequest = Depends(PageinationRequest),
) -> PageinationResponse[TaskChatInCRUDResponse]:
    result = await tasks_chat_services.get_chats(
        task_id=task_id, session=session, pageination=pageination
    )
    return result


@tasks_chat_route.post(
    path="",
    name="插入聊天记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def insert_task_chat(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    request_model: TaskChatCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    result = await tasks_chat_services.insert_task_chat(
        session=session, task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=result)
