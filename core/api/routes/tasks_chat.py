import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
    Paginator,
    PaginationRequest,
    PaginationResponse,
    TaskChatInCRUDResponse,
)
from core.models.services import TaskChatCreateRequestModel
from core.services import tasks_chat as tasks_chat_services


tasks_chat_route = fastapi.APIRouter(prefix="/{task_id}/chat", tags=["Tasks-chat"])


@tasks_chat_route.get(
    path="",
    name="获取聊天记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PaginationResponse,
)
async def get(
    task_id: int = fastapi.Path(description="任务 ID"),
    request: PaginationRequest = Depends(PaginationRequest),
) -> PaginationResponse:
    paginator = Paginator(
        request=request,
        serializer_cls=TaskChatInCRUDResponse,
    )
    paginator = await tasks_chat_services.upget_tasks_chat_pagination(
        task_id=task_id, paginator=paginator
    )
    return paginator.response


@tasks_chat_route.post(
    path="",
    name="插入聊天记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskChatInCRUDResponse],
)
async def insert_task_chat(
    request_model: TaskChatCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskChatInCRUDResponse]:
    result = await tasks_chat_services.create_chat(
        task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=TaskChatInCRUDResponse.model_validate(result))
