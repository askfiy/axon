import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
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
    response_model=PaginationResponse[TaskChatInCRUDResponse],
)
async def get(
    task_id: int = fastapi.Path(description="任务 ID"),
    pagination: PaginationRequest = Depends(PaginationRequest),
) -> PaginationResponse[TaskChatInCRUDResponse]:
    result = await tasks_chat_services.get_chats(
        task_id=task_id, pagination=pagination
    )
    return PaginationResponse(
        **result.model_dump(
            exclude={"db_objects"},
        ),
        result=[
            TaskChatInCRUDResponse.model_validate(chat) for chat in result.db_objects
        ],
    )


@tasks_chat_route.post(
    path="",
    name="插入聊天记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskChatCreateRequestModel],
)
async def insert_task_chat(
    request_model: TaskChatCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskChatInCRUDResponse]:
    result = await tasks_chat_services.create_chat(
        task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=TaskChatInCRUDResponse.model_validate(result))
