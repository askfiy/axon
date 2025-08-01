import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
    PaginationRequest,
    PaginationResponse,
    TaskHistoryInCRUDResponse,
)
from core.models.services import TaskHistoryCreateRequestModel
from core.services import tasks_history as tasks_history_services


tasks_history_route = fastapi.APIRouter(
    prefix="/{task_id}/history", tags=["Tasks-history"]
)


@tasks_history_route.get(
    path="",
    name="获取执行记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PaginationResponse[TaskHistoryInCRUDResponse],
)
async def get(
    task_id: int = fastapi.Path(description="任务 ID"),
    pagination: PaginationRequest = Depends(PaginationRequest),
) -> PaginationResponse[TaskHistoryInCRUDResponse]:
    result = await tasks_history_services.get_histories(
        task_id=task_id, pagination=pagination
    )
    return PaginationResponse(
        **result.model_dump(
            exclude={"db_objects"},
        ),
        result=[
            TaskHistoryInCRUDResponse.model_validate(history)
            for history in result.db_objects
        ],
    )


@tasks_history_route.post(
    path="",
    name="插入执行记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskHistoryInCRUDResponse],
)
async def insert_task_history(
    request_model: TaskHistoryCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskHistoryInCRUDResponse]:
    history = await tasks_history_services.create_history(
        task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=TaskHistoryInCRUDResponse.model_validate(history))
