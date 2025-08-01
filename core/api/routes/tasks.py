import fastapi
from fastapi import Depends

from core.services import tasks as tasks_services
from core.models.http import (
    ResponseModel,
    PaginationRequest,
    PaginationResponse,
    TaskInCRUDResponse,
)
from core.models.services import TaskCreateRequestModel, TaskUpdateRequestModel

tasks_route = fastapi.APIRouter(prefix="/tasks", tags=["Tasks"])


@tasks_route.post(
    path="",
    name="创建任务",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def create(
    request_model: TaskCreateRequestModel,
) -> ResponseModel[TaskInCRUDResponse]:
    task = await tasks_services.create_task(request_model=request_model)
    return ResponseModel(result=TaskInCRUDResponse.model_validate(task))


@tasks_route.get(
    path="",
    name="获取全部任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PaginationResponse[TaskInCRUDResponse],
)
async def get(
    pagination: PaginationRequest = Depends(PaginationRequest),
) -> PaginationResponse[TaskInCRUDResponse]:
    result = await tasks_services.get_tasks(pagination=pagination)
    return PaginationResponse(
        **result.model_dump(
            exclude={"db_objects"},
        ),
        result=[TaskInCRUDResponse.model_validate(task) for task in result.db_objects],
    )


@tasks_route.get(
    path="/{task_id}",
    name="获取某个任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def get_by_id(
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    task = await tasks_services.get_task_by_id(task_id=task_id)
    return ResponseModel(result=TaskInCRUDResponse.model_validate(task))


@tasks_route.put(
    path="/{task_id}",
    name="更新某个任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[TaskInCRUDResponse],
)
async def update(
    request_model: TaskUpdateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskInCRUDResponse]:
    task = await tasks_services.update_task(
        task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=TaskInCRUDResponse.model_validate(task))


@tasks_route.delete(
    path="/{task_id}",
    name="删除某个任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[bool],
)
async def delete(
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[bool]:
    result = await tasks_services.delete_task_by_id(task_id=task_id)
    return ResponseModel(result=result)
