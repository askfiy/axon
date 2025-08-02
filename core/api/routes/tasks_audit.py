import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
    Paginator,
    PaginationRequest,
    PaginationResponse,
    TaskAuditInCRUDResponse,
)
from core.models.services import TaskAuditCreateRequestModel
from core.services import tasks_audit as tasks_audit_services


tasks_audit_route = fastapi.APIRouter(prefix="/{task_id}/audit", tags=["Tasks-audit"])


@tasks_audit_route.get(
    path="",
    name="获取审查记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PaginationResponse,
)
async def get(
    task_id: int = fastapi.Path(description="任务 ID"),
    request: PaginationRequest = Depends(PaginationRequest),
) -> PaginationResponse:
    paginator = Paginator(
        request=request,
        serializer_cls=TaskAuditInCRUDResponse,
    )
    paginator = await tasks_audit_services.upget_tasks_audit_pagination(
        task_id=task_id, paginator=paginator
    )
    return paginator.response


@tasks_audit_route.post(
    path="",
    name="插入审查记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskAuditInCRUDResponse],
)
async def insert_task_chat(
    request_model: TaskAuditCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskAuditInCRUDResponse]:
    task_audit = await tasks_audit_services.create_audit(
        task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=TaskAuditInCRUDResponse.model_validate(task_audit))
