from typing import Annotated

import fastapi
from fastapi import Depends

from core.models.http import (
    ResponseModel,
    PageinationRequest,
    PageinationResponse,
    TaskAuditInCRUDResponse,
    TaskAuditCreateRequestModel,
)
from core.services import tasks_audit as tasks_audit_services
from core.api.dependencies import get_async_session, AsyncSession


tasks_audit_route = fastapi.APIRouter(prefix="/{task_id}/audit", tags=["Tasks-audit"])


@tasks_audit_route.get(
    path="",
    name="获取审查记录",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=PageinationResponse[TaskAuditInCRUDResponse],
)
async def get(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    task_id: int = fastapi.Path(description="任务 ID"),
    pageination: PageinationRequest = Depends(PageinationRequest),
) -> PageinationResponse[TaskAuditInCRUDResponse]:
    result = await tasks_audit_services.get_audits(
        task_id=task_id, session=session, pageination=pageination
    )
    return result


@tasks_audit_route.post(
    path="",
    name="插入审查记录",
    status_code=fastapi.status.HTTP_201_CREATED,
    response_model=ResponseModel[TaskAuditInCRUDResponse],
)
async def insert_task_chat(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    request_model: TaskAuditCreateRequestModel,
    task_id: int = fastapi.Path(description="任务 ID"),
) -> ResponseModel[TaskAuditInCRUDResponse]:
    result = await tasks_audit_services.insert_task_audit(
        session=session, task_id=task_id, request_model=request_model
    )
    return ResponseModel(result=result)
