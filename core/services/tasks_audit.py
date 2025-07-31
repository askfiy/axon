import fastapi
from fastapi import HTTPException

from core.models.http import (
    PageinationRequest,
    PageinationResponse,
    TaskAuditInCRUDResponse,
    TaskAuditCreateRequestModel,
)
from core.repository.crud import (
    TasksCRUDRepository,
    TasksAuditRepository,
)
from core.api.dependencies import (
    AsyncSession,
    AsyncTxSession,
)


async def get_audits(
    task_id: int, session: AsyncSession, pageination: PageinationRequest
) -> PageinationResponse[TaskAuditInCRUDResponse]:
    tasks_audit_repo = TasksAuditRepository(session=session)

    return await tasks_audit_repo.get_audits_pageination_response(
        task_id=task_id,
        pageination=pageination,
    )


async def insert_task_audit(
    session: AsyncTxSession, task_id: int, request_model: TaskAuditCreateRequestModel
) -> TaskAuditInCRUDResponse:
    tasks_repo = TasksCRUDRepository(
        session=session,
    )
    task_exists = await tasks_repo.exists(pk=task_id)

    if not task_exists:
        raise HTTPException(
            status_code=fastapi.status.HTTP_404_NOT_FOUND,
            detail=f"任务: {task_id} 不存在",
        )

    tasks_audit_repo = TasksAuditRepository(session=session)

    task_audit = await tasks_audit_repo.create(
        create_info={"task_id": task_id, **request_model.model_dump()}
    )

    return TaskAuditInCRUDResponse.model_validate(task_audit)
