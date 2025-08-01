from core.models.db import TasksAudit
from core.models.http import PaginationRequest
from core.models.services import Paginator, TaskAuditCreateRequestModel
from core.repository.crud import TasksCRUDRepository, TasksAuditRepository
from core.exceptions import ServiceNotFoundException
from core.database.connection import (
    get_async_session_direct,
    get_async_tx_session_direct,
)


async def get_audits(
    task_id: int, pagination: PaginationRequest
) -> Paginator[TasksAudit]:
    async with get_async_session_direct() as session:
        tasks_audit_repo = TasksAuditRepository(session=session)

        return await tasks_audit_repo.get_audits_pagination_response(
            task_id=task_id,
            pagination=pagination,
        )


async def create_audit(
    task_id: int, request_model: TaskAuditCreateRequestModel
) -> TasksAudit:
    async with get_async_tx_session_direct() as session:
        tasks_repo = TasksCRUDRepository(
            session=session,
        )
        task_exists = await tasks_repo.exists(pk=task_id)

        if not task_exists:
            raise ServiceNotFoundException(f"任务: {task_id} 不存在")

        tasks_audit_repo = TasksAuditRepository(session=session)

        return await tasks_audit_repo.create(
            create_info={"task_id": task_id, **request_model.model_dump()}
        )
