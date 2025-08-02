from core.models.db import TasksAudit
from core.models.http import Paginator
from core.models.services import TaskAuditCreateRequestModel
from core.repository.crud import TasksCRUDRepository, TasksAuditRepository
from core.exceptions import ServiceNotFoundException
from core.database.connection import (
    get_async_session_direct,
    get_async_tx_session_direct,
)


async def upget_tasks_audit_pagination(task_id: int, paginator: Paginator) -> Paginator:
    async with get_async_session_direct() as session:
        tasks_audit_repo = TasksAuditRepository(session=session)

        return await tasks_audit_repo.upget_tasks_audit_pagination(
            task_id=task_id,
            paginator=paginator,
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
