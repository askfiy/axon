from core.models.db import TasksHistory
from core.models.http import Paginator
from core.models.services import TaskHistoryCreateRequestModel
from core.repository.crud import TasksCRUDRepository, TasksHistoryRepository
from core.exceptions import ServiceNotFoundException
from core.database.connection import (
    get_async_session_direct,
    get_async_tx_session_direct,
)


async def upget_tasks_history_pagination(
    task_id: int, paginator: Paginator
) -> Paginator:
    async with get_async_session_direct() as session:
        tasks_history_repo = TasksHistoryRepository(session=session)

        return await tasks_history_repo.upget_tasks_history_pagination(
            task_id=task_id,
            paginator=paginator,
        )


async def create_history(
    task_id: int, request_model: TaskHistoryCreateRequestModel
) -> TasksHistory:
    async with get_async_tx_session_direct() as session:
        tasks_repo = TasksCRUDRepository(
            session=session,
        )
        task_exists = await tasks_repo.exists(pk=task_id)

        if not task_exists:
            raise ServiceNotFoundException(f"任务: {task_id} 不存在")

        tasks_history_repo = TasksHistoryRepository(session=session)

        return await tasks_history_repo.create(
            create_info={"task_id": task_id, **request_model.model_dump()}
        )
