from core.models.db import TasksChat
from core.models.http import Paginator
from core.models.services import TaskChatCreateRequestModel
from core.repository.crud import TasksCRUDRepository, TasksChatRepository
from core.exceptions import ServiceNotFoundException
from core.database.connection import (
    get_async_session_direct,
    get_async_tx_session_direct,
)


async def upget_tasks_chat_pagination(task_id: int, paginator: Paginator) -> Paginator:
    async with get_async_session_direct() as session:
        tasks_chat_repo = TasksChatRepository(session=session)

        return await tasks_chat_repo.upget_tasks_chat_pagination(
            task_id=task_id,
            paginator=paginator,
        )


async def create_chat(
    task_id: int, request_model: TaskChatCreateRequestModel
) -> TasksChat:
    async with get_async_tx_session_direct() as session:
        tasks_repo = TasksCRUDRepository(
            session=session,
        )
        task_exists = await tasks_repo.exists(pk=task_id)

        if not task_exists:
            raise ServiceNotFoundException(f"任务: {task_id} 不存在")

        tasks_chat_repo = TasksChatRepository(session=session)

        return await tasks_chat_repo.create(
            create_info={"task_id": task_id, **request_model.model_dump()}
        )
