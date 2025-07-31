from core.models.db import TasksChat
from core.models.http import PageinationRequest
from core.models.services import PageinationInfo, TaskChatCreateRequestModel
from core.repository.crud import TasksCRUDRepository, TasksChatRepository
from core.exceptions import ServiceNotFoundException
from core.database.connection import (
    get_async_session_direct,
    get_async_tx_session_direct,
)


async def get_chats(
    task_id: int, pageination: PageinationRequest
) -> PageinationInfo[TasksChat]:
    async with get_async_session_direct() as session:
        tasks_chat_repo = TasksChatRepository(session=session)

        return await tasks_chat_repo.get_chats_pageination_response(
            task_id=task_id,
            pageination=pageination,
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
