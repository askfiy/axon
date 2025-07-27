import fastapi
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.db import Tasks
from core.models.http import (
    PageinationRequest,
    PageinationResponse,
    TaskInCRUDResponse,
    TaskChatInCRUDResponse,
    TaskChatCreateRequestModel,
)

from core.repository.crud import (
    TasksCRUDRepository,
    TasksChatRepository,
)

from core.utils.decorators import transactional


async def get_chats(
    task_id: int, session: AsyncSession, pageination: PageinationRequest
) -> PageinationResponse[TaskChatInCRUDResponse]:
    tasks_chat_repo = TasksChatRepository(session=session)

    return await tasks_chat_repo.get_chats_pageination_response(
        task_id=task_id,
        pageination=pageination,
    )


@transactional
async def insert_task_chat(
    session: AsyncSession, task_id: int, request_model: TaskChatCreateRequestModel
) -> TaskInCRUDResponse:
    tasks_repo = TasksCRUDRepository(
        session=session,
    )
    task_exists = await tasks_repo.exists(pk=task_id)

    if not task_exists:
        raise HTTPException(
            status_code=fastapi.status.HTTP_404_NOT_FOUND,
            detail=f"任务: {task_id} 不存在",
        )

    tasks_chat_repo = TasksChatRepository(session=session)

    await tasks_chat_repo.create(
        create_info={"task_id": task_id, **request_model.model_dump()}
    )

    task = await tasks_repo.get(
        pk=task_id,
    )
    return TaskInCRUDResponse.model_validate(task)
