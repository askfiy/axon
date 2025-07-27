import fastapi
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.http import (
    PageinationRequest,
    PageinationResponse,
    TaskInCRUDResponse,
    TaskHistoryInCRUDResponse,
    TaskHistoryCreateRequestModel,
)

from core.repository.crud import (
    TasksCRUDRepository,
    TasksHistoryRepository,
)

from core.utils.decorators import transactional


async def get_histories(
    task_id: int, session: AsyncSession, pageination: PageinationRequest
) -> PageinationResponse[TaskHistoryInCRUDResponse]:
    tasks_history_repo = TasksHistoryRepository(session=session)

    return await tasks_history_repo.get_histories_pageination_response(
        task_id=task_id,
        pageination=pageination,
    )


@transactional
async def insert_task_history(
    session: AsyncSession, task_id: int, request_model: TaskHistoryCreateRequestModel
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

    tasks_history_repo = TasksHistoryRepository(session=session)

    await tasks_history_repo.create(
        create_info={"task_id": task_id, **request_model.model_dump()}
    )

    task = await tasks_repo.get(
        pk=task_id,
    )
    return TaskInCRUDResponse.model_validate(task)
