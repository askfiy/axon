import sqlalchemy as sa

from core.models.db import TasksChat
from core.repository.crud.base import BaseCRUDRepository
from core.models.http import (
    PageinationRequest,
    PageinationResponse,
    TaskChatInCRUDResponse,
)


class TasksChatRepository(BaseCRUDRepository[TasksChat]):
    async def get_chats_pageination_response(
        self,
        task_id: int,
        pageination: PageinationRequest,
    ) -> PageinationResponse[TaskChatInCRUDResponse]:
        query_stmt = sa.select(self.model).where(
            TasksChat.task_id == task_id, sa.not_(self.model.is_deleted)
        )

        return await super().get_pageination_response_by_stmt(
            pageination_request=pageination,
            stmt=query_stmt,
            response_model_cls=TaskChatInCRUDResponse,
        )
