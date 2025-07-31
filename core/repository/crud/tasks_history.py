import sqlalchemy as sa

from core.models.db import TasksHistory
from core.models.http import PageinationRequest
from core.models.services import PageinationInfo
from core.repository.crud.base import BaseCRUDRepository


class TasksHistoryRepository(BaseCRUDRepository[TasksHistory]):
    async def get_histories_pageination_response(
        self,
        task_id: int,
        pageination: PageinationRequest,
    ) -> PageinationInfo[TasksHistory]:
        query_stmt = sa.select(self.model).where(
            self.model.task_id == task_id, sa.not_(self.model.is_deleted)
        )

        return await super().get_pageination_response_by_stmt(
            pageination_request=pageination,
            stmt=query_stmt,
        )
