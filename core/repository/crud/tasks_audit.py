import sqlalchemy as sa

from core.models.db import TasksAudit
from core.models.http import PaginationRequest
from core.models.services import Paginator
from core.repository.crud.base import BaseCRUDRepository


class TasksAuditRepository(BaseCRUDRepository[TasksAudit]):
    async def get_audits_pagination_response(
        self,
        task_id: int,
        pagination: PaginationRequest,
    ) -> Paginator[TasksAudit]:
        query_stmt = sa.select(self.model).where(
            self.model.task_id == task_id, sa.not_(self.model.is_deleted)
        )

        return await super().get_pagination_response_by_stmt(
            pagination_request=pagination,
            stmt=query_stmt,
        )
