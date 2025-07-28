import sqlalchemy as sa

from core.models.db import TasksAudit
from core.repository.crud.base import BaseCRUDRepository
from core.models.http import (
    PageinationRequest,
    PageinationResponse,
    TaskAuditInCRUDResponse,
)


class TasksAuditRepository(BaseCRUDRepository[TasksAudit]):
    async def get_audits_pageination_response(
        self,
        task_id: int,
        pageination: PageinationRequest,
    ) -> PageinationResponse[TaskAuditInCRUDResponse]:
        query_stmt = sa.select(self.model).where(
            self.model.task_id == task_id, sa.not_(self.model.is_deleted)
        )

        return await super().get_pageination_response_by_stmt(
            pageination_request=pageination,
            stmt=query_stmt,
            response_model_cls=TaskAuditInCRUDResponse,
        )
