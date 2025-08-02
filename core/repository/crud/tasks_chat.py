from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from core.models.db import TasksChat
from core.models.http import Paginator
from core.repository.crud.base import BaseCRUDRepository


class TasksChatRepository(BaseCRUDRepository[TasksChat]):
    async def upget_tasks_chat_pagination(
        self,
        task_id: int,
        paginator: Paginator,
        joined_loads: list[InstrumentedAttribute[Any]] | None = None,
    ) -> Paginator:
        query_stmt = sa.select(self.model).where(
            self.model.task_id == task_id, sa.not_(self.model.is_deleted)
        )

        return await super().upget_pagination_by_stmt(
            paginator=paginator,
            stmt=query_stmt,
        )
