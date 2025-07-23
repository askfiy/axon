from typing import override

import sqlalchemy as sa
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.db.tasks import Tasks
from core.crud import BaseCRUDRepository
from core.crud.tasks_metadata import TasksMetadataRepository


class TasksCRUDRepository(BaseCRUDRepository[Tasks]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session)

        self.tasks_metadata_repo = TasksMetadataRepository(session=self.session)

    @override
    async def delete(self, db_obj: Tasks) -> Tasks:
        task = db_obj

        cte = (
            sa.select(Tasks.id)
            .where(Tasks.id == task.id)
            .cte("descendants", recursive=True)
        )
        aliased_tasks = aliased(Tasks)  # 给 Tasks 表起个别名

        cte = cte.union_all(
            sa.select(aliased_tasks.id).where(aliased_tasks.parent_id == cte.c.id)
        )

        stmt = (
            sa.update(Tasks)
            .where(Tasks.id.in_(sa.select(cte.c.id)), sa.not_(Tasks.is_deleted))
            .values(is_deleted=True, deleted_at=sa.func.now())
        )

        # 因为有事务装饰器的存在， 故这里所有的操作均为原子操作.
        if db_obj.parent is None and not task.metadata_info.is_deleted:
            await self.tasks_metadata_repo.delete(db_obj.metadata_info)

        await self.session.execute(stmt)
        await self.session.refresh(task)

        return task
