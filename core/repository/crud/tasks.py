from typing import override, Any
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import aliased, subqueryload, with_loader_criteria
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import QueryableAttribute
from sqlalchemy.orm.strategy_options import _AbstractLoad
from sqlalchemy.orm.util import LoaderCriteriaOption

from core.models.db import Tasks, TasksChat, TasksHistory
from core.repository.crud.base import BaseCRUDRepository
from core.repository.crud.tasks_metadata import TasksMetadataRepository


class TasksCRUDRepository(BaseCRUDRepository[Tasks]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session)

        self.tasks_metadata_repo = TasksMetadataRepository(session=self.session)

    def _get_history_loader_options(
        self, limit_count: int = 10
    ) -> list[_AbstractLoad | LoaderCriteriaOption]:
        history_alias_for_ranking = aliased(TasksHistory)
        ranked_histories_cte = (
            sa.select(
                history_alias_for_ranking.id,
                sa.func.row_number()
                .over(
                    partition_by=history_alias_for_ranking.task_id,
                    order_by=history_alias_for_ranking.created_at.desc(),
                )
                .label("rn"),
            )
            .where(history_alias_for_ranking.task_id == Tasks.id)
            .cte("ranked_histories_cte")
        )

        return [
            subqueryload(Tasks.histories),
            with_loader_criteria(
                TasksHistory,
                TasksHistory.id.in_(
                    sa.select(ranked_histories_cte.c.id).where(
                        ranked_histories_cte.c.rn <= limit_count
                    )
                ),
            ),
        ]

    def _get_chat_loader_options(
        self, limit_count: int = 10
    ) -> list[_AbstractLoad | LoaderCriteriaOption]:
        chat_alias_for_ranking = aliased(TasksChat)
        ranked_chats_cte = (
            sa.select(
                chat_alias_for_ranking.id,
                sa.func.row_number()
                .over(
                    partition_by=chat_alias_for_ranking.task_id,
                    order_by=chat_alias_for_ranking.created_at.desc(),
                )
                .label("rn"),
            )
            .where(chat_alias_for_ranking.task_id == Tasks.id)
            .cte("ranked_chats_cte")
        )

        return [
            subqueryload(Tasks.chats),
            with_loader_criteria(
                TasksChat,
                TasksChat.id.in_(
                    sa.select(ranked_chats_cte.c.id).where(
                        ranked_chats_cte.c.rn <= limit_count
                    )
                ),
            ),
        ]

    @override
    async def get(
        self, pk: int, joined_loads: list[QueryableAttribute[Any]] | None = None
    ) -> Tasks | None:
        joined_loads = joined_loads or []

        stmt = sa.select(self.model).where(
            self.model.id == pk, sa.not_(self.model.is_deleted)
        )

        if joined_loads:
            for join_field in joined_loads:
                if Tasks.chats is join_field:
                    stmt = stmt.options(*self._get_chat_loader_options())
                if Tasks.histories is join_field:
                    stmt = stmt.options(*self._get_history_loader_options())

        result = await self.session.execute(stmt)

        return result.unique().scalar_one_or_none()

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
