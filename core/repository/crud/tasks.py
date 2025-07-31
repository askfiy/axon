from typing import override, Any
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import aliased, subqueryload, with_loader_criteria, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.strategy_options import _AbstractLoad  # pyright: ignore[reportPrivateUsage]
from sqlalchemy.orm.util import LoaderCriteriaOption

from core.models.db import Tasks, TasksChat, TasksHistory
from core.models.http import PageinationRequest, PageinationResponse, TaskInCRUDResponse
from core.repository.crud.base import BaseCRUDRepository
from core.repository.crud.tasks_metadata import TasksMetadataRepository


class TasksCRUDRepository(BaseCRUDRepository[Tasks]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session)

        self.default_limit_count = 10
        self.default_joined_loads = [Tasks.chats, Tasks.histories]
        self.tasks_metadata_repo = TasksMetadataRepository(session=self.session)

    def _get_history_loader_options(
        self, limit_count: int
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
        self, limit_count: int
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
    async def create(self, create_info: dict[str, Any]) -> Tasks:
        task = await super().create(create_info=create_info)
        # 创建 task 后需要手动 load 一下 chats 和 histories.
        await self.session.refresh(task, [Tasks.chats.key, Tasks.histories.key])
        return task

    @override
    async def get(
        self, pk: int, joined_loads: list[InstrumentedAttribute[Any]] | None = None
    ) -> Tasks | None:
        joined_loads = joined_loads or self.default_joined_loads

        stmt = sa.select(self.model).where(
            self.model.id == pk, sa.not_(self.model.is_deleted)
        )

        if joined_loads:
            for join_field in joined_loads:
                if Tasks.chats is join_field:
                    stmt = stmt.options(
                        *self._get_chat_loader_options(self.default_limit_count)
                    )
                elif Tasks.histories is join_field:
                    stmt = stmt.options(
                        *self._get_history_loader_options(self.default_limit_count)
                    )
                else:
                    stmt = stmt.options(joinedload(join_field))

        result = await self.session.execute(stmt)

        return result.unique().scalar_one_or_none()

    @override
    async def get_all(
        self, joined_loads: list[InstrumentedAttribute[Any]] | None = None
    ) -> Sequence[Tasks]:
        return await super().get_all(
            joined_loads=joined_loads or self.default_joined_loads
        )

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

        # 获得需要删除的所有的任务, 子任务和当前任务
        related_task_ids = sa.select(cte.c.id)

        # 软删除 tasks
        await self.session.execute(
            sa.update(Tasks)
            .where(Tasks.id.in_(related_task_ids), sa.not_(Tasks.is_deleted))
            .values(is_deleted=True, deleted_at=sa.func.now())
        )

        # 软删除若有任务的 chats
        await self.session.execute(
            sa.update(TasksChat)
            .where(
                TasksChat.task_id.in_(related_task_ids),
                sa.not_(TasksChat.is_deleted),
            )
            .values(is_deleted=True, deleted_at=sa.func.now())
        )

        # 软删除所有任务的 histories
        await self.session.execute(
            sa.update(TasksHistory)
            .where(
                TasksHistory.task_id.in_(related_task_ids),
                sa.not_(TasksHistory.is_deleted),
            )
            .values(is_deleted=True, deleted_at=sa.func.now())
        )

        # 若为根任务. 且 metainfo 未被删除, 则软删除.
        if db_obj.parent is None and not task.metadata_info.is_deleted:
            await self.tasks_metadata_repo.delete(db_obj.metadata_info)

        # 因为有事务装饰器的存在， 故这里所有的操作均为原子操作.
        await self.session.refresh(task)

        return task

    async def get_tasks_pageination_response(
        self,
        pageination: PageinationRequest,
    ) -> PageinationResponse[TaskInCRUDResponse]:
        query_stmt = sa.select(self.model).where(sa.not_(self.model.is_deleted))
        query_stmt = query_stmt.options(
            *self._get_chat_loader_options(self.default_limit_count)
        )
        query_stmt = query_stmt.options(
            *self._get_history_loader_options(self.default_limit_count)
        )

        return await super().get_pageination_response_by_stmt(
            pageination_request=pageination,
            stmt=query_stmt,
            response_model_cls=TaskInCRUDResponse,
        )
