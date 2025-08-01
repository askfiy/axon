import typing
from typing import Any, Generic, TypeVar
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.db import BaseTableModel
from core.models.http import PaginationRequest
from core.models.services import Paginator

ModelType = TypeVar("ModelType", bound=BaseTableModel)


class BaseCRUDRepository(Generic[ModelType]):
    """
    基本的 Crud Repository. 将自动提供 get/get_all/create/delete 等方法.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.model: type[ModelType] = typing.get_args(self.__class__.__orig_bases__[0])[  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            0
        ]

    async def exists(self, pk: int) -> bool:
        exists_stmt = (
            sa.select(self.model.id)
            .where(self.model.id == pk, sa.not_(self.model.is_deleted))
            .exists()
        )

        stmt = sa.select(sa.literal(True)).where(exists_stmt)

        result = await self.session.execute(stmt)

        return result.scalar_one_or_none() is not None

    async def get(
        self, pk: int, joined_loads: list[InstrumentedAttribute[Any]] | None = None
    ) -> ModelType | None:
        """根据主键 ID 获取单个对象"""
        stmt = sa.select(self.model).where(
            self.model.id == pk, sa.not_(self.model.is_deleted)
        )

        if joined_loads:
            for join_field in joined_loads:
                stmt = stmt.options(joinedload(join_field))

        result = await self.session.execute(stmt)

        return result.unique().scalar_one_or_none()

    async def get_all(
        self, joined_loads: list[InstrumentedAttribute[Any]] | None = None
    ) -> Sequence[ModelType]:
        """获取所有未被软删除的对象"""
        stmt = sa.select(self.model).where(sa.not_(self.model.is_deleted))

        if joined_loads:
            for join_field in joined_loads:
                stmt = stmt.options(joinedload(join_field))

        result = await self.session.execute(stmt)
        return result.scalars().unique().all()

    async def create(self, create_info: dict[str, Any]) -> ModelType:
        """创建一个新对象"""
        db_obj = self.model(**create_info)
        self.session.add(db_obj)
        await self.session.flush()

        return db_obj

    async def delete(self, db_obj: ModelType) -> ModelType:
        """根据主键 ID 软删除一个对象"""
        db_obj.is_deleted = True

        self.session.add(db_obj)
        return db_obj

    async def update(self, db_obj: ModelType, update_info: dict[str, Any]) -> ModelType:
        """更新一个已有的对象"""
        for key, value in update_info.items():
            setattr(db_obj, key, value)

        self.session.add(db_obj)
        return db_obj

    async def get_pagination_response(
        self,
        pagination_request: PaginationRequest,
        joined_loads: list[InstrumentedAttribute[Any]] | None = None,
    ) -> Paginator[ModelType]:
        """
        返回默认的分页对象
        """
        stmt = sa.select(self.model).where(sa.not_(self.model.is_deleted))

        if joined_loads:
            for join_field in joined_loads:
                stmt = stmt.options(joinedload(join_field))

        return await self.get_pagination_response_by_stmt(
            pagination_request=pagination_request,
            stmt=stmt,
        )

    async def get_pagination_response_by_stmt(
        self,
        pagination_request: PaginationRequest,
        stmt: sa.Select[Any],
    ) -> Paginator[ModelType]:
        """
        执行 stmt 语句. 并将结果返回为分页对象.
        """

        # 应用排序逻辑
        for field_name, order_direction in pagination_request.order_by_rule:
            if not hasattr(self.model, field_name):
                raise ValueError(
                    f"{self.model.__name__} is not has field'{field_name}'"
                )
            order_func = sa.asc if order_direction == "asc" else sa.desc
            stmt = stmt.order_by(order_func(getattr(self.model, field_name)))

        page = pagination_request.page
        page_size = pagination_request.size

        # 计算总记录数
        count_stmt = sa.select(sa.func.count()).select_from(stmt.subquery())
        total_items_result = await self.session.execute(count_stmt)
        total_items = total_items_result.scalar_one()

        # 应用分页逻辑
        paginated_stmt = stmt.offset((page - 1) * page_size).limit(page_size)

        result = await self.session.execute(paginated_stmt)
        return Paginator(
            current_page=page,
            current_size=page_size,
            total_counts=total_items,
            db_objects=result.scalars().unique().all(),
        )
