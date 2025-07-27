import typing
from typing import Any, Generic, TypeVar, Type, Literal
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.db import BaseTableModel
from core.models.http import BaseHttpModel, PageinationRequest, PageinationResponse

ModelType = TypeVar("ModelType", bound=BaseTableModel)
ModelInCRUDResponse = TypeVar("ModelInCRUDResponse", bound=BaseHttpModel)


class BaseCRUDRepository(Generic[ModelType]):
    """
    基本的 Crud Repository. 将自动提供 get/get_all/create/delete 等方法.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.model: type[ModelType] = typing.get_args(self.__class__.__orig_bases__[0])[
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

    async def get_pageination_response(
        self,
        pageination_request: PageinationRequest,
        response_model_cls: Type[ModelInCRUDResponse],
        joined_loads: list[InstrumentedAttribute[Any]] | None = None,
    ) -> PageinationResponse[ModelInCRUDResponse]:
        """
        返回默认的分页对象
        """
        stmt = sa.select(self.model).where(sa.not_(self.model.is_deleted))

        if joined_loads:
            for join_field in joined_loads:
                stmt = stmt.options(joinedload(join_field))

        return await self.get_pageination_response_by_stmt(
            pageination_request=pageination_request,
            stmt=stmt,
            response_model_cls=response_model_cls,
        )

    async def get_pageination_response_by_stmt(
        self,
        pageination_request: PageinationRequest,
        stmt: sa.Select[Any],
        response_model_cls: Type[ModelInCRUDResponse],
    ) -> PageinationResponse[ModelInCRUDResponse]:
        """
        执行 stmt 语句. 并将结果返回为分页对象.
        """

        # 应用排序逻辑
        for field_name, order_direction in pageination_request.order_by_rule:
            if not hasattr(self.model, field_name):
                raise ValueError(
                    f"{self.model.__name__} is not has field'{field_name}'"
                )
            order_func = sa.asc if order_direction == "asc" else sa.desc
            stmt = stmt.order_by(order_func(getattr(self.model, field_name)))

        page = pageination_request.page
        page_size = pageination_request.size

        # 计算总记录数
        count_stmt = sa.select(sa.func.count()).select_from(stmt.subquery())
        total_items_result = await self.session.execute(count_stmt)
        total_items = total_items_result.scalar_one()

        # 应用分页逻辑
        paginated_stmt = stmt.offset((page - 1) * page_size).limit(page_size)

        # 执行查询并获取 ORM 模型的列表
        orm_models = (
            (await self.session.execute(paginated_stmt)).scalars().unique().all()
        )

        # 将 ORM 模型列表转换为 Pydantic 响应模型列表
        paginated_response_items = [
            response_model_cls.model_validate(model) for model in orm_models
        ]

        return PageinationResponse(
            current_page=page,
            current_size=page_size,
            total_counts=total_items,
            result=paginated_response_items,
        )
