import typing
from typing import Any, Generic, TypeVar, Type
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
        await self.session.refresh(db_obj)
        return db_obj

    async def update(self, db_obj: ModelType, update_info: dict[str, Any]) -> ModelType:
        """更新一个已有的对象"""
        for key, value in update_info.items():
            setattr(db_obj, key, value)

        self.session.add(db_obj)
        self.session.refresh(db_obj)
        return db_obj

    async def get_pageination_response(
        self,
        pagination_request: PageinationRequest,
        query_stmt: sa.Select[Any],
        response_model_cls: Type[ModelInCRUDResponse],
    ) -> PageinationResponse[ModelInCRUDResponse]:
        page = pagination_request.page
        page_size = pagination_request.size

        # 计算总记录数
        count_stmt = sa.select(sa.func.count()).select_from(query_stmt.subquery())
        total_items_result = await self.session.execute(count_stmt)
        total_items = total_items_result.scalar_one()

        # 应用分页逻辑
        paginated_stmt = query_stmt.offset((page - 1) * page_size).limit(page_size)

        # 执行查询并获取 ORM 模型的列表
        orm_models = (
            (await self.session.execute(paginated_stmt)).scalars().unique().all()
        )

        # 将 ORM 模型列表转换为 Pydantic 响应模型列表
        paginated_response_items = [
            response_model_cls.model_validate(model) for model in orm_models
        ]

        # 构造并返回 PaginationResponse
        # 这里的 T_ResponseModel 类型参数会自动从 response_model_cls 推断出来
        return PageinationResponse(
            current_page=page,
            current_size=page_size,
            total_counts=total_items,
            result=paginated_response_items,
        )
