import typing
from typing import Any, Generic, TypeVar
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.attributes import QueryableAttribute
from sqlalchemy.ext.asyncio import AsyncSession
from core.models.db.base import BaseTableModel

ModelType = TypeVar("ModelType", bound=BaseTableModel)


class BaseCRUDRepository(Generic[ModelType]):
    """
    基本的 CRUDRespository. 将自动提供 get/get_all/create/delete 等方法.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.model: type[ModelType] = typing.get_args(self.__class__.__orig_bases__[0])[
            0
        ]

    async def get(
        self, pk: int, joined_loads: list[QueryableAttribute[Any]] | None = None
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

    async def get_all(self) -> Sequence[ModelType]:
        """获取所有未被软删除的对象"""
        stmt = sa.select(self.model).where(sa.not_(self.model.is_deleted))
        result = await self.session.execute(stmt)
        return result.scalars().all()

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
        self.session.refresh(db_obj)
        return db_obj

    async def update(self, db_obj: ModelType, update_info: dict[str, Any]) -> ModelType:
        """更新一个已有的对象"""
        for key, value in update_info.items():
            setattr(db_obj, key, value)

        self.session.add(db_obj)
        self.session.refresh(db_obj)
        return db_obj
