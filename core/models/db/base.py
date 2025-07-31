from typing import Any
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import DeclarativeBase, Mapped, Mapper, mapped_column
from sqlalchemy.orm.attributes import get_history


class BaseTableModel(DeclarativeBase):
    __abstract__ = True
    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)

    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        index=True,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=sa.func.now(),
        comment="创建时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        index=True,
        nullable=True,
        onupdate=sa.func.now(),
        server_onupdate=sa.func.now(),
        comment="更新时间",
    )

    deleted_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
        comment="删除时间",
    )

    is_deleted: Mapped[bool] = mapped_column(
        sa.Boolean,
        index=True,
        default=False,
        server_default=sa.text("0"),
        nullable=False,
        comment="0：未删除 1：已删除",
    )

    @classmethod
    def __table_cls__(
        cls, table_name: str, metadata: sa.MetaData, *args: Any, **kwargs: Any
    ):
        # 在生成 table 时, 必须确保 ID 排在第一个
        columns = sorted(
            args,
            key=lambda field: 0
            if (isinstance(field, sa.Column) and field.name == "id")
            else 1,
        )
        return sa.Table(table_name, metadata, *columns, **kwargs)


@event.listens_for(BaseTableModel, "before_update", propagate=True)
def set_deleted_at_on_soft_delete(
    mapper: Mapper[Any], connection: Connection, obj: BaseTableModel
) -> None:
    """
    当 is_deleted 变更时，自动设置 deleted_at 字段。
    """
    history = get_history(obj, "is_deleted")

    if (
        history.added
        and history.added[0] is True
        and history.deleted
        and history.deleted[0] is False
    ):
        obj.deleted_at = datetime.now(timezone.utc)
    elif (
        history.added
        and history.added[0] is False
        and history.deleted
        and history.deleted[0] is True
    ):
        obj.deleted_at = None
