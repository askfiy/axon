import uuid
import datetime
from typing import Optional, TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.utils.enums import enum_values
from core.models.enums import TaskState
from core.models.db.base import BaseTableModel
from core.models.db.tasks_chat import TasksChat
from core.models.db.tasks_history import TasksHistory
from core.models.db.tasks_metadata import TasksMetadata


class Tasks(BaseTableModel):
    __tablename__ = "tasks"
    __table_args__ = (
        sa.Index(
            "idx_tasks_state_priority_time", "state", "priority", "expect_execute_time"
        ),
        {"comment": "任务表"},
    )

    name: Mapped[str] = mapped_column(
        sa.String(255), index=True, nullable=False, comment="任务的名称"
    )
    identifier: Mapped[uuid.UUID] = mapped_column(
        sa.CHAR(36),
        unique=True,
        nullable=False,
        default=uuid.uuid4,
        server_default=sa.text("UUID()"),
        comment="任务的标识符",
    )
    deep_level: Mapped[int] = mapped_column(
        sa.Integer,
        index=True,
        nullable=False,
        default=0,
        comment="任务的层级",
        server_default=sa.text("0"),
    )
    priority: Mapped[int] = mapped_column(
        sa.Integer,
        nullable=False,
        default=0,
        index=True,
        comment="任务优先级",
        server_default=sa.text("0"),
    )
    expect_execute_time: Mapped[datetime.datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="任务预期执行时间",
        server_default=sa.func.now(),
    )
    lasted_execute_time: Mapped[Optional[datetime.datetime]] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,
        comment="任务最终执行时间",
    )
    state: Mapped[TaskState] = mapped_column(
        sa.Enum(TaskState, values_callable=enum_values),
        nullable=False,
        default=TaskState.INITIAL,
        index=True,
        comment="任务当前状态",
        server_default=TaskState.INITIAL.value,
    )
    background: Mapped[str] = mapped_column(sa.Text, nullable=False, comment="任务背景")
    objective: Mapped[str] = mapped_column(sa.Text, nullable=False, comment="任务目标")
    details: Mapped[str] = mapped_column(
        sa.Text, nullable=False, comment="任务的详细信息"
    )
    dependencies: Mapped[Optional[list[int]]] = mapped_column(
        sa.JSON,
        default=sa.func.json_array(),
        comment="同级任务依赖关系",
        server_default=sa.text("JSON_ARRAY()"),
    )

    parent_id: Mapped[Optional[int]] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("tasks.id"),
        nullable=True,
        index=True,
        comment="任务的父任务ID",
    )

    metadata_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("tasks_metadata.id"),
        nullable=False,
        index=True,
        comment="任务的元信息ID",
    )

    metadata_info: Mapped["TasksMetadata"] = relationship(
        back_populates="tasks_rel",
    )

    chats: Mapped[list["TasksChat"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by=TasksChat.created_at.desc(),
    )

    histories: Mapped[list["TasksHistory"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by=TasksHistory.created_at.desc(),
    )

    parent: Mapped[Optional["Tasks"]] = relationship(
        remote_side="Tasks.id",
        back_populates="children",
    )
    children: Mapped[list["Tasks"]] = relationship(
        back_populates="parent",
    )
