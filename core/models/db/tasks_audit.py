import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from core.models.enums import TaskState, TaskAuditSource
from core.models.db.base import BaseTableModel
from core.utils.enums import enum_values


class TasksAudit(BaseTableModel):
    __tablename__ = "tasks_audit"
    __table_args__ = (
        sa.Index("idx_tasks_audit_task_id_created_at", "task_id", "created_at"),
        {"comment": "任务状态审计表"},
    )

    task_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("tasks.id"),
        nullable=False,
        index=True,
        comment="关联任务ID",
    )

    from_state: Mapped[TaskState] = mapped_column(
        sa.Enum(TaskState, values_callable=enum_values),
        nullable=False,
        index=True,
        comment="任务执行状态",
    )

    to_state: Mapped[TaskState] = mapped_column(
        sa.Enum(TaskState, values_callable=enum_values),
        nullable=False,
        index=True,
        comment="任务执行状态",
    )

    source: Mapped[TaskAuditSource] = mapped_column(
        sa.Enum(TaskAuditSource, values_callable=enum_values),
        nullable=False,
        index=True,
        comment="触发变更的来源",
    )

    source_context: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        comment="变更来源的上下文信息, 如 user_id, worker_id 等等 ..",
    )

    comment: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        comment="变更上下文的注释, 为什么要变更. 变更背景是什么 ..",
    )
