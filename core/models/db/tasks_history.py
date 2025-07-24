import typing

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.models.enums import TaskState
from core.models.db.base import BaseTableModel
from core.utils.enums import enum_values

if typing.TYPE_CHECKING:
    from core.models.db.tasks import Tasks


class TasksHistory(BaseTableModel):
    __tablename__ = "tasks_histories"
    __table_args__ = (
        sa.Index("idx_tasks_history_task_state", "task_id", "state"),
        {"comment": "任务历史记录表"},
    )

    task_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("tasks.id"),
        nullable=False,
        index=True,
        comment="关联任务ID",
    )

    state: Mapped[TaskState] = mapped_column(
        sa.Enum(TaskState, values_callable=enum_values),
        nullable=False,
        index=True,
        comment="任务执行状态",
    )

    process: Mapped[str] = mapped_column(
        sa.Text, nullable=False, comment="任务执行过程"
    )

    think: Mapped[str] = mapped_column(
        sa.Text, nullable=False, comment="Agent 的思考过程"
    )

    task: Mapped["Tasks"] = relationship(
        back_populates="histories",
    )
