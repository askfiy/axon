import typing
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.models.enums import MessageRole
from core.models.db.base import BaseTableModel
from core.utils.enums import enum_values

if typing.TYPE_CHECKING:
    from core.models.db.tasks import Tasks

class TasksChat(BaseTableModel):
    __tablename__ = "tasks_chat"
    __table_args__ = (
        sa.Index("idx_tasks_chat_task_role", "task_id", "role"),
        {"comment": "任务聊天记录表"},
    )

    task_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.ForeignKey("tasks.id"),
        nullable=False,
        index=True,
        comment="关联任务ID",
    )

    role: Mapped[MessageRole] = mapped_column(
        sa.Enum(MessageRole, values_callable=enum_values),
        nullable=False,
        index=True,
        comment="发送消息的角色",
    )

    message: Mapped[str] = mapped_column(sa.Text, nullable=False, comment="对话消息")

    task: Mapped["Tasks"] = relationship(
        back_populates="chats",
    )
