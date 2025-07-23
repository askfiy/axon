from typing import TYPE_CHECKING


import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.models.db.base import BaseTableModel

if TYPE_CHECKING:
    from core.models.db.tasks import Tasks


class TasksMetadata(BaseTableModel):
    __tablename__ = "tasks_metadata"
    __table_args__ = (
        sa.Index("idx_keywords_fulltext", "keywords", mysql_prefix="FULLTEXT"),
        {"comment": "任务元信息表"},
    )

    owner: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
        comment="任务所有者",
        index=True,
    )
    owner_timezone: Mapped[str] = mapped_column(
        sa.String(255),
        nullable=False,
        default="UTC",
        comment="所有者所在时区",
        server_default=sa.text("'UTC'"),
    )
    keywords: Mapped[str] = mapped_column(sa.Text, nullable=False, comment="关键字信息")
    original_user_input: Mapped[str] = mapped_column(
        sa.Text, nullable=False, comment="原始用户输入"
    )
    planning: Mapped[str] = mapped_column(sa.Text, nullable=False, comment="执行规划")
    description: Mapped[str] = mapped_column(
        sa.Text, nullable=False, comment="描述信息"
    )
    accept_criteria: Mapped[str] = mapped_column(
        sa.Text, nullable=False, comment="验收标准"
    )

    tasks_rel: Mapped[list["Tasks"]] = relationship(
        back_populates="metadata_info",
        foreign_keys="[Tasks.metadata_id]",
    )
