import datetime

from pydantic import field_validator


from ..enums import TaskState
from ..model import BaseModel
from .tasks_chat import TaskChatInCRUDResponse
from .tasks_history import TaskHistoryInCRUDResponse


class TaskInCRUDResponse(BaseModel):
    id: int
    state: TaskState

    name: str
    deep_level: int
    expect_execute_time: datetime.datetime
    background: str
    objective: str
    details: str
    dependencies: list[int] | None
    parent_id: int | None

    chats: list[TaskChatInCRUDResponse]
    histories: list[TaskHistoryInCRUDResponse]

    # DEP: 已废弃. 使用 db 查询时就完全做好排序. 不需要下面的二次操作
    # raw_chats: list[TaskChatInCRUDResponse] = Field(default_factory=list, exclude=True, alias="chats")
    #
    # @computed_field
    # @property
    # def chats(self) -> list[TaskChatInCRUDResponse]:
    #     """
    #     处理聊天记录的排序和限制逻辑：
    #     1. 接收已按 created_at 倒序排列的原始聊天记录 (_raw_chats)。
    #     2. 取前 10 条（即最近的 10 条）。
    #     3. 对这 10 条记录再按 created_at 升序排序。
    #     """
    #     # 2. 对这 10 条记录再按 created_at 升序排序
    #     final_sorted_chats = sorted(self.raw_chats, key=lambda chat: chat.created_at)
    #     return final_sorted_chats

    @field_validator("expect_execute_time", mode="before")
    @classmethod
    def assume_utc_if_naive(
        cls, v: datetime.datetime | None
    ) -> datetime.datetime | None:
        """
        如果传入的 datetime 对象是“天真”的，就强制为它附加 UTC 时区。
        我们知道数据库存的是 UTC，所以这是安全的。
        """
        if isinstance(v, datetime.datetime) and v.tzinfo is None:
            utc_aware_time = v.replace(tzinfo=datetime.timezone.utc)
            return utc_aware_time
        return v
