import datetime

from ..model import BaseModel
from ..enums import TaskState
from .tasks_metadata import TaskMetaDataRequestModel


class TaskCreateRequestModel(BaseModel):
    name: str
    expect_execute_time: datetime.datetime
    background: str
    objective: str
    details: str

    dependencies: list[int] = []
    parent_id: int | None = None
    metadata: TaskMetaDataRequestModel | None = None


class TaskUpdateRequestModel(BaseModel):
    name: str | None = None
    state: TaskState | None = None
    priority: int | None = None
    expect_execute_time: datetime.datetime | None = None
    lasted_execute_time: datetime.datetime | None = None

    background: str | None = None
    objective: str | None = None
    details: str | None = None

    dependencies: list[int] | None = None
    parent_id: int | None = None
    metadata: TaskMetaDataRequestModel | None = None
