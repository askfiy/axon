import datetime

from core.models.enums import TaskState
from core.models.http.base import BaseHttpModel


class TaskHistoryInCRUDResponse(BaseHttpModel):
    state: TaskState
    process: str
    thinking: str
    created_at: datetime.datetime


class TaskHistoryCreateRequestModel(BaseHttpModel):
    state: TaskState
    process: str
    thinking: str
