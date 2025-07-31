from ..enums import TaskState
from ..model import BaseModel


class TaskHistoryCreateRequestModel(BaseModel):
    state: TaskState
    process: str
    thinking: str
