from ..model import BaseModel
from ..enums import TaskState, TaskAuditSource


class TaskAuditCreateRequestModel(BaseModel):
    from_state: TaskState
    to_state: TaskState
    source: TaskAuditSource
    source_context: str
    comment: str
