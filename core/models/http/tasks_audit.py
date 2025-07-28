import datetime

from core.models.enums import TaskState, TaskAuditSource
from core.models.http.base import BaseHttpModel


class TaskAuditInCRUDResponse(BaseHttpModel):
    from_state: TaskState
    to_state: TaskState
    source: TaskAuditSource
    source_context: str
    comment: str
    created_at: datetime.datetime


class TaskAuditCreateRequestModel(BaseHttpModel):
    from_state: TaskState
    to_state: TaskState
    source: TaskAuditSource
    source_context: str
    comment: str
