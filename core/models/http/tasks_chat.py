import datetime

from core.models.enums import MessageRole
from core.models.http.base import BaseHttpModel


class TaskChatInCRUDResponse(BaseHttpModel):
    message: str
    role: MessageRole
    created_at: datetime.datetime


class TaskChatCreateRequestModel(BaseHttpModel):
    message: str
    role: MessageRole
