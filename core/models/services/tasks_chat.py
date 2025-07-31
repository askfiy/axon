from ..model import BaseModel
from ..enums import MessageRole


class TaskChatCreateRequestModel(BaseModel):
    message: str
    role: MessageRole
