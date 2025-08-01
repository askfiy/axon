from enum import StrEnum


class TaskState(StrEnum):
    # 任务建立状态
    INITIAL = "initial"
    # 任务等待调度
    SCHEDULED = "scheduled"
    # 任务进入队列
    ENQUEUED = "enqueued"
    # 任务正在执行
    ACTIVATING = "activating"
    # 任务等待子任务
    PENDING = "pending"
    # 任务等待用户输入
    WAITING = "waiting"
    # 任务正在重试
    RETRYING = "retrying"
    # 任务已被取消
    CANCEL = "cancel"
    # 任务已经完成
    FINISH = "finish"
    # 任务已经失败
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class TaskAuditSource(StrEnum):
    """
    触发任务状态变更的“来源”枚举
    """

    USER = "user"
    ADMIN = "admin"
    AGENT = "agent"
    SYSTEM = "system"
