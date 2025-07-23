import datetime
from typing import Generic, TypeVar

import pydantic
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel


T = TypeVar("T")


model_config = pydantic.ConfigDict(
    # 自动将 snake_case 字段名生成 camelCase 别名，用于 JSON 输出
    alias_generator=to_camel,
    # 允许在创建模型时使用别名（如 'taskId'）
    populate_by_name=True,
    # 允许从 ORM 对象等直接转换
    from_attributes=True,
    # 统一处理所有 datetime 对象的 JSON 序列化格式
    json_encoders={datetime.datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")},
)


class ResponseModel(BaseModel, Generic[T]):
    """
    为 Axon API 设计的、标准化的泛型响应模型。
    """

    model_config = model_config

    code: int = Field(default=200, description="状态码")
    message: str = Field(default="Success", description="响应消息")
    is_failed: bool = Field(default=False, description="是否失败")

    result: T | None = Field(default=None, description="响应体负载")
