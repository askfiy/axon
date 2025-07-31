import datetime
from typing import TypeVar

import pydantic
from pydantic.alias_generators import to_camel


T = TypeVar("T")

model_config = pydantic.ConfigDict(
    # 自动将 snake_case 字段名生成 camelCase 别名，用于 JSON 输出
    alias_generator=to_camel,
    # 允许在创建模型时使用别名（如 'taskId'）
    populate_by_name=True,
    # 允许从 ORM 对象等直接转换
    from_attributes=True,
    # 允许任意类型作为字段
    arbitrary_types_allowed=True,
    # 统一处理所有 datetime 对象的 JSON 序列化格式
    json_encoders={datetime.datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")},
)


class BaseModel(pydantic.BaseModel):
    model_config = model_config


__all__ = ["BaseModel", "T"]
