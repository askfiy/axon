import datetime
from typing import Generic, TypeVar
from collections.abc import Sequence


import pydantic
from pydantic import BaseModel, Field, computed_field
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


class BaseHttpModel(BaseModel):
    model_config = model_config


class BaseHttpResponseModel(BaseHttpModel, Generic[T]):
    """
    为 Axon API 设计的、标准化的泛型响应模型。
    """

    code: int = Field(default=200, description="状态码")
    message: str = Field(default="Success", description="响应消息")
    is_failed: bool = Field(default=False, description="是否失败")


class ResponseModel(BaseHttpResponseModel[T]):
    result: T | None = Field(default=None, description="响应体负载")


class PageinationRequest(BaseHttpModel):
    """
    分页器请求对象
    """

    page: int = Field(default=1, ge=1, description="页码, 从 1 开始")
    size: int = Field(
        default=10, ge=1, le=100, description="单页数量, 最小 1, 最大 100"
    )


class PageinationResponse(BaseHttpResponseModel[T]):
    """
    分页器响应对象
    """

    current_page: int = Field(description="当前页")
    current_size: int = Field(description="当前数")
    total_counts: int = Field(description="总记录数")
    result: list[T] = Field(default_factory=list, description="所有记录对象")

    @computed_field
    @property
    def total_pages(self) -> int:
        if self.current_size == 0:
            return 0
        return (self.total_counts + self.current_size - 1) // self.current_size
