import re
from typing import Generic, Literal, Any, TypeAlias
from collections.abc import Sequence


import pydantic
from pydantic import Field, computed_field
from pydantic.alias_generators import to_snake

from ..model import BaseModel, T


class BaseHttpResponseModel(BaseModel):
    """
    为 Taxonsk API 设计的、标准化的泛型响应模型。
    """

    code: int = Field(default=200, description="状态码")
    message: str = Field(default="Success", description="响应消息")
    is_failed: bool = Field(default=False, description="是否失败")


class ResponseModel(BaseHttpResponseModel, Generic[T]):
    result: T | None = Field(default=None, description="响应体负载")


PaginationSerializer: TypeAlias = BaseModel


class PaginationRequest(BaseModel):
    """
    分页请求对象
    """

    page: int = Field(default=1, ge=1, description="页码, 从 1 开始")
    size: int = Field(
        default=10, ge=1, le=100, description="单页数量, 最小 1, 最大 100"
    )
    order_by: str | None = Field(
        default=None,
        description="排序字段/方向, 默认按照 id 进行 DESC 排序.",
        examples=["id=asc,createAt=desc", "id"],
    )

    @computed_field
    @property
    def order_by_rule(self) -> list[tuple[str, Literal["asc", "desc"]]]:
        order_by = self.order_by or "id=desc"

        _order_by = [item.strip() for item in order_by.split(",") if item.strip()]
        _struct_order_by: list[tuple[str, Literal["asc", "desc"]]] = []

        for item in _order_by:
            match = re.match(r"([\w_]+)(=(asc|desc))?", item, re.IGNORECASE)
            if match:
                field_name = to_snake(match.group(1))
                order_direction = match.group(3)
                direction: Literal["asc", "desc"] = "desc"
                if order_direction and order_direction.lower() == "asc":
                    direction = "asc"
                _struct_order_by.append((field_name, direction))
            else:
                raise pydantic.ValidationError(f"Invalid order_by format: {item}")

        return _struct_order_by


class PaginationResponse(BaseHttpResponseModel):
    """
    分页响应对象
    """

    current_page: int = Field(default=0, description="当前页")
    current_size: int = Field(default=0, description="当前数")
    total_counts: int = Field(default=0, description="总记录数")
    result: list[Any] = Field(
        default_factory=list, description="所有记录对象"
    )

    @computed_field
    @property
    def total_pages(self) -> int:
        if self.current_size == 0:
            return 0
        return (self.total_counts + self.current_size - 1) // self.current_size


class Paginator(
    BaseModel,
):
    """
    分页器对象
    """

    serializer_cls: type[PaginationSerializer]
    request: PaginationRequest
    response: PaginationResponse = Field(
        default_factory=PaginationResponse, description="分页响应对象"
    )

    def with_serializer_response(
        self, total_counts: int, orm_sequence: Sequence[Any]
    ) -> None:
        self.response.current_page = self.request.page
        self.response.current_size = self.request.size
        self.response.total_counts = total_counts

        self.response.result = [
            self.serializer_cls.model_validate(obj) for obj in orm_sequence
        ]
