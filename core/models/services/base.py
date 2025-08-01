from typing import Generic

from collections.abc import Sequence
from pydantic import Field

from ..model import BaseModel, T


class Paginator(BaseModel, Generic[T]):
    """
    分页响应信息
    """

    current_page: int = Field(description="当前页")
    current_size: int = Field(description="当前数")
    total_counts: int = Field(description="总记录数")
    db_objects: Sequence[T] = Field(
        default_factory=Sequence, description="所有记录对象"
    )
