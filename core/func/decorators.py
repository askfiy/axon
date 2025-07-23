from typing import Callable, Any, TypeVar, cast
from collections.abc import Awaitable
from functools import wraps
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")
R = TypeVar("R", bound=Awaitable[Any])


def transactional(
    func: Callable[..., R],
) -> Callable[..., R]:
    """
    安全的自动提交回滚事务.
    """

    @wraps(func)
    async def wrapper(session: AsyncSession | Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(session, AsyncSession):
            raise TypeError("添加了自动事务的业务层函数. 第一个参数必须是 session.")

        try:
            result = await func(session, *args, **kwargs)
            await session.commit()
            return result
        except Exception as exc:
            await session.rollback()
            raise exc

    return cast(Callable[..., R], wrapper)
