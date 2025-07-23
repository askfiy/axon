from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from core.config import env


engine = create_async_engine(
    env.ASYNC_DB_URL,
    # echo=True,
)

AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,  # 明确指定使用 AsyncSession
)


async def get_async_session():
    async with AsyncSessionLocal(bind=engine) as session:
        yield session


# 导出你想要在其他模块中直接使用的内容
__all__ = [
    "get_async_session",
    "engine",
]
