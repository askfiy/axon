import json
from typing import Any

import redis.asyncio as redis


class RCache:
    """
    基于 Redis 实现的 Simple 缓存系统
    """

    def __init__(self, redis_client: redis.Redis):
        self._client = redis_client

    async def has(self, key: str) -> bool:
        return await self._client.exists(key) > 0

    async def get(self, key: str, default: Any = None) -> Any:
        value = await self._client.get(key)

        if value is None:
            return default

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not ttl and await self._client.exists(key):
            current_ttl = await self._client.ttl(key)
            if current_ttl == -1:
                raise ValueError(
                    f"Key '{key}' exists without TTL. Refusing to set without TTL."
                )

        if isinstance(value, (dict, list)):
            value = json.dumps(value)

        await self._client.set(key, value, ex=ttl)

    async def delete(self, key: str) -> int:
        return await self._client.delete(key)

    async def persist(self, key: str) -> bool:
        current_ttl = await self._client.ttl(key)

        if current_ttl == -1:
            raise ValueError(f"Key '{key}' has no TTL. Cannot persist.")

        return await self._client.persist(key)
