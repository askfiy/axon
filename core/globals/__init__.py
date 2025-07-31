from core.database.connection import get_redis_client
from core.components import RBroker, RCache

_redis_client = get_redis_client()

broker = RBroker(redis_client=_redis_client)
cache = RCache(redis_client=_redis_client)

__all__ = ["broker", "cache"]
