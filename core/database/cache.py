import json
from typing import Any
import redis.asyncio as redis

import asyncio_atexit

# from core.config import env_helper

url = "redis://localhost:6379"

pool = redis.ConnectionPool.from_url(url, decode_responses=True)


class Cacher:
    """一个封装了 redis.asyncio 客户端的缓存工具类"""

    def __init__(self):
        """初始化客户端并注册异步退出事件，以确保连接被优雅关闭。"""
        self.client = redis.Redis.from_pool(pool)
        # atexit 钩子确保在程序退出时，连接池和客户端被正确关闭
        # 只需要关闭连接池即可，它会管理所有从它创建的客户端连接
        asyncio_atexit.register(pool.aclose)

    async def set(
        self, key: str, value: Any, ttl: float, json_dumps: bool = True
    ) -> None:
        """
        设置一个键值对到缓存中，并指定过期时间。

        :param key: 缓存键 (str)
        :param value: 缓存值 (Any)
        :param ttl: 过期时间，单位为秒 (float)
        :param json_dumps: 是否需要将 value 序列化为 JSON 字符串 (bool)
        """
        if json_dumps:
            value = json.dumps(value)

        # redis-py 的 set 命令使用 `ex` 参数来设置秒级过期时间
        await self.client.set(key, value, ex=int(ttl))

    async def get(self, key: str, default: Any = None, json_loads: bool = True) -> Any:
        """
        根据键从缓存中获取值。

        :param key: 缓存键 (str)
        :param default: 如果键不存在时返回的默认值 (Any)
        :param json_loads: 是否需要将获取到的值从 JSON 字符串反序列化 (bool)
        :return: 缓存中的值，或默认值 (Any)
        """
        value = await self.client.get(key)

        if value is None:
            return default

        if json_loads:
            # pool 中设置 decode_responses=True 后，value 会是 str，可以直接 loads
            return json.loads(value)

        return value

    async def has(self, key: str) -> bool:
        """
        检查指定的键是否存在于缓存中。

        :param key: 缓存键 (str)
        :return: 如果存在则返回 True, 否则返回 False (bool)
        """
        # .exists() 返回存在的键的数量 (0 或 1)，转换为布尔值即可
        return bool(await self.client.exists(key))

    async def add_ttl(self, key: str, ttl: float) -> bool:
        """
        为一个已存在的键设置或更新过期时间。

        :param key: 缓存键 (str)
        :param ttl: 新的过期时间，单位为秒 (float)
        :return: 如果成功设置了过期时间则返回 True, 如果键不存在则返回 False (bool)
        """
        # .expire() 返回 1 (成功) 或 0 (键不存在)
        return bool(await self.client.expire(key, int(ttl)))

    async def del_ttl(self, key: str) -> bool:
        """
        移除一个键的过期时间，使其永久存在（直到被手动删除）。

        :param key: 缓存键 (str)
        :return: 如果成功移除了过期时间则返回 True, 如果键不存在或没有设置过期时间则返回 False (bool)
        """
        return bool(await self.client.persist(key))

    async def delete(self, key: str) -> bool:
        """

        :param key: 缓存键 (str)
        :return: 成功删除的键的数量是否大于0 (bool)
        """
        # .delete() 返回成功删除的键的数量
        return bool(await self.client.delete(key))


class RedisQueue:
    """
    一个接口与您提供的Kafka类完全一致的、基于Redis Streams的可靠队列。
    """

    def __init__(self, address: str):
        """
        :param address: Redis连接URL, e.g., "redis://localhost:6379"
        """
        self.address = address
        self.pool = redis.ConnectionPool.from_url(self.address, decode_responses=True)

        # 生产者（使用属性进行懒加载）
        self._producer: Optional[redis.Redis] = None

        # 存储后台运行的消费者任务
        self.consumer_tasks: Dict[str, List[asyncio.Task]] = {}

    @property
    def producer(self) -> redis.Redis:
        """懒加载生产者，确保只初始化一次并注册关闭钩子。"""
        if self._producer is None:
            self._producer = redis.Redis.from_pool(self.pool)
            # 确保程序退出时，连接池被关闭 (只需要注册一次)
            asyncio_atexit.register(self.pool.aclose)
        return self._producer

    async def send(self, topic: str, data: typing.Any):
        """
        向指定的topic（Stream）发布一条消息。
        """
        # 注意：我们不再需要每次都启动生产者
        message_body = {
            "data": json.dumps(data)  # 将整个数据作为JSON字符串存入'data'字段
        }
        try:
            await self.producer.xadd(topic, message_body)
        except Exception as exc:
            logging.error(
                f"发送 Redis 消息失败, topic: {topic}, data: {data}, exc: {exc}"
            )

    async def _consume_loop(
        self,
        topic: str,
        group_id: str,
        consumer_name: str,
        callback: Callable[..., Awaitable],
        *args,
        **kwargs,
    ):
        """单个消费者协程的无限循环，这是实际的工作单元。"""
        # 为这个循环创建一个独立的客户端
        client = redis.Redis.from_pool(self.pool)

        while True:
            try:
                response = await client.xreadgroup(
                    group_id, consumer_name, {topic: ">"}, count=1, block=0
                )
                if not response:
                    continue

                message_id, message_data = response[0][1][0]
                # 从 'data' 字段中解析出原始数据
                original_data = json.loads(message_data["data"])

                try:
                    # 调用用户提供的回调函数来处理消息
                    await callback(original_data, *args, **kwargs)
                    # 如果回调成功，则发送ACK确认
                    await client.xack(topic, group_id, message_id)
                except Exception as exc:
                    logger.error(f"处理消息 {message_id} 失败: {exc}", exc_info=True)
            except Exception as exc:
                logger.error(f"消费者 '{consumer_name}' 循环异常: {exc}", exc_info=True)
                await asyncio.sleep(5)

    async def consumer(
        self,
        topic: str,
        group_id: str,
        callback: Callable[..., Awaitable],
        count: int,
        *args,
        **kwargs,
    ):
        """
        为指定的topic启动并运行指定数量的消费者。
        这个方法会一直运行，直到所有消费者任务被取消。
        """
        # 确保Stream和消费者组存在
        try:
            await self.producer.xgroup_create(topic, group_id, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        if topic not in self.consumer_tasks:
            self.consumer_tasks[topic] = []

        # 启动指定数量的并发消费者任务
        for i in range(count):
            consumer_name = f"{group_id}-consumer-{i + 1}"
            task = asyncio.create_task(
                self._consume_loop(
                    topic, group_id, consumer_name, callback, *args, **kwargs
                )
            )
            self.consumer_tasks[topic].append(task)

        logging.info(f"已为主题 '{topic}' 启动 {count} 个消费者。")

        # 等待所有消费者任务完成（在实践中，这意味着永远运行，直到被取消）
        await asyncio.gather(*self.consumer_tasks[topic])
