import logging
import asyncio
from typing import Any, TypeAlias
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone

import asyncio_atexit
import redis.asyncio as redis
from redis.typing import FieldT, EncodableT
from redis.exceptions import ResponseError
from pydantic import BaseModel, Field

RbokerMessage: TypeAlias = Any


class RbokerPayloadMetadata(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RbokerPayloadExcInfo(BaseModel):
    message: str
    type: str
    failed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RbokerPayload(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: RbokerMessage
    exc_info: RbokerPayloadExcInfo | None = Field(default=None)


class RBoker:
    """
    基于 Redis Streams 实现的发布订阅系统
    """

    def __init__(self, redis_client: redis.Redis):
        self._client = redis_client
        self._stop = False
        self._consumer_tasks: list[asyncio.Task[None]] = []

    async def _callback_ack(
        self,
        topic: str,
        group_id: str,
        message_id: str,
        rboker_message: RbokerPayload,
        callback: Callable[[RbokerMessage], Coroutine[Any, Any, None]],
    ):
        try:
            await callback(rboker_message.content)
        except Exception as exc:
            rboker_message.exc_info = RbokerPayloadExcInfo(
                message=str(exc), type=exc.__class__.__name__
            )
            print(rboker_message.model_dump_json())
            # 放入死信队列. 后续可通过消费该死信队列获得新的讯息
            await self._client.xadd(
                f"{topic}-dlq",
                {"message": rboker_message.model_dump_json()},
                maxlen=1000,
            )
            logging.error(
                f"Error in background task for message {message_id}: {exc}",
                exc_info=True,
            )
        finally:
            await self._client.xack(topic, group_id, message_id)

    async def _consume_worker(
        self,
        topic: str,
        group_id: str,
        consumer_name: str,
        callback: Callable[[RbokerMessage], Coroutine[Any, Any, None]],
    ):
        while True:
            try:
                # xreadgroup 会阻塞，但只会阻塞当前这一个任务，不会影响其他任务
                # block 0 一直阻塞
                response = await self._client.xreadgroup(
                    group_id, consumer_name, {topic: ">"}, count=1, block=0
                )
                if not response:
                    continue

                stream_key, messages = response[0]
                message_id, data = messages[0]

                try:
                    rboker_message = RbokerPayload.model_validate_json(data["message"])

                    asyncio.create_task(
                        self._callback_ack(
                            topic=topic,
                            group_id=group_id,
                            message_id=message_id,
                            rboker_message=rboker_message,
                            callback=callback,
                        )
                    )

                except Exception as e:
                    logging.error(
                        f"Error processing message {message_id.decode()}: {e}",
                        exc_info=True,
                    )

            except asyncio.CancelledError:
                logging.info(f"Consumer '{consumer_name}' is shutting down.")
                break

            except Exception as e:
                logging.error(
                    f"Consumer '{consumer_name}' loop error: {e}", exc_info=True
                )
                await asyncio.sleep(5)

    async def send(self, topic: str, message: RbokerMessage) -> str:
        rboker_message = RbokerPayload(content=message)

        message_payload: dict[FieldT, EncodableT] = {
            "message": rboker_message.model_dump_json()
        }
        message_id = await self._client.xadd(topic, message_payload)
        logging.info(f"Sent message {message_id} to topic '{topic}'")
        return message_id

    async def consumer(
        self,
        topic: str,
        group_id: str,
        callback: Callable[[RbokerMessage], Coroutine[Any, Any, None]],
        count: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        """
        创建并启动消费者后台任务。
        """
        try:
            await self._client.xgroup_create(topic, group_id, mkstream=True)
            logging.info(f"Consumer group '{group_id}' created for topic '{topic}'.")
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        for i in range(count):
            consumer_name = f"{group_id}-consumer-{i + 1}"
            task = asyncio.create_task(
                self._consume_worker(topic, group_id, consumer_name, callback)
            )
            self._consumer_tasks.append(task)
            logging.info(f"Started consumer task '{consumer_name}' on topic '{topic}'.")

        if not self._stop:
            asyncio_atexit.register(self.shutdown)  # pyright: ignore[reportUnknownMemberType]
            self._stop = True

    async def shutdown(self):
        logging.info("Shutting down consumer tasks...")

        for task in self._consumer_tasks:
            task.cancel()

        await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        logging.info("All consumer tasks have been shut down.")

        self._client.close()


if __name__ == "__main__":
    redis_client = redis.from_url("redis://127.0.0.1:6379", decode_responses=True)  # pyright: ignore[reportUnknownMemberType]

    async def tester():
        topic = "Test Topic"
        group = "Test Group"

        async def handle_message(message: RbokerMessage):
            print(message)
            raise RuntimeError("Exc")

        boker = RBoker(redis_client=redis_client)
        await boker.consumer(
            topic=topic, group_id=group, callback=handle_message, count=5
        )

        count = 1
        import uuid

        while True:
            await boker.send(topic=topic, message=f"Hi {count}: {uuid.uuid4()}")
            count += 1
            await asyncio.sleep(0.5)

    asyncio.run(tester())
