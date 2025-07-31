import logging
import asyncio
from typing import Any, TypeAlias
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone

import redis.asyncio as redis
from redis.typing import FieldT, EncodableT
from redis.exceptions import ResponseError
from pydantic import BaseModel, Field

RbrokerMessage: TypeAlias = Any


class RbrokerPayloadMetadata(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RbrokerPayloadExcInfo(BaseModel):
    message: str
    type: str
    failed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RbrokerPayload(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: RbrokerMessage
    exc_info: RbrokerPayloadExcInfo | None = Field(default=None)


class RBroker:
    """
    基于 Redis Streams 实现的发布订阅系统
    """

    def __init__(self, redis_client: redis.Redis):
        self._client = redis_client
        self._consumer_tasks: list[asyncio.Task[None]] = []
        self._dlq_maxlen = 1000

    async def _handle_callback_ack(
        self,
        topic: str,
        group_id: str,
        message_id: str,
        rbroker_message: RbrokerPayload,
        callback: Callable[[RbrokerMessage], Coroutine[Any, Any, None]],
    ):
        try:
            await callback(rbroker_message.content)
        except Exception as exc:
            rbroker_message.exc_info = RbrokerPayloadExcInfo(
                message=str(exc), type=exc.__class__.__name__
            )
            # 放入死信队列. 后续可通过消费该死信队列获得新的讯息
            await self._client.xadd(
                f"{topic}-dlq",
                {"message": rbroker_message.model_dump_json()},
                maxlen=self._dlq_maxlen,
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
        callback: Callable[[RbrokerMessage], Coroutine[Any, Any, None]],
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
                    rbroker_message = RbrokerPayload.model_validate_json(
                        data["message"]
                    )

                    asyncio.create_task(
                        self._handle_callback_ack(
                            topic=topic,
                            group_id=group_id,
                            message_id=message_id,
                            rbroker_message=rbroker_message,
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

    async def send(self, topic: str, message: RbrokerMessage) -> str:
        rbroker_message = RbrokerPayload(content=message)

        message_payload: dict[FieldT, EncodableT] = {
            "message": rbroker_message.model_dump_json()
        }
        message_id = await self._client.xadd(topic, message_payload)
        logging.info(f"Sent message {message_id} to topic '{topic}'")
        return message_id

    async def consumer(
        self,
        topic: str,
        callback: Callable[[RbrokerMessage], Coroutine[Any, Any, None]],
        group_id: str | None = None,
        count: int = 1,
        *args: Any,
        **kwargs: Any,
    ):
        """
        创建并启动消费者后台任务。
        """
        group_id = group_id or topic + "_group"

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

    async def shutdown(self):
        logging.info("Shutting down consumer tasks...")

        for task in self._consumer_tasks:
            task.cancel()

        await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        logging.info("All consumer tasks have been shut down.")
