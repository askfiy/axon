import asyncio

from core.globals import broker
from core.services.tasks import get_dispatch_tasks_id, get_dispatch_task_by_id


class Dispatch:
    TOPIC = "Task-Scheduler"

    @classmethod
    async def production(cls):
        while True:
            tasks_id = await get_dispatch_tasks_id()

            for task_id in tasks_id:
                await broker.send(topic=cls.TOPIC, message={"task_id": task_id})

            await asyncio.sleep(60)

    @classmethod
    async def consumption(cls, message: dict[str, int]):
        task_id = message["task_id"]
        task = await get_dispatch_task_by_id(task_id)
        print(
            f"消费啦: {task.id} {task.name} {task.histories} {task.chats} {task.metadata_info.keywords}"
        )

    @classmethod
    async def forever(cls):
        asyncio.create_task(cls.production())
        await broker.consumer(topic=cls.TOPIC, callback=cls.consumption, count=5)

    @classmethod
    async def shutdown(cls):
        await broker.shutdown()
