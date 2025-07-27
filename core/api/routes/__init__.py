import fastapi

from .tasks import tasks_route
from .tasks_chat import tasks_chat_route
from .tasks_history import tasks_history_route


api_router = fastapi.APIRouter()

tasks_route.include_router(tasks_chat_route)
tasks_route.include_router(tasks_history_route)


api_router.include_router(tasks_route)
