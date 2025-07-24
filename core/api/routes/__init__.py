import fastapi

from .tasks import tasks_route


api_router = fastapi.APIRouter()

api_router.include_router(tasks_route)
