import fastapi

from .tasks import tasks_route
from .processes import processes_route
from .schedules import schedules_route


api_router = fastapi.APIRouter()

api_router.include_router(tasks_route)
api_router.include_router(processes_route)
api_router.include_router(schedules_route)
