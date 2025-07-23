import fastapi

from core.models.http import ResponseModel

schedules_route = fastapi.APIRouter(prefix="/schedules", tags=["Schedules"])


@schedules_route.post(
    path="/call-soon-task",
    name="尝试立即执行任务",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[bool],
)
async def call_soon_task(
    task_id: str = fastapi.Query(description="任务 ID"),
) -> ResponseModel[bool]:
    return ResponseModel(result=True)
