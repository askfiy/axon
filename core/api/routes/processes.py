import fastapi

from core.models.http import ResponseModel

processes_route = fastapi.APIRouter(prefix="/processes", tags=["Processes"])


# TODO: 返回任务模型
@processes_route.post(
    path="/on-create",
    name="创建任务流程",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[bool],
)
async def create() -> ResponseModel[bool]:
    return ResponseModel(result=True)


@processes_route.post(
    path="/on-update",
    name="更新任务流程",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[bool],
)
async def update() -> ResponseModel[bool]:
    return ResponseModel(result=True)


@processes_route.post(
    path="/on-execute",
    name="执行任务流程",
    status_code=fastapi.status.HTTP_200_OK,
    response_model=ResponseModel[bool],
)
async def get() -> ResponseModel[bool]:
    return ResponseModel(result=True)
