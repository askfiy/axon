from pydantic import BaseModel

from core.models.http import model_config as base_model_config


class BaseHttpModel(BaseModel):
    model_config = base_model_config
