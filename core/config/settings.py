import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import MySQLDsn, field_validator


configure_path = os.path.join(".", ".env", ".local.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, env_file=configure_path)

    SYNC_DB_URL: str
    ASYNC_DB_URL: str
    OPENAI_API_KEY: str

    @field_validator("SYNC_DB_URL", "ASYNC_DB_URL", mode="before")
    @classmethod
    def _validate_db_url(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise TypeError("Database URL must be a string")
        try:
            # 验证是否符合 MySQLDsn 类型.
            MySQLDsn(v)
        except Exception as e:
            raise ValueError(f"Invalid MySQL DSN: {e}") from e

        return str(v)
