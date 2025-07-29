import os
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, MySQLDsn, RedisDsn, field_validator


configure_path = os.path.join(".", ".env", ".local.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True, env_file=configure_path)

    SYNC_DB_URL: str = Field(examples=["mysql+pymysql://root:123@127.0.0.1:3306/db1"])
    ASYNC_DB_URL: str = Field(examples=["mysql+asyncmy://root:123@127.0.0.1:3306/db1"])
    OPENAI_API_KEY: str = Field(examples=["sk-proj-..."])
    ASYNC_REDIS_URL: str = Field(examples=[""])

    @field_validator("SYNC_DB_URL", "ASYNC_DB_URL", mode="before")
    @classmethod
    def _validate_db_url(cls, db_url: Any) -> str:
        if not isinstance(db_url, str):
            raise TypeError("Database URL must be a string")
        try:
            # 验证是否符合 MySQLDsn 类型.
            MySQLDsn(db_url)
        except Exception as e:
            raise ValueError(f"Invalid MySQL DSN: {e}") from e

        return str(db_url)

    @field_validator("ASYNC_REDIS_URL", mode="before")
    @classmethod
    def _validate_redis_url(cls, redis_url: Any) -> str:
        if not isinstance(redis_url, str):
            raise TypeError("Redis URL must be a string")
        try:
            # 验证是否符合 RedisDsn 类型.
            MySQLDsn(redis_url)
        except Exception as e:
            raise ValueError(f"Invalid redis_url DSN: {e}") from e

        return str(redis_url)
