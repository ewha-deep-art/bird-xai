"""Application settings for Bird XAI server.

Env prefix: ``BIRD_XAI_`` (e.g. ``BIRD_XAI_WISH_FLUSH_THRESHOLD``, ``BIRD_XAI_WISH_RATE_LIMIT_SEC``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, validation_alias=AliasChoices("BIRD_XAI_PORT", "PORT"))
    frame_interval: float = 1.0
    wish_rate_limit_sec: float = 5.0
    wish_flush_threshold: int = 10

    model_config = SettingsConfigDict(env_prefix="BIRD_XAI_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
