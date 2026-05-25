"""Application settings for Bird XAI server."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    frame_interval: float = 1.0

    model_config = SettingsConfigDict(env_prefix="BIRD_XAI_")


def get_settings() -> Settings:
    return Settings()
