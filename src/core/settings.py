from __future__ import annotations
from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- OpenRouter ---
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: HttpUrl = Field(  # строго валидный URL
        "https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    app_public_url: Optional[HttpUrl] = Field(None, alias="APP_PUBLIC_URL")
    app_title: str = Field("mcp-openrouterai", alias="APP_TITLE")
    default_model: str = Field("openai/gpt-4o-mini", alias="DEFAULT_MODEL")

    # --- Сервис ---
    port: int = Field(8000, alias="PORT")

    # --- Доп. опции (пример) ---
    # Включать ли расширенный reasoning у моделей, которые его поддерживают
    include_reasoning: Optional[Literal[True, False]] = Field(None, alias="INCLUDE_REASONING")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("openrouter_api_key")
    @classmethod
    def _check_api_key(cls, v: str) -> str:
        # Не жёсткая проверка, но даст раннее предупреждение при ошибках
        if not v or len(v) < 20:
            raise ValueError("OPENROUTER_API_KEY выглядит некорректным")
        return v

    @field_validator("app_title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        return v.strip() or "mcp-openrouterai"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # читается из .env + окружения
