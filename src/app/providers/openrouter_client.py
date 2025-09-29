from __future__ import annotations
from openai import OpenAI

# импорт из src-пакета; предполагается, что PYTHONPATH указывает на app/src
from core.settings import get_settings


def build_openrouter_client() -> OpenAI:
    s = get_settings()

    # Заголовки атрибуции — опциональны, но полезны для трекинга вашего приложения в OpenRouter
    default_headers = {
        "HTTP-Referer": str(s.app_public_url) if s.app_public_url else "https://example.com",
        "X-Title": s.app_title,
    }

    return OpenAI(
        api_key=s.openrouter_api_key,
        base_url=str(s.openrouter_base_url),
        default_headers=default_headers,
    )
