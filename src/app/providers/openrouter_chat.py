from __future__ import annotations
from typing import Any, Iterable, Mapping

from core.settings import get_settings
from .openrouter_client import build_openrouter_client

_settings = get_settings()
_client = build_openrouter_client()


def chat_completion(
    messages: Iterable[Mapping[str, Any]],
    *,
    model: str | None = None,
    **kwargs: Any,
):
    """
    Узкий слой для Chat Completions через OpenRouter.
    Совместим с вашим внутренним контрактом провайдера.
    Никаких os.getenv — все значения берутся из Settings.
    """
    selected_model = model or _settings.default_model

    # При необходимости автоматически прокинем include_reasoning из настроек
    if _settings.include_reasoning is not None and "include_reasoning" not in kwargs:
        kwargs["include_reasoning"] = _settings.include_reasoning

    return _client.chat.completions.create(
        model=selected_model,
        messages=list(messages),  # на случай, если пришёл генератор
        **kwargs,
    )
