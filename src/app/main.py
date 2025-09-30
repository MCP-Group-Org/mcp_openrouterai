from __future__ import annotations
from typing import Any, Dict
from core.settings import get_settings
from fastapi import FastAPI, HTTPException, Request
from app.providers.openrouter_chat import chat_completion

app = FastAPI(title="mcp-openrouterai")
_settings = get_settings()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_url": str(_settings.openrouter_base_url),
        "default_model": _settings.default_model,
        "app_title": _settings.app_title,
    }

# --- Минимальный JSON-RPC обработчик ---
@app.post("/mcp")
async def mcp_entry(req: Request):
    """
    Простой JSON-RPC 2.0 вход: поддерживает метод 'completions.create'.
    Формат запроса:
    {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "completions.create",
      "params": {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role":"user","content":"hi"}],
        ... любые опции OpenAI chat.completions.create
      }
    }
    """
    try:
        payload: Dict[str, Any] = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    jsonrpc = payload.get("jsonrpc")
    rpc_id = payload.get("id")
    method = payload.get("method")
    params: Dict[str, Any] = payload.get("params") or {}

    if jsonrpc != "2.0":
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32600, "message": "Invalid Request"}}

    if method != "completions.create":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": "Method not found", "data": {"method": method}},
        }

    # Извлекаем параметры для чата
    model = params.pop("model", None)  # если None — возьмем из Settings в провайдере
    messages = params.pop("messages", None)
    if not messages:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32602, "message": "Missing 'messages' in params"},
        }

    try:
        # Пробрасываем оставшиеся параметры как kwargs (temperature, max_tokens и т.д.)
        resp = chat_completion(messages=messages, model=model, **params)
        # Возвращаем как result. Можно вернуть сырой объект resp.model_dump() при необходимости.
        result = {
            "id": getattr(resp, "id", None),
            "model": getattr(resp, "model", None),
            "choices": [
                {
                    "index": ch.index,
                    "message": {"role": ch.message.role, "content": ch.message.content},
                    "finish_reason": getattr(ch, "finish_reason", None),
                }
                for ch in getattr(resp, "choices", []) or []
            ],
            "usage": getattr(resp, "usage", None).__dict__ if getattr(resp, "usage", None) else None,
        }
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32000, "message": "Upstream error", "data": str(e)}}
