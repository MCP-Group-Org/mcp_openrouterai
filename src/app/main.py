# app/main.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request

from core.settings import get_settings
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


# ---------------- JSON-RPC helpers ----------------

def _err(rpc_id: Any, code: int, message: str, data: Any | None = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": err}


def _ok(rpc_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _format_chat_result(resp: Any) -> Dict[str, Any]:
    choices = []
    for ch in (getattr(resp, "choices", None) or []):
        msg = getattr(ch, "message", None)
        choices.append({
            "index": getattr(ch, "index", None),
            "message": {
                "role": getattr(msg, "role", None) if msg else None,
                "content": getattr(msg, "content", None) if msg else None,
            },
            "finish_reason": getattr(ch, "finish_reason", None),
        })
    usage = getattr(resp, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = getattr(usage, "model_dump", None) and usage.model_dump() or getattr(usage, "__dict__", None)
    return {
        "id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None),
        "choices": choices,
        "usage": usage_dict,
    }


def _do_chat(arguments: Dict[str, Any]) -> Dict[str, Any]:
    # совместимо со схемой OpenAI Chat Completions
    model: Optional[str] = arguments.pop("model", None)
    messages: Optional[List[Dict[str, Any]]] = arguments.pop("messages", None)
    if not messages:
        raise ValueError("Missing 'messages' in arguments")
    resp = chat_completion(messages=messages, model=model, **arguments)
    return _format_chat_result(resp)


# ---------------- MCP JSON-RPC entry ----------------

@app.post("/mcp")
async def mcp_entry(req: Request):
    """
    MCP JSON-RPC 2.0:
      - tools/list: объявляет инструмент 'chat'
      - tools/call: выполняет инструмент по имени (поддерживается 'chat')
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
        return _err(rpc_id, -32600, "Invalid Request")

    try:
        if method == "tools/list":
            return _ok(rpc_id, {
                "tools": [
                    {
                        "name": "chat",
                        "description": "Chat completion over OpenRouter (OpenAI-compatible schema).",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string", "description": "Optional model id"},
                                "messages": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "role": {"type": "string"},
                                            "content": {"type": ["string", "array", "object"]},
                                        },
                                        "required": ["role", "content"],
                                    },
                                },
                                "temperature": {"type": ["number", "null"]},
                                "max_tokens": {"type": ["integer", "null"]},
                            },
                            "required": ["messages"],
                            "additionalProperties": True,
                        },
                    }
                ]
            })

        if method == "tools/call":
            name = params.get("name")
            arguments: Dict[str, Any] = params.get("arguments") or {}
            if not name:
                return _err(rpc_id, -32602, "Missing 'name' in params")
            if name != "chat":
                return _err(rpc_id, -32601, "Tool not found", {"name": name})
            result = _do_chat(arguments)
            return _ok(rpc_id, result)

        return _err(rpc_id, -32601, "Method not found", {"method": method})
    except ValueError as ve:
        return _err(rpc_id, -32602, "Invalid params", str(ve))
    except Exception as e:
        return _err(rpc_id, -32000, "Upstream error", str(e))
