"""Local Ollama service and model health checks."""

from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import urlopen


def check_ollama(base_url: str, chat_model: str, embedding_model: str) -> tuple[bool, str]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return False, f"Cannot reach Ollama at {base_url}: {exc}"
    names = {model.get("name", "") for model in payload.get("models", [])}
    missing = [
        model
        for model in (chat_model, embedding_model)
        if not any(name == model or name.startswith(f"{model}:") for name in names)
    ]
    if missing:
        return False, f"Ollama is running, but these models are missing: {', '.join(missing)}"
    return True, "Ollama is running and both required models are available."
