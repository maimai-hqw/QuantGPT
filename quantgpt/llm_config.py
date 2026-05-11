"""Provider-neutral LLM config.

Reads generic LLM_* env vars first, falls back to legacy OPENAI_* for
backward compatibility. Use this everywhere the project talks to an
OpenAI-compatible chat completion API (DeepSeek, Anthropic-compat proxies,
local vLLM/Ollama, etc.).
"""

import os


def get_llm_config() -> dict:
    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower()
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = (
        os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    )
    return {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}
