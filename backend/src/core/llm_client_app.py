"""
Lightweight OpenAI-compatible client for the LLM-first MVP.

Notes:
- Kept separate from experimental notebooks code.
- Minimal surface: a single generate_text() with simple retries and env-driven config.
- Expects environment variables: OPENAI_API_KEY, OPENAI_MODEL (optional), OPENAI_BASE_URL (optional).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI


_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_BASE_URL = os.getenv("OPENAI_BASE_URL")  # Optional; useful for proxies/compat layers


class LLMClientApp:
    def __init__(self, model: Optional[str] = None) -> None:
        client_kwargs: Dict[str, Any] = {}
        if _BASE_URL:
            client_kwargs["base_url"] = _BASE_URL
        # API key is read from OPENAI_API_KEY by the SDK
        self.client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
        self.model = model or _DEFAULT_MODEL

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        retries: int = 2,
        backoff_base: float = 0.6,
    ) -> Dict[str, Any]:
        """Call the chat completion API with simple retries.

        Returns a dict with keys: text, usage, model.
        """
        attempt = 0
        last_error: Optional[Exception] = None
        used_model = model or self.model

        while attempt <= retries:
            try:
                resp = self.client.chat.completions.create(
                    model=used_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                usage_dict = {
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                } if usage is not None else {}
                return {"text": text, "usage": usage_dict, "model": used_model}
            except Exception as e:  # Broad catch: MVP-level; refine later
                last_error = e
                if attempt == retries:
                    break
                sleep_s = backoff_base * (2 ** attempt)
                time.sleep(sleep_s)
                attempt += 1
        # If here, all retries failed
        raise RuntimeError(f"LLM generate_text failed after {retries + 1} attempts: {last_error}")


# Convenience singleton for simple usage
_client_singleton: Optional[LLMClientApp] = None

def get_llm_client_app() -> LLMClientApp:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClientApp()
    return _client_singleton
