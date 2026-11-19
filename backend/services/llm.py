"""
LLM provider abstraction — the rest of the code never imports a vendor SDK.

    provider = get_llm_provider(settings)
    if provider.available:
        result = provider.generate(system=..., messages=[...], model=...)

Add another vendor (OpenAI, Gemini, a local model) by writing one more
subclass of LLMProvider and registering it in get_llm_provider().

If no API key is configured, NullLLMProvider is used and the chatbot runs in
fully deterministic "offline" mode (rules + retrieval + reviewed templates).
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure (network, auth, refusal, bad output). Callers fall back to rules."""


class UsageTracker:
    """Per-request token usage, kept thread-local so concurrent requests don't mix."""

    def __init__(self) -> None:
        self._tl = threading.local()

    @property
    def last_usage(self) -> list[dict]:
        if not hasattr(self._tl, "usage"):
            self._tl.usage = []
        return self._tl.usage

    @last_usage.setter
    def last_usage(self, value: list[dict]) -> None:
        self._tl.usage = value


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name = "abstract"
    available = False

    @abstractmethod
    def generate(self, *, system: str, messages: list[dict], model: str,
                 max_tokens: int = 16000, effort: Optional[str] = None) -> LLMResult: ...

    @abstractmethod
    def generate_structured(self, *, system: str, messages: list[dict], model: str,
                            schema: Type[T], max_tokens: int = 8000) -> tuple[T, dict]: ...


class NullLLMProvider(LLMProvider):
    name = "none"
    available = False

    def generate(self, **kwargs) -> LLMResult:
        raise LLMError("No LLM configured (offline mode)")

    def generate_structured(self, **kwargs):
        raise LLMError("No LLM configured (offline mode)")


# Models that support the server-side refusal fallback ("fallbacks": "default")
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
_UNUSABLE_STOPS = {"refusal", "max_tokens", "model_context_window_exceeded"}


class AnthropicProvider(LLMProvider):
    """Claude via the official `anthropic` Python SDK. Key from ANTHROPIC_API_KEY."""
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # imported lazily so offline mode needs no SDK

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=60.0, max_retries=2)
        self.available = True

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage", None)
        return {"input_tokens": getattr(u, "input_tokens", 0), "output_tokens": getattr(u, "output_tokens", 0)}

    def generate(self, *, system, messages, model, max_tokens=16000, effort=None) -> LLMResult:
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if effort:
            kwargs["output_config"] = {"effort": effort}
        try:
            if model in _FALLBACK_MODELS:
                # If a safety classifier declines, the API re-runs the request on a
                # fallback model inside the same call instead of failing.
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                resp = self.client.messages.create(**kwargs)
        except self._anthropic.APIError as e:  # status, connection, timeout errors
            raise LLMError(f"Anthropic API error: {e}") from e
        except Exception as e:  # missing credentials raise TypeError / CredentialsError, not APIError
            raise LLMError(f"Anthropic client error: {type(e).__name__}: {e}") from e
        if resp.stop_reason in _UNUSABLE_STOPS:
            # refusal = declined; max_tokens / context exceeded = cut off mid-answer.
            # A truncated medical answer may drop the warning signs -> never use it.
            raise LLMError(f"Unusable response (stop_reason={resp.stop_reason})")
        blocks = list(resp.content)
        # After a server-side fallback, text written BEFORE the last fallback block
        # belongs to the model that declined — keep only the final model's text.
        last_fb = max((i for i, b in enumerate(blocks) if getattr(b, "type", "") == "fallback"), default=-1)
        text = "".join(b.text for b in blocks[last_fb + 1:] if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError(f"Empty response (stop_reason={resp.stop_reason})")
        return LLMResult(text=text, model=getattr(resp, "model", model), usage=self._usage(resp))

    def generate_structured(self, *, system, messages, model, schema, max_tokens=8000):
        try:
            resp = self.client.messages.parse(
                model=model, max_tokens=max_tokens, system=system, messages=messages, output_format=schema
            )
        except self._anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e
        except Exception as e:  # credentials, validation / parsing problems
            raise LLMError(f"Structured output failed: {type(e).__name__}: {e}") from e
        if resp.stop_reason in _UNUSABLE_STOPS or resp.parsed_output is None:
            raise LLMError(f"No structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output, self._usage(resp)


def get_llm_provider(provider: str) -> LLMProvider:
    """provider: auto | anthropic | none"""
    if provider == "none":
        return NullLLMProvider()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if provider in ("auto", "anthropic") and (has_key or provider == "anthropic"):
        try:
            return AnthropicProvider()
        except ImportError:
            log.warning("`anthropic` package not installed -> offline mode. Run: pip install anthropic")
        except Exception as e:  # pragma: no cover
            log.warning("Could not initialise Anthropic client (%s) -> offline mode", e)
    return NullLLMProvider()
