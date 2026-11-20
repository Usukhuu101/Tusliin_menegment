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

    def model_for(self, requested: str) -> str:
        """The model that will really answer when an agent asks for `requested`."""
        return requested

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


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-flash-lite-latest"     # alias; checked against /models at start-up
_SKIP_MODEL_WORDS = ("preview", "exp", "tts", "image", "live", "audio", "embedding", "vision", "learnlm", "robotics")


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible Chat Completions API — Google Gemini (free tier),
    Groq, OpenRouter, a local Ollama server ... Plain HTTP (urllib): no extra SDK.

    The agents ask for Claude model ids (RESPONSE_MODEL etc.); those are replaced
    by LLM_MODEL, or — if LLM_MODEL is not set — by a model picked from the
    provider's /models list (Gemini: the newest Flash-Lite = highest free quota).
    """

    def __init__(self, name: str, base_url: str, api_key: str = "", model: Optional[str] = None,
                 prefer: tuple[str, ...] = (), default_model: str = "", timeout: float = 60.0,
                 max_output_tokens: int = 8192) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.model = model or self._discover_model(prefer, default_model)
        self.available = True
        log.info("LLM provider %s -> model %s", name, self.model)

    def model_for(self, requested: str) -> str:
        return self.model if (not requested or requested.startswith("claude-")) else requested

    # ------------------------------------------------------------- HTTP
    def _request(self, path: str, body: Optional[dict] = None, retries: int = 1) -> dict:
        import json
        import time
        import urllib.error
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.base_url + path, data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2.0)                      # free tiers: a few requests per minute
                    continue
                raise LLMError(f"{self.name} HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise LLMError(f"{self.name} connection error: {e}") from e
            except ValueError as e:
                raise LLMError(f"{self.name} returned invalid JSON: {e}") from e
        raise LLMError(f"{self.name}: retries exhausted")          # pragma: no cover

    def _discover_model(self, prefer: tuple[str, ...], default_model: str) -> str:
        """Pick a model the key can actually use. Never raises: falls back to the default."""
        import re

        try:
            ids = [str(m.get("id", "")).removeprefix("models/") for m in self._request("/models", retries=0).get("data", [])]
        except LLMError as e:
            log.warning("%s: could not list models (%s) -> using %s", self.name, e, default_model)
            return default_model
        if default_model in ids:
            return default_model

        def version(mid: str) -> tuple:
            return tuple(float(x) for x in re.findall(r"\d+(?:\.\d+)?", mid)) or (0.0,)

        for word in prefer:
            cands = [m for m in ids if word in m and not any(s in m for s in _SKIP_MODEL_WORDS)]
            if cands:
                return max(cands, key=version)
        return default_model or (ids[0] if ids else "")

    @staticmethod
    def _usage(data: dict, model: str) -> dict:
        u = data.get("usage") or {}
        return {"model": model, "input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}

    def _chat(self, system: str, messages: list[dict], model: str, max_tokens: int, json_mode: bool) -> tuple[str, dict]:
        m = self.model_for(model)
        body = {"model": m, "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": min(max_tokens, self.max_output_tokens)}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            data = self._request("/chat/completions", body)
        except LLMError as e:
            if not (json_mode and "HTTP 400" in str(e)):
                raise
            body.pop("response_format")              # provider without JSON mode: ask in the prompt only
            data = self._request("/chat/completions", body)
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.name}: no choices in response") from e
        finish = choice.get("finish_reason")
        if finish in ("length", "content_filter", "safety", "recitation"):
            # cut off or blocked -> a medical answer may have lost its warning signs
            raise LLMError(f"Unusable response (finish_reason={finish})")
        text = ((choice.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError(f"Empty response (finish_reason={finish})")
        return text, self._usage(data, data.get("model") or m)

    # -------------------------------------------------------------- API
    def generate(self, *, system, messages, model, max_tokens=16000, effort=None) -> LLMResult:
        text, usage = self._chat(system, messages, model, max_tokens, json_mode=False)
        return LLMResult(text=text, model=usage["model"], usage={k: v for k, v in usage.items() if k != "model"})

    def generate_structured(self, *, system, messages, model, schema, max_tokens=8000):
        import json

        instruction = ("\n\nReturn ONLY one JSON object — no markdown, no explanation — that matches this JSON Schema:\n"
                       + json.dumps(schema.model_json_schema(), ensure_ascii=False))
        text, usage = self._chat(system + instruction, messages, model, max_tokens, json_mode=True)
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        start, end = raw.find("{"), raw.rfind("}")
        try:
            return schema.model_validate_json(raw[start:end + 1] if start >= 0 else raw), usage
        except Exception as e:  # pydantic ValidationError / JSON errors
            raise LLMError(f"Structured output did not match {schema.__name__}: {e}") from e


def _real_key(name: str) -> str:
    """Value of an API-key variable, ignoring the placeholders from .env.example
    ("sk-ant-...") — an uncommented example line must not count as a key."""
    v = (os.environ.get(name) or "").strip()
    return "" if (not v or "..." in v or v.lower() in {"your-key", "changeme", "xxx"}) else v


def get_llm_provider(provider: str) -> LLMProvider:
    """provider: auto | anthropic | gemini | openai_compat | none

    auto = Claude if ANTHROPIC_API_KEY is set, else Gemini if GEMINI_API_KEY is set,
           else offline. LLM_MODEL overrides the model of the non-Claude providers."""
    if provider == "none":
        return NullLLMProvider()
    has_key = bool(_real_key("ANTHROPIC_API_KEY") or _real_key("ANTHROPIC_AUTH_TOKEN"))
    if not has_key and os.environ.get("ANTHROPIC_API_KEY") and provider == "auto":
        log.warning("ANTHROPIC_API_KEY looks like the .env.example placeholder -> ignored")
    if provider in ("auto", "anthropic") and (has_key or provider == "anthropic"):
        try:
            return AnthropicProvider()
        except ImportError:
            log.warning("`anthropic` package not installed -> offline mode. Run: pip install anthropic")
        except Exception as e:  # pragma: no cover
            log.warning("Could not initialise Anthropic client (%s) -> offline mode", e)
    model = os.environ.get("LLM_MODEL") or None
    gemini_key = _real_key("GEMINI_API_KEY") or _real_key("GOOGLE_API_KEY")
    if provider in ("auto", "gemini") and gemini_key:
        return OpenAICompatProvider("gemini", os.environ.get("LLM_BASE_URL") or GEMINI_BASE_URL, gemini_key, model,
                                    prefer=("flash-lite", "flash"), default_model=GEMINI_DEFAULT_MODEL)
    if provider == "gemini":
        log.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set -> offline mode")
    base_url = os.environ.get("LLM_BASE_URL")
    if provider == "openai_compat" and base_url:
        return OpenAICompatProvider("openai_compat", base_url, os.environ.get("LLM_API_KEY", ""), model)
    if provider == "openai_compat":
        log.warning("LLM_PROVIDER=openai_compat but LLM_BASE_URL is not set -> offline mode")
    return NullLLMProvider()
