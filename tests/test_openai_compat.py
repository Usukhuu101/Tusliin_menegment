"""OpenAICompatProvider (Gemini free tier, Groq, OpenRouter, Ollama ...) contract tests.

A tiny local HTTP server plays the provider — no network, no API key, no cost.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from backend.agents.safety_agent import JudgeOutput
from backend.services import llm as llm_mod
from backend.services.llm import LLMError, NullLLMProvider, OpenAICompatProvider, get_llm_provider


class FakeAPI:
    """Queue of (status, body) replies; records every request it receives."""

    def __init__(self):
        self.replies: list[tuple[int, dict]] = []
        self.models = ["models/gemini-9.0-pro", "models/gemini-9.0-flash", "models/gemini-9.0-flash-lite",
                       "models/gemini-9.5-flash-lite-preview", "models/gemini-8.0-flash-lite", "models/text-embedding-9"]
        self.requests: list[dict] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                api.requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
                self._send(200, {"data": [{"id": m} for m in api.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
                api.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                status, reply = api.replies.pop(0) if api.replies else (500, {"error": "no reply queued"})
                self._send(status, reply)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, content="", finish="stop", status=200, usage=(11, 7)):
        if status != 200:
            self.replies.append((status, {"error": {"message": "nope"}}))
        else:
            self.replies.append((200, {"model": "gemini-9.0-flash-lite",
                                       "choices": [{"message": {"role": "assistant", "content": content},
                                                    "finish_reason": finish}],
                                       "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}))


@pytest.fixture()
def api():
    a = FakeAPI()
    yield a
    a.server.shutdown()


def provider(api, model=None, default="gemini-flash-lite-latest"):
    return OpenAICompatProvider("gemini", api.url, "test-key", model, prefer=("flash-lite", "flash"),
                                default_model=default, timeout=5)


def test_discovers_newest_stable_flash_lite(api):
    p = provider(api)
    assert p.model == "gemini-9.0-flash-lite"            # preview + embedding skipped, 9.0 > 8.0
    assert api.requests[0]["path"] == "/v1/models" and api.requests[0]["auth"] == "Bearer test-key"


def test_alias_is_used_when_the_key_offers_it(api):
    api.models.append("models/gemini-flash-lite-latest")
    assert provider(api).model == "gemini-flash-lite-latest"


def test_explicit_model_skips_discovery(api):
    p = provider(api, model="my-model")
    assert p.model == "my-model" and api.requests == []


def test_claude_model_ids_are_replaced(api):
    p = provider(api, model="gemini-x")
    assert p.model_for("claude-opus-5-5") == "gemini-x" and p.model_for("other") == "other"


def test_generate_sends_system_and_returns_text(api):
    p = provider(api, model="gemini-x")
    api.reply("Сайн байна уу")
    r = p.generate(system="SYS", messages=[{"role": "user", "content": "Q"}], model="claude-opus-5-5", effort="high")
    assert r.text == "Сайн байна уу" and r.usage == {"input_tokens": 11, "output_tokens": 7}
    body = api.requests[-1]["body"]
    assert body["model"] == "gemini-x" and body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["max_tokens"] <= 8192


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_truncated_or_blocked_answer_is_rejected(api, finish):
    p = provider(api, model="gemini-x")
    api.reply("half an answer", finish=finish)
    with pytest.raises(LLMError):
        p.generate(system="S", messages=[{"role": "user", "content": "Q"}], model="m")


def test_auth_error_becomes_llm_error(api):
    p = provider(api, model="gemini-x")
    api.reply(status=401)
    with pytest.raises(LLMError, match="401"):
        p.generate(system="S", messages=[{"role": "user", "content": "Q"}], model="m")


def test_rate_limit_is_retried_once(api, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = provider(api, model="gemini-x")
    api.reply(status=429)
    api.reply("ok")
    assert p.generate(system="S", messages=[{"role": "user", "content": "Q"}], model="m").text == "ok"


def test_structured_output_parses_fenced_json(api):
    p = provider(api, model="gemini-x")
    api.reply('```json\n{"verdict": "REVISE", "issues": [{"code": "X", "severity": "REVISE", "detail": "d"}]}\n```')
    out, usage = p.generate_structured(system="JUDGE", messages=[{"role": "user", "content": "draft"}],
                                       model="claude-haiku-5-5", schema=JudgeOutput)
    assert out.verdict == "REVISE" and out.issues[0].code == "X" and usage["model"] == "gemini-9.0-flash-lite"
    body = api.requests[-1]["body"]
    assert body["response_format"] == {"type": "json_object"} and "JSON Schema" in body["messages"][0]["content"]


def test_structured_output_that_does_not_match_schema_is_rejected(api):
    p = provider(api, model="gemini-x")
    api.reply('{"verdict": "PROBABLY_FINE"}')
    with pytest.raises(LLMError):
        p.generate_structured(system="S", messages=[{"role": "user", "content": "Q"}], model="m", schema=JudgeOutput)


def test_provider_without_json_mode_gets_a_second_try(api):
    p = provider(api, model="gemini-x")
    api.reply(status=400)
    api.reply('{"verdict": "SAFE"}')
    out, _ = p.generate_structured(system="S", messages=[{"role": "user", "content": "Q"}], model="m", schema=JudgeOutput)
    assert out.verdict == "SAFE" and "response_format" not in api.requests[-1]["body"]


def test_unreachable_server_is_llm_error():
    p = OpenAICompatProvider("x", "http://127.0.0.1:9/v1", model="m", timeout=2)
    with pytest.raises(LLMError):
        p.generate(system="S", messages=[{"role": "user", "content": "Q"}], model="m")


# ------------------------------------------------------------ selection
@pytest.fixture()
def clean_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_no_key_means_offline(clean_env):
    assert isinstance(get_llm_provider("auto"), NullLLMProvider)
    assert isinstance(get_llm_provider("gemini"), NullLLMProvider)
    assert isinstance(get_llm_provider("openai_compat"), NullLLMProvider)


def test_auto_picks_gemini_when_only_gemini_key(clean_env, api):
    clean_env.setenv("GEMINI_API_KEY", "g-key")
    clean_env.setenv("LLM_BASE_URL", api.url)
    p = get_llm_provider("auto")
    assert p.name == "gemini" and p.available and p.model == "gemini-9.0-flash-lite"


def test_placeholder_claude_key_does_not_hide_gemini(clean_env, api):
    # an uncommented "ANTHROPIC_API_KEY=sk-ant-..." example line used to win over a real Gemini key
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    clean_env.setenv("GEMINI_API_KEY", "g-key")
    clean_env.setenv("LLM_BASE_URL", api.url)
    assert get_llm_provider("auto").name == "gemini"


def test_openai_compat_uses_base_url_and_model(clean_env, api):
    clean_env.setenv("LLM_BASE_URL", api.url)
    clean_env.setenv("LLM_MODEL", "llama-test")
    p = get_llm_provider("openai_compat")
    assert p.name == "openai_compat" and p.model == "llama-test"


def test_gemini_default_base_url():
    assert llm_mod.GEMINI_BASE_URL.startswith("https://generativelanguage.googleapis.com/")
