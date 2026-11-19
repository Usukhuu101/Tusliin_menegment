"""
LLM path of the multi-agent pipeline, tested WITHOUT an API key.

Part A — a scriptable FakeLLMProvider is injected into an offline ChatOrchestrator
         (temp DB) to exercise: LLM extraction (Agent 2) + risk merge (Agent 1),
         LLM answer (Agent 4), LLM judge + regeneration + fail-closed fallbacks (Agent 5),
         and token-usage bookkeeping in the stored trace.
Part B — AnthropicProvider contract tests against the INSTALLED `anthropic` SDK,
         using an in-process mock HTTP transport (no network, no real API calls).
"""
from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from backend.config import settings
from backend.database.db import Database
from backend.models.schemas import RiskLevel, SessionState
from backend.services.llm import LLMError, LLMProvider, LLMResult

ANSWER_USAGE = {"input_tokens": 1200, "output_tokens": 400}
STRUCTURED_USAGE = {"input_tokens": 300, "output_tokens": 60}


# =========================================================== fake provider
class FakeLLMProvider(LLMProvider):
    """Returns queued outputs in order; raises queued exceptions.

    A queue item may be: str (generate), dict / pydantic model (generate_structured),
    an exception INSTANCE (raised), or a callable(call) -> any of those.
    When a queue is empty the `*_default` item is used; if that is None -> LLMError.
    """
    name = "fake"
    available = True

    def __init__(self, *, answers=(), extraction=(), judge=(),
                 answer_default=None, extraction_default=None, judge_default=None):
        self.queues = {"generate": list(answers), "LLMExtraction": list(extraction), "JudgeOutput": list(judge)}
        self.defaults = {"generate": answer_default, "LLMExtraction": extraction_default, "JudgeOutput": judge_default}
        self.calls: list[dict] = []

    def _next(self, key: str, call: dict) -> Any:
        q = self.queues.setdefault(key, [])
        item = q.pop(0) if q else self.defaults.get(key)
        if item is None:
            raise LLMError(f"FakeLLM: nothing scripted for {key}")
        if callable(item) and not isinstance(item, (BaseModel, BaseException)):
            item = item(call)
        if isinstance(item, BaseException):
            raise item
        return item

    def generate(self, *, system, messages, model, max_tokens=16000, effort=None) -> LLMResult:
        call = {"kind": "generate", "system": system, "user": messages[-1]["content"], "model": model,
                "effort": effort, "max_tokens": max_tokens}
        self.calls.append(call)
        text = self._next("generate", call)
        return LLMResult(text=text, model=model, usage=dict(ANSWER_USAGE))

    def generate_structured(self, *, system, messages, model, schema, max_tokens=8000):
        call = {"kind": schema.__name__, "system": system, "user": messages[-1]["content"], "model": model,
                "max_tokens": max_tokens}
        self.calls.append(call)
        out = self._next(schema.__name__, call)
        if isinstance(out, dict):
            out = schema.model_validate(out)
        assert isinstance(out, schema), f"scripted {type(out).__name__} for {schema.__name__}"
        return out, dict(STRUCTURED_USAGE)

    def of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]


# ---------------------------------------------------------------- scripts
def extraction(intent="general_info", concepts=(), red_flags=(), risk="LOW", **extra) -> dict:
    return {"intent": intent, "concepts": list(concepts), "red_flags_detected": list(red_flags),
            "llm_risk": risk, **extra}


JUDGE_SAFE = {"verdict": "SAFE", "issues": []}


def judge_revise(code="UNSUPPORTED_CLAIM", detail="Баримтад байхгүй мэдээлэл орсон") -> dict:
    return {"verdict": "REVISE", "issues": [{"code": code, "severity": "REVISE", "detail": detail}]}


DOC_RE = re.compile(r'<document id="(MED_[A-Z0-9_]+)"')


def answer(tag: str = "1", *, cite: str | None = None, extra: str = "", no_cite: bool = False):
    """Callable script: a well-formed 6-section Mongolian answer citing the FIRST retrieved doc."""
    def build(call: dict) -> str:
        ids = DOC_RE.findall(call["user"])
        assert ids, "response prompt must contain the retrieved documents"
        c = "" if no_cite else (cite or f"[{ids[0]}]")
        return (
            f"🩺 **Таны асуулт**\n- Таны асуусан сэдвийн талаар ерөнхий мэдээлэл (хувилбар {tag}).\n\n"
            f"🔎 **Ерөнхий мэдээлэл**\nБаримт бичигт дурдсанаар энэ байдал олон хүнд тохиолддог, "
            f"тодорхой дүгнэлтийг зөвхөн эмчийн үзлэгээр гаргана {c}.\n\n"
            f"💡 **Одоохондоо юу хийж болох вэ?**\n- Амарч, хангалттай шингэн ууж, биеийн байдлаа ажиглаарай {c}.{extra}\n\n"
            f"⚠️ **Анхаарах шинж**\n- Амьсгал давчдах, ухаан балартах шинж илэрвэл даруй 103 дугаарт залгаарай {c}.\n\n"
            f"🏥 **Хэзээ эмчид хандах вэ?**\n- Шинж тэмдэг хүндэрвэл өнөөдөр эмчид үзүүлээрэй {c}.\n\n"
            f"📚 **Эх сурвалж**\n- {c} Лавлах баримт\n\n"
            "ℹ️ Энэ бол ерөнхий мэдээлэл бөгөөд эмчийн үзлэг, оношийг орлохгүй."
        )
    return build


def text_of(fake: FakeLLMProvider, i: int, script) -> str:
    """Re-render the text a script produced for the i-th generate() call."""
    return script(fake.of("generate")[i])


# ================================================================ fixtures
@pytest.fixture(scope="module")
def orch(tmp_path_factory):
    from backend.services.orchestrator import ChatOrchestrator

    db_path = tmp_path_factory.mktemp("llm") / "llm.db"
    s = dataclasses.replace(settings, db_path=db_path, llm_provider="none")
    return ChatOrchestrator(s, db=Database(db_path))


@pytest.fixture
def use(orch):
    """use(fake) -> installs the fake LLM in the orchestrator and every LLM-using agent."""
    def install(fake: FakeLLMProvider) -> FakeLLMProvider:
        orch.llm = fake
        orch.symptom.llm = fake
        orch.response.llm = fake
        orch.safety.llm = fake
        return fake
    return install


def steps(r, agent: str):
    return [a for a in r.agents if a.agent == agent]


def stored_trace(orch, turn_id: int) -> dict:
    con = sqlite3.connect(orch.db.path)
    try:
        row = con.execute("SELECT trace_json, mode FROM turns WHERE turn_id=?", (turn_id,)).fetchone()
    finally:
        con.close()
    assert row is not None
    trace = json.loads(row[0])
    trace["_mode"] = row[1]
    return trace


# ======================================================== Agent 2 + merge
def test_llm_extraction_adds_valid_concept_and_ignores_invalid_ids(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction(
        intent="symptom_check", concepts=["headache", "not_a_concept", "MED_SYM_001"],
        red_flags=["made_up_flag"], risk="LOW")]))
    r = orch.handle("Бие тааруу л байна даа")          # rules alone: out_of_scope, no concepts
    assert r.message_type == "FOLLOWUP"               # LLM found a symptom -> Agent 2 asks follow-ups
    t = stored_trace(orch, r.turn_id)
    assert t["symptoms"]["concepts"] == ["headache"]
    assert t["symptoms"]["llm_red_flags"] == []
    assert t["symptoms"]["source"] == "rules+llm"
    assert "дүрэм + LLM" in steps(r, "symptom")[0].detail
    call = fake.of("LLMExtraction")[0]
    assert call["model"] == orch.s.extraction_model and "<allowed_concepts>" in call["user"]


def test_llm_never_adds_a_concept_the_user_denied(orch, use):
    use(FakeLLMProvider(extraction=[extraction(intent="symptom_check", concepts=["headache", "fever"])]))
    a = orch.symptom.analyze(orch.processor.process("Толгой өвдөөд байна, халуураагүй"), SessionState(session_id="neg"))
    assert "fever" in a.negated_concepts and "fever" not in a.concepts and "headache" in a.concepts


def test_llm_risk_emergency_routes_to_emergency_template(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction(intent="symptom_check", risk="EMERGENCY")]))
    r = orch.handle("Бие тааруу л байна даа")
    assert r.message_type == "EMERGENCY" and r.risk_level == RiskLevel.EMERGENCY and "103" in r.reply
    merge = steps(r, "triage_merge")[0]
    assert merge.status == "alert" and "LLM" in merge.detail
    assert fake.of("generate") == [] and fake.of("JudgeOutput") == []   # emergency text is never LLM-written


def test_llm_risk_high_suppresses_followups(orch, use):
    use(FakeLLMProvider(extraction=[extraction(intent="symptom_check", concepts=["headache"], risk="HIGH")],
                        answers=[LLMError("down")]))
    r = orch.handle("Толгой өвдөөд байна")             # offline this is LOW -> FOLLOWUP
    assert r.risk_level == RiskLevel.HIGH and r.message_type == "ANSWER"
    assert "өнөөдөр" in r.reply.lower()


def test_llm_cannot_lower_rule_based_high(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction(intent="general_info", risk="LOW")],
                               answers=[answer()], judge=[JUDGE_SAFE]))
    r = orch.handle("Цээжээр өвдөж байна")
    assert r.risk_level == RiskLevel.HIGH
    t = stored_trace(orch, r.turn_id)
    assert t["triage_final"]["risk_level"] == "HIGH" and t["triage_final"]["source"] == "rules"
    assert "risk_level: HIGH" in fake.of("generate")[0]["user"]       # Agent 4 is told the rule risk
    assert "<triage_risk>HIGH</triage_risk>" in fake.of("JudgeOutput")[0]["user"]


def test_llm_cannot_lower_rule_based_emergency_pass1(orch, use):
    fake = use(FakeLLMProvider(extraction_default=extraction(intent="general_info", risk="LOW"),
                               answer_default=answer(), judge_default=JUDGE_SAFE))
    r = orch.handle("Цээж маш хүчтэй өвдөөд амьсгал давчдаад байна")
    assert r.message_type == "EMERGENCY" and r.risk_level == RiskLevel.EMERGENCY
    assert fake.calls == []                              # pass-1 emergency never consults the LLM


def test_llm_cannot_lower_rule_based_emergency_pass2(orch, use):
    use(FakeLLMProvider(extraction_default=extraction(intent="general_info", risk="LOW"),
                        answer_default=LLMError("down")))
    r1 = orch.handle("Цээжээр өвдөж байна")
    assert r1.risk_level == RiskLevel.HIGH
    r2 = orch.handle("Амьсгал давчдаад байна", r1.session_id)   # merged chest pain + dyspnea
    assert r2.message_type == "EMERGENCY" and r2.risk_level == RiskLevel.EMERGENCY


def test_extraction_llm_error_falls_back_to_rules(orch, use):
    fake = use(FakeLLMProvider(extraction=[LLMError("timeout")]))
    r = orch.handle("Толгой өвдөөд байна")
    assert r.message_type == "FOLLOWUP"                  # same as offline
    t = stored_trace(orch, r.turn_id)
    assert t["symptoms"]["source"] == "rules" and t["llm_usage"] == []
    assert len(fake.of("LLMExtraction")) == 1


# ================================================== Agent 4 + Agent 5 (LLM)
Q = "Даралт ихсэлт гэж юу вэ?"


def test_llm_answer_released_when_rules_and_judge_safe(orch, use):
    script = answer("1")
    fake = use(FakeLLMProvider(extraction=[extraction()], answers=[script], judge=[JUDGE_SAFE]))
    r = orch.handle(Q)
    assert r.message_type == "ANSWER"
    assert r.reply == text_of(fake, 0, script)
    assert r.mode == "llm:fake" and r.sources
    resp = steps(r, "response")
    assert len(resp) == 1 and "LLM" in resp[0].detail
    saf = steps(r, "safety")
    assert len(saf) == 1 and saf[0].status == "ok" and "LLM шүүгч" in saf[0].detail
    gen = fake.of("generate")[0]
    assert gen["model"] == orch.s.response_model and gen["effort"] == orch.s.response_effort
    assert gen["system"] == orch.response.system_prompt and "<retrieved_documents>" in gen["user"]
    jc = fake.of("JudgeOutput")[0]
    assert jc["model"] == orch.s.safety_model and "<draft_answer>" in jc["user"] and "хувилбар 1" in jc["user"]
    t = stored_trace(orch, r.turn_id)
    assert t["safety"][-1]["verdict"] == "SAFE" and t["safety"][-1]["checked_by"] == ["rules", "llm_judge"]
    assert t["drafts"][0]["generator"] == "llm"


@pytest.mark.parametrize("bad_cite", ["[MED_FAKE]", "[MED_SYM_999]"])
def test_fabricated_citation_is_never_released(orch, use, bad_cite):
    fake = use(FakeLLMProvider(extraction=[extraction()], answer_default=answer(cite=bad_cite),
                               judge_default=JUDGE_SAFE))
    r = orch.handle(Q)
    n = orch.s.max_regenerations + 1
    assert len(fake.of("generate")) == n                 # first draft + max regenerations
    assert "FABRICATED_CITATION" in fake.of("generate")[1]["user"]       # feedback reached the writer
    assert r.message_type == "ANSWER" and "хувилбар" not in r.reply and bad_cite not in r.reply
    assert steps(r, "safety")[-1].status == "fallback"
    t = stored_trace(orch, r.turn_id)
    assert t["drafts"][-1]["generator"] == "template" and t["safety"][-1]["verdict"] == "SAFE"


def test_fabricated_citation_inside_multi_id_bracket_is_caught(orch, use):
    def multi(call):
        real = DOC_RE.findall(call["user"])[0]
        return answer(cite=f"[{real}, MED_FAKE_9]")(call)
    use(FakeLLMProvider(extraction=[extraction()], answer_default=multi, judge_default=JUDGE_SAFE))
    r = orch.handle(Q)
    assert "MED_FAKE_9" not in r.reply and "хувилбар" not in r.reply


def test_llm_answer_without_any_citation_is_not_released(orch, use):
    use(FakeLLMProvider(extraction=[extraction()], answer_default=answer(no_cite=True), judge_default=JUDGE_SAFE))
    r = orch.handle(Q)
    assert r.message_type == "ANSWER" and "хувилбар" not in r.reply
    assert re.search(r"\[MED_[A-Z0-9_]+\]", r.reply)   # the template answer is cited


def test_dose_in_llm_draft_is_blocked_without_judge_or_retry(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction()],
                               answers=[answer(extra="\n- Парацетамол 500 мг уугаарай.")], judge_default=JUDGE_SAFE))
    r = orch.handle(Q)
    assert "500 мг" not in r.reply and "хувилбар" not in r.reply
    assert len(fake.of("generate")) == 1                 # BLOCK -> no regeneration
    assert fake.of("JudgeOutput") == []                  # rules BLOCK -> judge not even asked
    t = stored_trace(orch, r.turn_id)
    assert t["safety"][0]["verdict"] == "BLOCK"
    assert any(i["code"] == "MEDICATION_DOSE" for i in t["safety"][0]["issues"])
    assert t["drafts"][-1]["generator"] == "template"


def test_judge_revise_once_then_regenerated_draft_released(orch, use):
    s1, s2 = answer("1"), answer("2")
    fake = use(FakeLLMProvider(extraction=[extraction()], answers=[s1, s2],
                               judge=[judge_revise("UNSUPPORTED_CLAIM", "Баримтад байхгүй тоо дурдсан"), JUDGE_SAFE]))
    r = orch.handle(Q)
    gens = fake.of("generate")
    assert len(gens) == 2
    assert "<safety_feedback>" not in gens[0]["user"]
    assert "<safety_feedback>" in gens[1]["user"]
    assert "UNSUPPORTED_CLAIM: Баримтад байхгүй тоо дурдсан" in gens[1]["user"]
    assert r.reply == text_of(fake, 1, s2) and "хувилбар 2" in r.reply
    assert [a.status for a in steps(r, "safety")] == ["alert", "ok"]
    assert len(steps(r, "response")) == 2
    t = stored_trace(orch, r.turn_id)
    assert [d["attempt"] for d in t["drafts"]] == [1, 2]
    resp_usage = [u for u in t["llm_usage"] if u["agent"] == "response"]
    assert [u["attempt"] for u in resp_usage] == [1, 2]


def test_judge_always_revise_ends_in_template(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction()], answer_default=answer(), judge_default=judge_revise()))
    r = orch.handle(Q)
    assert len(fake.of("generate")) == orch.s.max_regenerations + 1
    assert "хувилбар" not in r.reply and steps(r, "safety")[-1].status == "fallback"


def test_judge_block_goes_straight_to_template(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction()], answers=[answer()], judge=[
        {"verdict": "BLOCK", "issues": [{"code": "DIAGNOSIS", "severity": "BLOCK", "detail": "онош тавьсан"}]}]))
    r = orch.handle(Q)
    assert len(fake.of("generate")) == 1 and "хувилбар" not in r.reply
    assert stored_trace(orch, r.turn_id)["safety"][0]["verdict"] == "BLOCK"


def test_judge_unavailable_fails_closed(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction()], answers=[answer()], judge=[LLMError("judge down")]))
    r = orch.handle(Q)
    assert r.message_type == "ANSWER" and "хувилбар" not in r.reply
    assert len(fake.of("generate")) == 1                 # no pointless regeneration while the judge is down
    t = stored_trace(orch, r.turn_id)
    assert any(i["code"] == "JUDGE_UNAVAILABLE" for i in t["safety"][0]["issues"])
    assert t["drafts"][-1]["generator"] == "template" and t["safety"][-1]["verdict"] == "SAFE"


def test_response_llm_error_falls_back_to_template(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction()], answers=[LLMError("overloaded")], judge_default=JUDGE_SAFE))
    r = orch.handle(Q)
    assert r.message_type == "ANSWER" and "ℹ️" in r.reply and "хувилбар" not in r.reply
    assert fake.of("JudgeOutput") == []                  # template drafts are checked by rules only
    t = stored_trace(orch, r.turn_id)
    assert t["drafts"][0]["generator"] == "template" and t["safety"][0]["checked_by"] == ["rules"]


def test_regeneration_llm_error_falls_back_to_template(orch, use):
    fake = use(FakeLLMProvider(extraction=[extraction()], answers=[answer("1"), LLMError("down")],
                               judge=[judge_revise()], judge_default=JUDGE_SAFE))
    r = orch.handle(Q)
    assert len(fake.of("generate")) == 2 and "хувилбар" not in r.reply and r.message_type == "ANSWER"


def test_extraction_error_does_not_stop_llm_answer(orch, use):
    script = answer("1")
    fake = use(FakeLLMProvider(extraction=[LLMError("haiku down")], answers=[script], judge=[JUDGE_SAFE]))
    r = orch.handle(Q)
    assert r.reply == text_of(fake, 0, script)


# ==================================================================== usage
def test_usage_tokens_recorded_in_stored_trace(orch, use):
    use(FakeLLMProvider(extraction=[extraction()], answers=[answer()], judge=[JUDGE_SAFE]))
    r = orch.handle(Q)
    t = stored_trace(orch, r.turn_id)
    assert t["_mode"] == "llm:fake"
    usage = t["llm_usage"]
    assert [u["agent"] for u in usage] == ["symptom", "response", "safety"]
    by = {u["agent"]: u for u in usage}
    assert by["response"]["input_tokens"] == ANSWER_USAGE["input_tokens"]
    assert by["response"]["output_tokens"] == ANSWER_USAGE["output_tokens"]
    assert by["response"]["model"] == orch.s.response_model
    assert by["symptom"]["input_tokens"] == STRUCTURED_USAGE["input_tokens"]
    assert by["safety"]["output_tokens"] == STRUCTURED_USAGE["output_tokens"]
    assert by["safety"]["model"] == orch.s.safety_model


def test_usage_recorded_on_emergency_and_fallback_paths(orch, use):
    use(FakeLLMProvider(extraction=[extraction(intent="symptom_check", risk="EMERGENCY")]))
    r = orch.handle("Бие тааруу л байна даа")
    assert [u["agent"] for u in stored_trace(orch, r.turn_id)["llm_usage"]] == ["symptom"]
    use(FakeLLMProvider(extraction=[extraction()], answers=[answer()], judge=[LLMError("down")]))
    r = orch.handle(Q)
    assert [u["agent"] for u in stored_trace(orch, r.turn_id)["llm_usage"]] == ["symptom", "response"]


# ====================================================== Part B: SDK contract
httpx2 = pytest.importorskip("httpx2")
anthropic = pytest.importorskip("anthropic")


def _msg(text=None, *, stop="end_turn", model="claude-opus-5-5", content=None, usage=(11, 7)) -> dict:
    blocks = content if content is not None else ([{"type": "text", "text": text}] if text is not None else [])
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": blocks,
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}


@pytest.fixture
def provider(monkeypatch):
    """AnthropicProvider whose HTTP traffic goes to an in-process mock transport (never the network)."""
    from backend.services.llm import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-offline")
    p = AnthropicProvider()
    p.requests = []
    p.reply = _msg("сайн")
    p.status = 200

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        p.requests.append({"path": request.url.path, "query": dict(request.url.params),
                           "headers": dict(request.headers), "body": body})
        return httpx2.Response(p.status, json=p.reply)

    p.client = anthropic.Anthropic(api_key="sk-ant-test-offline", max_retries=0, timeout=5.0,
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    return p


FORBIDDEN_PARAMS = {"temperature", "top_p", "top_k", "thinking"}


def test_sdk_generate_uses_server_side_fallback_for_opus(provider):
    res = provider.generate(system="sys", messages=[{"role": "user", "content": "hi"}],
                            model="claude-opus-5-5", effort="medium")
    assert res.text == "сайн" and res.usage == {"input_tokens": 11, "output_tokens": 7}
    req = provider.requests[0]
    assert req["path"] == "/v1/messages" and req["query"].get("beta") == "true"
    assert "server-side-fallback-2026-07-01" in req["headers"]["anthropic-beta"]
    body = req["body"]
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "medium"}
    assert body["system"] == "sys" and body["model"] == "claude-opus-5-5"
    assert not FORBIDDEN_PARAMS & set(body), body.keys()


def test_sdk_generate_plain_messages_for_other_models(provider):
    provider.reply = _msg("сайн", model="claude-haiku-5-5")
    provider.generate(system="s", messages=[{"role": "user", "content": "hi"}], model="claude-haiku-5-5")
    req = provider.requests[0]
    assert "beta" not in req["query"] and "anthropic-beta" not in req["headers"]
    assert "fallbacks" not in req["body"] and "output_config" not in req["body"]
    assert not FORBIDDEN_PARAMS & set(req["body"])


def test_sdk_generate_keeps_only_text_after_fallback_boundary(provider):
    provider.reply = _msg(model="claude-opus-4-8", content=[
        {"type": "text", "text": "ТАСАЛДСАН"},
        {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"},
         "trigger": {"type": "refusal"}},
        {"type": "text", "text": "Бүтэн хариулт"},
    ])
    res = provider.generate(system="s", messages=[{"role": "user", "content": "hi"}], model="claude-opus-5-5")
    assert res.text == "Бүтэн хариулт" and res.model == "claude-opus-4-8"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens", "model_context_window_exceeded"])
def test_sdk_generate_rejects_refused_or_truncated_output(provider, stop):
    provider.reply = _msg("хагас хариу" if stop != "refusal" else None, stop=stop)
    with pytest.raises(LLMError):
        provider.generate(system="s", messages=[{"role": "user", "content": "hi"}], model="claude-opus-5-5")


@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
def test_sdk_generate_http_errors_become_llm_error(provider, status):
    provider.status = status
    provider.reply = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
    with pytest.raises(LLMError):
        provider.generate(system="s", messages=[{"role": "user", "content": "hi"}], model="claude-opus-5-5")


@pytest.mark.parametrize("exc", [TypeError("Could not resolve authentication method"),
                                 anthropic.CredentialsError("no credentials"),
                                 anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))])
def test_sdk_non_api_errors_become_llm_error(provider, exc):
    """Missing credentials raise TypeError / CredentialsError (NOT APIError) inside the SDK."""
    def boom(**kwargs):
        raise exc
    provider.client = SimpleNamespace(messages=SimpleNamespace(create=boom, parse=boom),
                                      beta=SimpleNamespace(messages=SimpleNamespace(create=boom)))
    for model in ("claude-opus-5-5", "claude-haiku-5-5"):
        with pytest.raises(LLMError):
            provider.generate(system="s", messages=[{"role": "user", "content": "hi"}], model=model)
    from backend.agents.safety_agent import JudgeOutput
    with pytest.raises(LLMError):
        provider.generate_structured(system="s", messages=[{"role": "user", "content": "hi"}],
                                     model="claude-haiku-5-5", schema=JudgeOutput)


def test_sdk_generate_structured_parses_judge_output(provider):
    from backend.agents.safety_agent import JudgeOutput

    provider.reply = _msg(json.dumps({"verdict": "REVISE", "issues": [
        {"code": "X", "severity": "REVISE", "detail": "засах"}]}, ensure_ascii=False), model="claude-haiku-5-5")
    out, usage = provider.generate_structured(system="judge", messages=[{"role": "user", "content": "d"}],
                                              model="claude-haiku-5-5", schema=JudgeOutput)
    assert isinstance(out, JudgeOutput) and out.verdict == "REVISE" and out.issues[0].code == "X"
    assert usage == {"input_tokens": 11, "output_tokens": 7}
    body = provider.requests[0]["body"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "verdict" in body["output_config"]["format"]["schema"]["properties"]
    assert "output_format" not in body                   # SDK moves it into output_config.format
    assert not FORBIDDEN_PARAMS & set(body)


def test_sdk_generate_structured_parses_extraction(provider):
    from backend.agents.symptom_agent import LLMExtraction

    provider.reply = _msg(json.dumps(extraction(intent="symptom_check", concepts=["headache"], risk="HIGH")))
    out, _ = provider.generate_structured(system="x", messages=[{"role": "user", "content": "d"}],
                                          model="claude-haiku-5-5", schema=LLMExtraction)
    assert out.concepts == ["headache"] and out.llm_risk == "HIGH"


@pytest.mark.parametrize("reply", [
    _msg("энэ JSON биш", model="claude-haiku-5-5"),
    _msg('{"verdict": "MAYBE"}', model="claude-haiku-5-5"),
    _msg(None, stop="refusal", model="claude-haiku-5-5"),
    _msg('{"verdict": "SA', stop="max_tokens", model="claude-haiku-5-5"),
])
def test_sdk_generate_structured_bad_output_becomes_llm_error(provider, reply):
    from backend.agents.safety_agent import JudgeOutput

    provider.reply = reply
    with pytest.raises(LLMError):
        provider.generate_structured(system="s", messages=[{"role": "user", "content": "d"}],
                                     model="claude-haiku-5-5", schema=JudgeOutput)
