"""End-to-end through the HTTP API (offline mode)."""
import dataclasses

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database.db import Database
from backend.services.orchestrator import ChatOrchestrator


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from backend import main

    db_path = tmp_path_factory.mktemp("api") / "api.db"
    s = dataclasses.replace(settings, db_path=db_path, llm_provider="none")
    with TestClient(main.app) as c:
        main.app.state.orchestrator = ChatOrchestrator(s, db=Database(db_path))
        yield c


def chat(client, msg, sid=None):
    r = client.post("/api/chat", json={"message": msg, "session_id": sid})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["kb_docs"] > 0


def test_followup_then_answer(client):
    r1 = chat(client, "Толгой өвдөөд байна")
    assert r1["message_type"] == "FOLLOWUP" and len(r1["followup_questions"]) >= 1
    r2 = chat(client, "Өчигдрөөс, 5 оноо, үгүй", r1["session_id"])
    assert r2["message_type"] == "ANSWER"
    assert any(s["doc_id"] == "MED_SYM_001" for s in r2["sources"])
    assert "ℹ️" in r2["reply"]


def test_emergency(client):
    r = chat(client, "Цээж маш хүчтэй өвдөөд амьсгал давчдаад байна")
    assert r["message_type"] == "EMERGENCY" and r["risk_level"] == "EMERGENCY" and "103" in r["reply"]


def test_unknown_topic_does_not_hallucinate(client):
    assert chat(client, "Шүд өвдөөд байна")["message_type"] in ("FALLBACK", "CLARIFY")


def test_feedback_and_delete(client):
    r = chat(client, "Даралт ихсэлт гэж юу вэ?")
    ok = client.post("/api/feedback", json={"session_id": r["session_id"], "turn_id": r["turn_id"],
                                             "helpful": False, "reason": "not_enough"})
    assert ok.status_code == 200
    bad = client.post("/api/feedback", json={"session_id": "someone-else-123", "turn_id": r["turn_id"], "helpful": True})
    assert bad.status_code == 404
    d = client.delete(f"/api/session/{r['session_id']}").json()
    assert d["deleted_turns"] >= 1


def test_all_five_agents_report_their_work(client):
    r = chat(client, "Даралт ихсэлт гэж юу вэ?")
    steps = {a["agent"]: a for a in r["agents"]}
    assert set(steps) == {"input", "triage", "symptom", "triage_merge", "rag", "response", "safety"}
    assert all(a["status"] != "skipped" for a in r["agents"])
    e = chat(client, "Цээж маш хүчтэй өвдөөд амьсгал давчдаад байна")
    es = {a["agent"]: a["status"] for a in e["agents"]}
    assert es["triage"] == "alert" and es["symptom"] == "skipped" and es["rag"] == "skipped"


def test_empty_message_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
