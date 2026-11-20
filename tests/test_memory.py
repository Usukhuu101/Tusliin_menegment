"""Conversation memory: the bot remembers what the user said earlier IN THIS CHAT."""
from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database.db import Database
from backend.models.schemas import RiskLevel
from backend.services.llm import LLMError, LLMProvider
from backend.services.orchestrator import ChatOrchestrator


def talk(orch, *messages):
    sid, out = None, []
    for m in messages:
        r = orch.handle(m, sid)
        sid = r.session_id
        out.append(r)
    return out


def test_self_description_is_acknowledged_and_used_later(orchestrator):
    noted, answer = talk(orchestrator, "Би 28 настай, жирэмсэн", "Парацетамол ууж болох уу")
    assert noted.message_type == "NOTED" and "28 настай" in noted.reply and "жирэмсэн" in noted.reply
    assert answer.message_type == "ANSWER" and answer.sources[0].doc_id == "MED_EML_024"
    assert "🧠 Энэ ярианд өмнө хэлсэн: 28 настай, жирэмсэн" in answer.reply
    # pregnancy matters for a medicine -> explicit "tell your pharmacist" line (adult age alone does not)
    assert "Та энэ ярианд өмнө **жирэмсэн** гэж хэлсэн" in answer.reply


def test_recall_question_lists_what_was_said(orchestrator):
    *_, r = talk(orchestrator, "Би 28 настай, жирэмсэн", "Ибупрофен жороор олгодог уу", "Миний нас хэд гэж хэлсэн бэ")
    assert r.message_type == "RECALL"
    assert "28 настай" in r.reply and "жирэмсэн" in r.reply and "Ибупрофен" in r.reply


def test_recall_with_nothing_remembered_says_so(orchestrator):
    (r,) = talk(orchestrator, "Намайг санаж байна уу")
    assert r.message_type == "RECALL" and "юу ч хэлээгүй" in r.reply


def test_that_medicine_means_the_one_asked_before(orchestrator):
    _, r = talk(orchestrator, "Ибупрофен жороор олгодог уу", "Тэр эмийн гаж нөлөө юу вэ")
    assert r.message_type == "ANSWER"
    assert "MED_EML_019" in [s.doc_id for s in r.sources]          # still about ibuprofen
    assert "«тэр эм» = Ибупрофен" in r.agents[0].detail


def test_remembered_baby_age_raises_risk_of_a_later_fever(orchestrator):
    noted, fever = talk(orchestrator, "Хүүхэд маань 3 сартай", "38.5 халуураад байна")
    assert noted.message_type == "NOTED"
    assert fever.risk_level in (RiskLevel.HIGH, RiskLevel.EMERGENCY)


def test_allergy_is_remembered_in_the_users_words(orchestrator):
    noted, r = talk(orchestrator, "Пенициллинд харшилтай", "Амоксициллин ууж болох уу")
    assert noted.message_type == "NOTED"
    assert "«пенициллинд харшилтай»" in r.reply and "эмч, эм зүйчдээ заавал хэлж" in r.reply


def test_chronic_disease_is_remembered_only_when_it_is_the_users_own(orchestrator):
    *_, r = talk(orchestrator, "Чихрийн шижин гэж юу вэ", "Миний тухай юу санаж байна")
    assert r.message_type == "RECALL" and "Хэлсэн нөхцөл" not in r.reply       # a question, not "I have it"
    *_, r = talk(orchestrator, "Би чихрийн шижинтэй", "Толгой эргээд байна", "Өчигдрөөс, 4 оноо, үгүй")
    assert r.message_type == "ANSWER" and "🧠 Энэ ярианд өмнө хэлсэн: чихрийн шижин" in r.reply


def test_symptom_mentioning_earlier_words_is_not_a_recall(orchestrator):
    (r,) = talk(orchestrator, "Өмнө хэлсэн шиг толгой өвдөөд байна")
    assert r.message_type == "FOLLOWUP"


def test_followup_answer_with_age_is_not_mistaken_for_a_self_description(orchestrator):
    q, a = talk(orchestrator, "Толгой өвдөөд байна", "Өчигдрөөс, 5 оноо, үгүй, 45 настай")
    assert q.message_type == "FOLLOWUP" and a.message_type == "ANSWER"


def test_memory_survives_a_restart_and_is_deleted_with_the_session(tmp_path):
    db_path = tmp_path / "m.db"
    s = dataclasses.replace(settings, db_path=db_path, llm_provider="none")
    first = ChatOrchestrator(s, db=Database(db_path))
    sid = first.handle("Би 70 настай").session_id
    again = ChatOrchestrator(s, db=Database(db_path))          # e.g. the server was restarted
    r = again.handle("Миний нас хэд гэж хэлсэн бэ", sid)
    assert "70 настай" in r.reply
    again.db.delete_session(sid)
    assert again.db.get_session(sid) is None


class _CaptureAnswerPrompt(LLMProvider):
    name, available = "capture", True

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, *, messages, **kw):
        self.prompts.append(messages[0]["content"])
        raise LLMError("captured")              # -> the pipeline falls back to the template answer

    def generate_structured(self, **kw):
        raise LLMError("not used")


def test_llm_sees_the_conversation_memory(tmp_path):
    db_path = tmp_path / "l.db"
    s = dataclasses.replace(settings, db_path=db_path, llm_provider="none")
    orch = ChatOrchestrator(s, db=Database(db_path))
    llm = _CaptureAnswerPrompt()
    orch.response.llm = llm
    talk(orch, "Би жирэмсэн", "Парацетамол ууж болох уу")
    prompt = llm.prompts[-1]
    assert "<conversation_memory>" in prompt and "жирэмсэн" in prompt and "«Би жирэмсэн» → NOTED" in prompt


# ------------------------------------------------------------------ API
@pytest.fixture()
def client(tmp_path):
    from backend import main

    db_path = tmp_path / "h.db"
    s = dataclasses.replace(settings, db_path=db_path, llm_provider="none")
    with TestClient(main.app) as c:
        main.app.state.orchestrator = ChatOrchestrator(s, db=Database(db_path))
        yield c


def test_history_endpoint_redraws_the_chat_without_personal_data(client):
    r1 = client.post("/api/chat", json={"message": "Би 30 настай, утас 99112233"}).json()
    sid = r1["session_id"]
    client.post("/api/chat", json={"message": "Толгой өвдөөд байна", "session_id": sid})
    h = client.get(f"/api/session/{sid}/history").json()
    assert [t["message_type"] for t in h["turns"]] == ["NOTED", "FOLLOWUP"]
    assert "99112233" not in h["turns"][0]["user"]                       # stored redacted
    assert client.get("/api/session/bad id!/history").status_code == 404
    assert client.get("/api/session/unknown-session-123/history").json()["turns"] == []
