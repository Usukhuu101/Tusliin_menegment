"""Agent 5: deterministic safety checks."""
import pytest

from backend.models.schemas import RiskLevel

OK_SECTIONS = "🩺 a\n🔎 b [MED_SYM_001]\n💡 c\n⚠️ d 103\n🏥 эмч\n📚 e\nℹ️ Энэ бол ерөнхий мэдээлэл"


def codes(orch, text, risk=RiskLevel.LOW, ids=("MED_SYM_001",)):
    return {i.code for i in orch.safety.check_rules(text, risk=risk, allowed_doc_ids=set(ids), allowed_urls=set(),
                                                    require_sections=True)}


def test_clean_answer_passes(orchestrator):
    assert codes(orchestrator, OK_SECTIONS) == set()


@pytest.mark.parametrize("bad,code", [
    ("Парацетамол 500 мг уугаарай", "MEDICATION_DOSE"),
    ("өдөрт 3 удаа уугаарай", "MEDICATION_DOSE"),
    ("Танд мигрень өвчин байна", "DEFINITIVE_DIAGNOSIS"),
    ("Эмчид үзүүлэх шаардлагагүй", "DISCOURAGES_CARE"),
    ("Даралтын эмээ зогсоогоорой", "MEDICATION_ADVICE"),
    ("[MED_FAKE_1]", "FABRICATED_CITATION"),
])
def test_unsafe_content_is_caught(orchestrator, bad, code):
    assert code in codes(orchestrator, OK_SECTIONS + "\n" + bad)


def test_do_not_stop_is_allowed(orchestrator):
    assert "MEDICATION_ADVICE" not in codes(orchestrator, OK_SECTIONS + "\nЭмээ зогсоохгүй байх")


def test_high_risk_needs_urgent_advice(orchestrator):
    text = OK_SECTIONS.replace("103", "").replace("эмч", "")
    assert "RISK_MISMATCH" in codes(orchestrator, text, risk=RiskLevel.HIGH)


def test_all_kb_answers_pass_safety(orchestrator):
    """Every reviewed KB document, composed into an answer, must pass the rules."""
    from backend.models.schemas import SessionState, TriageResult
    for doc_id, d in orchestrator.index.docs.items():
        norm = orchestrator.processor.process(d["title"])
        a = orchestrator.symptom.analyze(norm, SessionState(session_id="kb"))
        r = orchestrator.rag.retrieve(a, norm)
        if not r.docs:
            continue
        tri = TriageResult(risk_level=RiskLevel.LOW)
        draft = orchestrator.response.compose_template(norm=norm, analysis=a, triage=tri, retrieval=r)
        v = orchestrator.safety.review(draft, triage=tri, retrieval=r, norm=norm)
        assert v.verdict == "SAFE", (doc_id, [i.code + ": " + i.detail for i in v.issues])
