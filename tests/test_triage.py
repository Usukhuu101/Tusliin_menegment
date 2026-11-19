"""Agent 1: red-flag rules. Emergencies must never be missed."""
import json
from pathlib import Path

import pytest

from backend.config import settings
from backend.models.schemas import RiskLevel, SessionState

EMERGENCIES = json.loads((Path(settings.evaluation_dir) / "emergency_tests.json").read_text(encoding="utf-8"))


def assess(orch, text):
    norm = orch.processor.process(text)
    a = orch.symptom.analyze(norm, SessionState(session_id="t"))
    return orch.triage.assess(a.concepts, a.facts, bool(a.symptoms))


@pytest.mark.parametrize("case", EMERGENCIES, ids=[c["input"][:30] for c in EMERGENCIES])
def test_every_emergency_is_detected(orchestrator, case):
    assert assess(orchestrator, case["input"]).risk_level == RiskLevel.EMERGENCY


@pytest.mark.parametrize("text,expected", [
    ("Толгой өвдөөд байна", RiskLevel.LOW),
    ("Цээжээр өвдөж байна", RiskLevel.HIGH),
    ("2 сартай хүүхэд халуураад байна", RiskLevel.HIGH),
    ("Нуруу 3 долоо хоног өвдөж байна", RiskLevel.MODERATE),
    ("70 настай, толгой эргээд байна", RiskLevel.MODERATE),     # elderly modifier
])
def test_risk_levels(orchestrator, text, expected):
    assert assess(orchestrator, text).risk_level == expected


def test_llm_can_raise_but_never_lower(orchestrator):
    from backend.models.schemas import Facts
    t = orchestrator.triage
    assert t.assess(["chest_pain"], Facts(), True, llm_risk=RiskLevel.LOW).risk_level == RiskLevel.HIGH
    assert t.assess(["headache"], Facts(), True, llm_risk=RiskLevel.EMERGENCY).risk_level == RiskLevel.EMERGENCY


def test_emergency_template_mentions_103(orchestrator):
    tri = assess(orchestrator, "Цээж маш хүчтэй өвдөөд амьсгал давчдаад байна")
    assert "103" in orchestrator.triage.render_emergency(tri)
