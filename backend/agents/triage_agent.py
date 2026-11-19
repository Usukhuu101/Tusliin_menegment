"""
AGENT 1 — TRIAGE / RISK AGENT  (deterministic rules, optional LLM signal)

Runs TWICE per turn:
  pass 1: right after input processing, on the current message only
          -> catches obvious emergencies in milliseconds, even if the LLM API is down
  pass 2: after the Symptom Agent, on the session-merged concepts and facts
          (+ LLM risk signals, which may RAISE but never LOWER the risk)

All medical logic is in data/rules/*.yaml so a medical reviewer can audit it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml

from ..models.schemas import Facts, RedFlag, RiskLevel, TriageResult, max_risk, raise_one_level, risk_rank

ACTIONS = {
    RiskLevel.EMERGENCY: "Яг одоо 103 дугаарт залгаж яаралтай тусламж авна уу.",
    RiskLevel.HIGH: "Өнөөдөр аль болох түрүүн эмчид үзүүлнэ үү. Байдал муудвал 103 дугаарт залгана уу.",
    RiskLevel.MODERATE: "Ойрын 1–2 хоногт өрхийн эмчдээ үзүүлэхийг зөвлөж байна.",
    RiskLevel.LOW: "Гэртээ анхаарал хандуулж, анхаарах шинж илэрвэл эмчид хандана уу.",
}

_NUMERIC_CHECKS = {
    "temp_gte": lambda f, v: f.temperature_c is not None and f.temperature_c >= v,
    "bp_sys_gte": lambda f, v: f.bp_systolic is not None and f.bp_systolic >= v,
    "severity_gte": lambda f, v: f.severity is not None and f.severity >= v,
    "duration_days_gte": lambda f, v: f.duration_days is not None and f.duration_days >= v,
    "age_months_lt": lambda f, v: _age_months(f) is not None and _age_months(f) < v,
    "age_years_gte": lambda f, v: f.age_years is not None and f.age_years >= v,
}


def _age_months(f: Facts) -> Optional[float]:
    if f.age_months is not None:
        return f.age_months
    if f.age_years is not None:
        return f.age_years * 12
    return None


class TriageAgent:
    def __init__(self, rules_path: Path, templates_path: Path):
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.version = str(rules.get("version", "?"))
        self.rules: list[dict] = rules.get("rules", [])
        self.modifiers: list[dict] = rules.get("modifiers", [])
        self.templates = yaml.safe_load(templates_path.read_text(encoding="utf-8"))
        self._validate()

    def _validate(self) -> None:
        known_templates = set(self.templates.get("templates", {}))
        for r in self.rules:
            assert "id" in r and "risk" in r, f"rule missing id/risk: {r}"
            RiskLevel(r["risk"])
            has_condition = any(k in r for k in ("all", "any", *_NUMERIC_CHECKS))
            assert has_condition, f"rule {r['id']} has no condition"
            if r["risk"] == "EMERGENCY":
                assert r.get("template") in known_templates, f"rule {r['id']} needs a valid template"

    # ------------------------------------------------------------ matching
    @staticmethod
    def _rule_fires(rule: dict, concepts: set[str], facts: Facts, has_symptom: bool) -> bool:
        if rule.get("all") and not all(c in concepts for c in rule["all"]):
            return False
        if rule.get("any") and not any(c in concepts for c in rule["any"]):
            return False
        for key, check in _NUMERIC_CHECKS.items():
            if key in rule and not check(facts, rule[key]):
                return False
        if rule.get("requires_symptom") and not has_symptom:
            return False
        return True

    @staticmethod
    def _modifier_applies(mod: dict, concepts: set[str], facts: Facts) -> bool:
        # modifiers use OR semantics: any listed concept OR any numeric condition
        if mod.get("any") and any(c in concepts for c in mod["any"]):
            return True
        return any(key in mod and check(facts, mod[key]) for key, check in _NUMERIC_CHECKS.items())

    def assess(
        self,
        concepts: Iterable[str],
        facts: Facts,
        has_symptom: bool,
        llm_risk: Optional[RiskLevel] = None,
        llm_flags: Iterable[str] = (),
    ) -> TriageResult:
        cset = set(concepts)
        fired = [r for r in self.rules if self._rule_fires(r, cset, facts, has_symptom)]
        risk = max_risk(*(RiskLevel(r["risk"]) for r in fired)) if fired else RiskLevel.LOW

        modifiers = [m for m in self.modifiers if self._modifier_applies(m, cset, facts)]
        if modifiers and has_symptom and risk in (RiskLevel.LOW, RiskLevel.MODERATE):
            risk = raise_one_level(risk)

        source = "rules"
        flags = [RedFlag(rule_id=r["id"], label_mn=r["label_mn"], risk=RiskLevel(r["risk"])) for r in fired]
        if llm_risk is not None and risk_rank(llm_risk) > risk_rank(risk):
            # LLM may only raise the risk — never lower it
            risk = llm_risk
            source = "rules+llm"
            flags.append(RedFlag(rule_id="LLM_SIGNAL", label_mn="; ".join(llm_flags) or "LLM-ийн анхааруулга", risk=llm_risk))

        template = None
        if risk == RiskLevel.EMERGENCY:
            template = next((r["template"] for r in fired if r["risk"] == "EMERGENCY"), "general")
        flags.sort(key=lambda f: -risk_rank(f.risk))
        return TriageResult(
            risk_level=risk,
            red_flags=flags,
            modifiers=[m["label_mn"] for m in modifiers],
            template=template,
            recommended_action=ACTIONS[risk],
            source=source,
        )

    # ----------------------------------------------------- emergency output
    def render_emergency(self, triage: TriageResult) -> str:
        common = self.templates.get("common", {})
        tpl = self.templates["templates"].get(triage.template or "general") or self.templates["templates"]["general"]
        call_line = tpl.get("call_line", common.get("call_line", ""))
        footer = tpl.get("footer", common.get("footer", ""))
        steps = "\n".join(f"- {s}" for s in tpl.get("steps", []))
        if tpl.get("style") == "crisis":
            return f"💛 {tpl['headline']}\n\n{call_line}\n\n**Одоо хийж болох зүйлс:**\n{steps}\n\n{footer}"
        flags = "; ".join(f.label_mn for f in triage.red_flags if f.risk == RiskLevel.EMERGENCY)
        return (
            f"🚨 **ЯАРАЛТАЙ: {tpl['headline']}**\n\n{call_line}\n\n"
            f"**Түргэн тусламж ирэх хүртэл:**\n{steps}\n\n"
            f"⚠️ Илэрсэн анхаарах шинж: {flags}\n\n{footer}"
        )
