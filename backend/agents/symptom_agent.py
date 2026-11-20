"""
AGENT 2 — SYMPTOM ANALYSIS AGENT

Turns the user's (possibly messy) Mongolian into structured data:
  intent, symptom concepts, duration, severity, temperature, age, ...
and decides which follow-up questions are still needed.

Layers:
  1. Deterministic: lexicon concepts + regex facts (always runs, free, offline)
  2. Answers to the previous follow-up questions (e.g. a bare "7" -> severity)
  3. Optional LLM structured extraction (only ADDS information; it can raise
     risk via `llm_risk`, never lower it — that is enforced in Agent 1)
  4. Follow-up planning from data/protocols/symptom_protocols.yaml
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from ..models.schemas import Facts, NormalizedInput, RiskLevel, SessionState, SymptomAnalysis
from ..nlp.lexicon import ConceptLexicon
from ..services.llm import LLMError, LLMProvider, UsageTracker

log = logging.getLogger(__name__)

SYMPTOM_TYPES = {"symptom", "red_flag"}
_GREETING = re.compile(r"^\s*(саин|сн\s*уу|мэнд|hello|hi\b|hey\b|good (morning|evening))")
_THANKS = re.compile(r"баярлалаа|баярлаа|баярлала|thanks|thank you|thx")
_QUESTION = re.compile(r"гэж\s*ю|юу\s*вэ|\bю\s*вэ|яаж|хэрхэн|шалтгаан|сэргиил|эмчилгээ|юунаас|аюултаи\s*юу|халдвартаи|what is|how (to|do)|why")
_HEALTH = re.compile(r"эруул|овчин|эмч|эмнэлэг|шинж|овд|бие\s*муу|вакцин|дархлаа|ноир|стресс|сэтгэл|дасгал|жирэмс|хоол|health|doctor|sick")
# A concrete health TOPIC (body part, organ, disease word, symptom verb) that the
# lexicon does not know yet. Such a message is clear — it is not out of scope and
# the user should hear "no reliable information yet", not "I did not understand".
# Patterns are on FOLDED text (ө->о, ү->у, й->и, е->э, ь dropped).
_HEALTH_TOPIC = re.compile(
    r"\b(?:чих|арс|жин|яс)\b"                                            # short words: whole word only
    r"|\bус\s+(?:их\s+)?уна"                                             # үс унах (hair loss); ус = water
    r"|\b(?:шуд|чихн|чихэнд|чихээ|нуд|хамар|хузуу|уушг|элэг|боор|давсаг|умаи|бамбаи|булчирхаи|тархи|"
    r"арсн|арсан|хумс|ясн|уэ\s+моч|шээ|отгон|баас|цус|сарин\s+тэмдэг|бэлгии|хавд|хатаа|хатаж|хорс|батга|улаи|"
    r"шарх|урэвс|халдвар|вирус|нян|хорт|эмгэг|дутагдал|багадалт|ихсэлт|витамин|жинг|тураал|таргал|ковид|"
    r"короно|астма|багтраа|дааг|хуруу|шагаи)"
    r"|\b(?:tooth|teeth|ear|eye|eyes|kidney|skin|cancer|vitamin|asthma|covid|diabetes|anemia|leukemia|thyroid|"
    r"disease|pain|infection)\b"
)


def mentions_health_topic(norm: NormalizedInput) -> bool:
    """True when the message names a concrete health topic (even one the lexicon
    does not know). Checked on folded Cyrillic AND on the raw text (English words
    in an all-Latin message are transliterated in folded_text)."""
    return bool(_HEALTH_TOPIC.search(norm.folded_text) or _HEALTH_TOPIC.search(norm.original.lower()))
_NEGATIVE_ANSWER = re.compile(r"\b(угуи|баихгуи|бхгуи|алга|биш|no|nope|none)\b")
_AFFIRMATIVE = re.compile(r"\b(тиим|тиимээ|тиимэ|бии)\b")          # folded "тийм", "бий"
_EN_YES = re.compile(r"\b(yes|yeah|yep)\b")
_EN_NO = re.compile(r"\b(no|nope|none|not)\b")
_TIME_UNITS = r"(?!\s*(?:долоо|хоног|одор|удур|цаг|минут|сар|жил|нас|наст|хэм|градус|°|[.,]\d))"


class LLMExtraction(BaseModel):
    intent: Literal["symptom_check", "medication", "general_info", "greeting", "thanks", "out_of_scope", "unclear"]
    concepts: list[str] = Field(default_factory=list)
    red_flags_detected: list[str] = Field(default_factory=list)
    llm_risk: Literal["LOW", "MODERATE", "HIGH", "EMERGENCY"]
    duration_days: Optional[float] = None
    severity_1_10: Optional[int] = None
    temperature_c: Optional[float] = None
    age_years: Optional[float] = None
    normalized_text_mn: str = ""


class SymptomAgent(UsageTracker):
    def __init__(self, lexicon: ConceptLexicon, protocols_path: Path, llm: LLMProvider,
                 model: str, prompt_path: Path, max_questions: int = 3):
        UsageTracker.__init__(self)
        self.lexicon = lexicon
        data = yaml.safe_load(protocols_path.read_text(encoding="utf-8"))
        self.default_protocol: list[dict] = data["default"]
        self.protocols: dict[str, list[dict]] = data.get("protocols", {})
        self.llm = llm
        self.model = model
        self.prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        self.max_questions = max_questions

    # ------------------------------------------------------------------ API
    def analyze(self, norm: NormalizedInput, session: SessionState) -> SymptomAnalysis:
        self.last_usage = []
        facts_now = self._answers_to_pending(norm, session, norm.facts.model_copy())

        # A plain "тийм / yes" to a red-flag screening question is itself a red flag,
        # even if the user does not name which symptom (safety: never ignore it).
        current = norm.positive_concepts + self._screen_answer(norm, session)
        concepts = list(dict.fromkeys(session.concepts + current))
        negated = [c for c in dict.fromkeys(session.negated_concepts + norm.negated_concepts) if c not in concepts]
        facts = session.facts.merged_with(facts_now)
        intent = self._intent(norm, current, session)

        analysis = SymptomAnalysis(
            intent=intent,
            concepts=concepts,
            negated_concepts=negated,
            facts=facts,
            symptoms=[c for c in concepts if self.lexicon.type_of(c) in SYMPTOM_TYPES],
        )
        if self.llm.available and intent not in ("greeting", "thanks"):
            self._llm_enrich(norm, analysis)

        if analysis.intent == "symptom_check":
            primary = self._primary_symptom(current, analysis.symptoms)
            if primary:
                analysis.missing_slots, analysis.followup_questions = self._plan_followups(primary, analysis.facts, session)
        return analysis

    # -------------------------------------------------------------- intent
    def _intent(self, norm: NormalizedInput, current: list[str], session: SessionState) -> str:
        text = norm.folded_text
        types = {self.lexicon.type_of(c) for c in current}
        if not current:
            if _GREETING.search(text) and len(text.split()) <= 6:
                return "greeting"
            if _THANKS.search(text):
                return "thanks"
            if session.pending_slots:
                return "symptom_check"          # user is answering our questions
            if _HEALTH.search(text) or _QUESTION.search(text) or mentions_health_topic(norm):
                return "unclear"
            return "out_of_scope"
        if "medication" in types:
            return "medication"
        if types & SYMPTOM_TYPES:
            return "symptom_check"
        if _QUESTION.search(text) or types & {"condition", "exposure"}:
            return "general_info"
        return "symptom_check" if session.concepts else "unclear"

    # ----------------------------------------------- follow-up answer parsing
    def _answers_to_pending(self, norm: NormalizedInput, session: SessionState, f: Facts) -> Facts:
        pending = set(s.split(":")[0] for s in session.pending_slots)
        if not pending:
            return f
        text = norm.folded_text
        free_numbers = [m.group(1) for m in re.finditer(r"\b(\d{1,3}(?:[.,]\d)?)\b" + _TIME_UNITS, text)]
        for raw in free_numbers:
            n = float(raw.replace(",", "."))
            if "severity" in pending and f.severity is None and 1 <= n <= 10 and n == int(n):
                f.severity = int(n)
            elif "temperature" in pending and f.temperature_c is None and 34 <= n <= 43:
                f.temperature_c = n
            elif "age" in pending and f.age_years is None and 0 < n <= 120:
                f.age_years = n
            elif "duration" in pending and f.duration_days is None and 0 < n <= 365:
                f.duration_days, f.duration_text = n, f"{int(n)} хоног"
        return f

    def _screen_answer(self, norm: NormalizedInput, session: SessionState) -> list[str]:
        """Concept to add when the user answers YES to a pending red-flag screening question."""
        pending = [s for s in session.pending_slots if s.startswith("red_flag_screen:")]
        if not pending:
            return []
        original = norm.original.lower()
        negative = _NEGATIVE_ANSWER.search(norm.folded_text) or _EN_NO.search(original)
        positive = _AFFIRMATIVE.search(norm.folded_text) or _EN_YES.search(original)
        if not positive or negative:
            return []
        primary = pending[0].split(":", 1)[1]
        protocol = self.protocols.get(primary) or self.default_protocol
        item = next((i for i in protocol if i["slot"] == "red_flag_screen"), {})
        return [item.get("positive_concept", "screen_positive")]

    @staticmethod
    def negative_screen_answer(norm: NormalizedInput) -> bool:
        return bool(_NEGATIVE_ANSWER.search(norm.folded_text))

    # -------------------------------------------------------- follow-ups
    def _primary_symptom(self, current: list[str], symptoms: list[str]) -> Optional[str]:
        cur = [c for c in current if c in symptoms]
        for c in cur + symptoms:
            if c in self.protocols:
                return c
        return (cur or symptoms or [None])[0]

    def _plan_followups(self, primary: str, facts: Facts, session: SessionState) -> tuple[list[str], list[str]]:
        protocol = self.protocols.get(primary) or self.default_protocol
        filled = {
            "duration": facts.duration_days is not None,
            "severity": facts.severity is not None,
            "temperature": facts.temperature_c is not None,
            "age": facts.age_years is not None or facts.age_months is not None,
            "red_flag_screen": False,
        }
        missing, questions = [], []
        for item in protocol:
            slot = item["slot"]
            key = f"red_flag_screen:{primary}" if slot == "red_flag_screen" else slot
            if key in session.asked_slots or filled.get(slot, False):
                continue
            missing.append(key)
            questions.append(item["question"])
        return missing[: self.max_questions], questions[: self.max_questions]

    # --------------------------------------------------------------- LLM
    def _llm_enrich(self, norm: NormalizedInput, analysis: SymptomAnalysis) -> None:
        allowed = "\n".join(f"- {c.id}: {c.label_mn} ({c.type})" for c in self.lexicon.concepts.values())
        user = (
            f"<allowed_concepts>\n{allowed}\n</allowed_concepts>\n"
            f"<user_message>\n{norm.redacted}\n</user_message>\n"
            f"<transliterated>\n{norm.display_text}\n</transliterated>"
        )
        try:
            out, usage = self.llm.generate_structured(
                system=self.prompt, messages=[{"role": "user", "content": user}],
                model=self.model, schema=LLMExtraction,
            )
        except LLMError as e:
            log.info("LLM extraction unavailable, using rules only: %s", e)
            return
        self.last_usage.append({"agent": "symptom", "model": self.model, **usage})
        valid = set(self.lexicon.concepts)
        added = [c for c in out.concepts if c in valid and c not in analysis.concepts and c not in analysis.negated_concepts]
        analysis.concepts.extend(added)
        analysis.symptoms = [c for c in analysis.concepts if self.lexicon.type_of(c) in SYMPTOM_TYPES]
        f = analysis.facts
        if f.duration_days is None and out.duration_days is not None:
            f.duration_days, f.duration_text = out.duration_days, f"{out.duration_days:g} хоног"
        if f.severity is None and out.severity_1_10 and 1 <= out.severity_1_10 <= 10:
            f.severity = out.severity_1_10
        if f.temperature_c is None and out.temperature_c and 34 <= out.temperature_c <= 43:
            f.temperature_c = out.temperature_c
        if f.age_years is None and out.age_years is not None and 0 <= out.age_years <= 120:
            f.age_years = out.age_years
        analysis.llm_risk = RiskLevel(out.llm_risk)
        analysis.llm_red_flags = [c for c in out.red_flags_detected if c in valid]
        if analysis.intent in ("unclear", "out_of_scope") and out.intent in ("symptom_check", "medication", "general_info"):
            analysis.intent = out.intent
        analysis.source = "rules+llm"
