"""
Typed contracts between agents. Every agent reads a few fields and writes
ONLY its own section. No agent calls another agent directly — the
orchestrator passes these objects along.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------- risk
class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"


_ORDER = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2, RiskLevel.EMERGENCY: 3}


def risk_rank(level: RiskLevel) -> int:
    return _ORDER[level]


def max_risk(*levels: Optional[RiskLevel]) -> RiskLevel:
    present = [lv for lv in levels if lv is not None]
    return max(present, key=risk_rank) if present else RiskLevel.LOW


def raise_one_level(level: RiskLevel, cap: RiskLevel = RiskLevel.HIGH) -> RiskLevel:
    """LOW->MODERATE, MODERATE->HIGH. Never raises to EMERGENCY by itself."""
    nxt = {RiskLevel.LOW: RiskLevel.MODERATE, RiskLevel.MODERATE: RiskLevel.HIGH}.get(level, level)
    return nxt if risk_rank(nxt) <= risk_rank(cap) else level


MessageType = Literal["ANSWER", "FOLLOWUP", "EMERGENCY", "FALLBACK", "GREETING", "OUT_OF_SCOPE", "CLARIFY", "THANKS"]
Intent = Literal["symptom_check", "medication", "general_info", "greeting", "thanks", "out_of_scope", "unclear"]


# --------------------------------------------------------- input processor
class Facts(BaseModel):
    """Numbers & details pulled from text with deterministic regexes."""
    duration_days: Optional[float] = None
    duration_text: Optional[str] = None
    severity: Optional[int] = None            # 1-10
    temperature_c: Optional[float] = None
    age_years: Optional[float] = None
    age_months: Optional[float] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    sex: Optional[Literal["male", "female"]] = None

    def merged_with(self, newer: "Facts") -> "Facts":
        data = self.model_dump()
        for k, v in newer.model_dump().items():
            if v is not None:
                data[k] = v
        return Facts(**data)


class ConceptHit(BaseModel):
    concept: str
    label_mn: str
    type: str
    negated: bool = False
    source: Literal["lexicon", "llm"] = "lexicon"


class NormalizedInput(BaseModel):
    original: str
    redacted: str                 # personal info removed — the only text we store
    display_text: str             # Latin transliterated to Cyrillic (for display / LLM)
    folded_text: str              # internal matching form
    script: str
    pii_types: list[str] = Field(default_factory=list)
    concepts: list[ConceptHit] = Field(default_factory=list)
    facts: Facts = Field(default_factory=Facts)

    @property
    def positive_concepts(self) -> list[str]:
        return [c.concept for c in self.concepts if not c.negated]

    @property
    def negated_concepts(self) -> list[str]:
        return [c.concept for c in self.concepts if c.negated]


# --------------------------------------------------------------- triage
class RedFlag(BaseModel):
    rule_id: str
    label_mn: str
    risk: RiskLevel


class TriageResult(BaseModel):
    risk_level: RiskLevel
    red_flags: list[RedFlag] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    template: Optional[str] = None          # emergency template key
    recommended_action: str = ""
    source: str = "rules"                    # rules | rules+llm


# -------------------------------------------------------------- symptom
class SymptomAnalysis(BaseModel):
    intent: Intent
    symptoms: list[str] = Field(default_factory=list)       # positive symptom concept ids
    concepts: list[str] = Field(default_factory=list)       # all positive concept ids (session-merged)
    negated_concepts: list[str] = Field(default_factory=list)
    facts: Facts = Field(default_factory=Facts)
    missing_slots: list[str] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    llm_risk: Optional[RiskLevel] = None
    llm_red_flags: list[str] = Field(default_factory=list)
    source: str = "rules"                                    # rules | rules+llm


# ------------------------------------------------------------------ RAG
class KBSection(BaseModel):
    title: str
    text: str


class RetrievedDoc(BaseModel):
    doc_id: str
    title: str
    category: str
    source: str
    source_url: str
    source_tier: int
    status: str
    version: str
    score: float
    concept_overlap: int
    best_cosine: float
    sections: list[KBSection] = Field(default_factory=list)

    def section(self, name_start: str) -> Optional[str]:
        for s in self.sections:
            if s.title.lower().startswith(name_start.lower()):
                return s.text
        return None


class RetrievalResult(BaseModel):
    query: str
    query_concepts: list[str]
    docs: list[RetrievedDoc] = Field(default_factory=list)
    sufficient: bool = False
    top_score: float = 0.0
    reason: str = ""
    index_name: str = ""


# -------------------------------------------------------------- response
class DraftResponse(BaseModel):
    text: str
    cited_doc_ids: list[str] = Field(default_factory=list)
    generator: Literal["template", "llm"] = "template"
    attempt: int = 1


class SafetyIssue(BaseModel):
    code: str
    severity: Literal["BLOCK", "REVISE"]
    detail: str


class SafetyVerdict(BaseModel):
    verdict: Literal["SAFE", "REVISE", "BLOCK"]
    issues: list[SafetyIssue] = Field(default_factory=list)
    checked_by: list[str] = Field(default_factory=list)


# --------------------------------------------------------------- session
class SessionState(BaseModel):
    session_id: str
    turn_count: int = 0
    concepts: list[str] = Field(default_factory=list)
    negated_concepts: list[str] = Field(default_factory=list)
    facts: Facts = Field(default_factory=Facts)
    asked_slots: list[str] = Field(default_factory=list)
    pending_slots: list[str] = Field(default_factory=list)
    followup_rounds: int = 0
    max_risk: RiskLevel = RiskLevel.LOW
    last_message_type: Optional[str] = None
    # Concepts / facts of EARLIER, already-answered topics. Not used for retrieval
    # or follow-ups, but Agent 1 (triage) still sees them: "цээж өвдөж байна" ->
    # answer -> "амьсгал давчдаж байна" must still be an EMERGENCY.
    background_concepts: list[str] = Field(default_factory=list)
    background_facts: Facts = Field(default_factory=Facts)

    def triage_concepts(self, concepts: list[str]) -> list[str]:
        return list(dict.fromkeys(self.background_concepts + concepts))

    def triage_facts(self, facts: "Facts") -> "Facts":
        return self.background_facts.merged_with(facts)


# ------------------------------------------------------------------- API
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=64)


class SourceOut(BaseModel):
    doc_id: str
    title: str
    source: str
    url: str
    tier: int
    status: str
    score: float


class AgentStep(BaseModel):
    """One visible step of the multi-agent pipeline (shown in the UI)."""
    agent: str                                   # input | triage | symptom | triage_merge | rag | response | safety
    title: str
    status: Literal["ok", "alert", "skipped", "fallback"] = "ok"
    detail: str = ""
    ms: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    turn_id: int
    reply: str
    message_type: MessageType
    risk_level: RiskLevel
    red_flags: list[str] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    mode: str
    content_notice: Optional[str] = None
    latency_ms: int = 0
    agents: list[AgentStep] = Field(default_factory=list)


FeedbackReason = Literal["incorrect", "not_relevant", "did_not_understand", "not_enough", "other"]


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., max_length=64)
    turn_id: int
    helpful: bool
    reason: Optional[FeedbackReason] = None
    comment: Optional[str] = Field(default=None, max_length=500)


class TurnTrace(BaseModel):
    """Everything that happened in one turn — stored for debugging & evaluation."""
    input: dict[str, Any]
    triage_rules: Optional[dict[str, Any]] = None
    symptoms: Optional[dict[str, Any]] = None
    triage_final: Optional[dict[str, Any]] = None
    retrieval: Optional[dict[str, Any]] = None
    drafts: list[dict[str, Any]] = Field(default_factory=list)
    safety: list[dict[str, Any]] = Field(default_factory=list)
    message_type: str = ""
    versions: dict[str, str] = Field(default_factory=dict)
    llm_usage: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    steps: list[AgentStep] = Field(default_factory=list)
