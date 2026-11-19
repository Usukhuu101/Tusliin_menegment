"""
AGENT 5 — SAFETY REVIEW AGENT (final gate)

Layer A — deterministic checks (always, free, instant):
  fabricated citations / URLs, medication doses, definitive diagnosis,
  discouraging care, stop-your-medicine advice, risk-level consistency,
  required sections, disclaimer, Mongolian-script ratio, length, PII echo.
Layer B — LLM judge (only for LLM-written drafts, when an API key exists).

Policy (fail closed):
  * template drafts (assembled from reviewed KB text) need Layer A = SAFE
  * LLM drafts need Layer A = SAFE  AND  Layer B = SAFE.
    If the judge is unavailable, the LLM draft is NOT shown (REVISE ->
    orchestrator falls back to the template answer).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..models.schemas import DraftResponse, NormalizedInput, RetrievalResult, RiskLevel, SafetyIssue, SafetyVerdict, TriageResult
from ..services.llm import LLMError, LLMProvider, UsageTracker

log = logging.getLogger(__name__)

_DOSE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:мг|mg|мкг|mcg|µg|мл|ml|гр|г|ш|ширхэг|шахмал|таблет|капсул|дусал|units?|iu)\b"
    r"|өдөрт\s*\d+\s*(?:удаа|ширхэг|шахмал|таблет)"
    r"|\d+\s*цаг\s*тутам"
    r"|(?:нэг|хоёр|гурав|1|2|3)\s*(?:шахмал|таблет|капсул)\s*(?:уу|ууна|уугаарай)",
    re.IGNORECASE,
)
_DIAGNOSIS = re.compile(
    r"\b(?:танд|таныг)\s+[^.!?\n]{0,40}(?:өвчин|үрэвсэл|хавдар|шигдээс|харвалт|халдвар|мигрень)\s+(?:байна|туссан|илэрсэн|мөн)\b"
    r"|\bта\s+[^.!?\n]{0,30}(?:өвчтэй|өвчин туссан)\s+байна"
    r"|\bэнэ\s+бол\s+[^.!?\n]{0,30}(?:өвчин|үрэвсэл|хавдар|мигрень|шигдээс)\b"
    r"|оношлогдлоо|онош\s+нь\s+[^.!?\n]{0,30}\s+мөн|100\s*%\s*(?:итгэлтэй|магадлал)",
    re.IGNORECASE,
)
_DISCOURAGE = re.compile(
    r"(?:эмчид|эмнэлэгт)\s+(?:үзүүлэх|хандах|очих)\s+(?:шаардлагагүй|хэрэггүй)"
    r"|103[^.\n]{0,25}(?:хэрэггүй|шаардлагагүй)"
    r"|түргэн\s+тусламж\s+(?:дуудах\s+)?(?:хэрэггүй|шаардлагагүй)",
    re.IGNORECASE,
)
_FALSE_REASSURE = re.compile(r"санаа\s+зовох\s+(?:хэрэггүй|зүйлгүй)|ямар\s+ч\s+аюулгүй", re.IGNORECASE)
_STOP_MED = re.compile(
    r"\bэм(?:ээ|ийг|ийнхээ)?\s+(?:уухаа\s+)?(?:зогсоо|боли|хая)(?![а-яөүё]*гүй)[а-яөүё]*"
    r"|(?:антибиотик|парацетамол|ибупрофен|аспирин|анальгин)\s+(?:уу(?:гаарай|на уу|х хэрэгтэй)|хэрэглэ(?:ээрэй|нэ үү))",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s\]]+")


def _clean_url(u: str) -> str:
    """Trim trailing punctuation and UNBALANCED ')' — WHO URLs contain '(…)'."""
    u = u.rstrip(".,;:")
    while u.endswith(")") and u.count("(") < u.count(")"):
        u = u[:-1]
    return u
_DOC_ID = re.compile(r"\bMED_[A-Z0-9_]+")


def cited_doc_ids(text: str) -> set[str]:
    """Every knowledge-base id mentioned anywhere — "[A]", "[A, B]" or bare —
    so a fabricated id cannot hide inside a multi-id bracket."""
    return set(_DOC_ID.findall(text))
_PII_ECHO = re.compile(r"(?<!\d)[5-9]\d{3}[\s-]?\d{4}(?!\d)|\b[А-ЯӨҮЁа-яөүё]{2}\d{8}\b")
_CYR = re.compile(r"[а-яёөү]", re.IGNORECASE)
_LAT = re.compile(r"[a-z]", re.IGNORECASE)
REQUIRED_HEADERS = ["🩺", "🔎", "💡", "⚠️", "🏥", "📚"]


class JudgeIssue(BaseModel):
    code: str
    severity: Literal["BLOCK", "REVISE"]
    detail: str


class JudgeOutput(BaseModel):
    verdict: Literal["SAFE", "REVISE", "BLOCK"]
    issues: list[JudgeIssue] = Field(default_factory=list)


class SafetyAgent(UsageTracker):
    def __init__(self, llm: LLMProvider, model: str, prompt_path: Path):
        UsageTracker.__init__(self)
        self.llm = llm
        self.model = model
        self.prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    # ------------------------------------------------------- deterministic
    def check_rules(self, text: str, *, risk: RiskLevel, allowed_doc_ids: set[str], allowed_urls: set[str],
                    require_sections: bool) -> list[SafetyIssue]:
        issues: list[SafetyIssue] = []

        def add(code, sev, detail):
            issues.append(SafetyIssue(code=code, severity=sev, detail=detail))

        unknown = sorted(cited_doc_ids(text) - allowed_doc_ids)
        if unknown:
            add("FABRICATED_CITATION", "REVISE", f"Cited ids not in retrieved documents: {unknown}")
        bad_urls = [u for u in map(_clean_url, _URL.findall(text)) if u not in allowed_urls]
        if bad_urls:
            add("UNKNOWN_URL", "REVISE", f"URLs not from retrieved sources: {bad_urls[:3]}")
        if m := _DOSE.search(text):
            add("MEDICATION_DOSE", "BLOCK", f"Dose / dosing schedule found: '{m.group(0)}'")
        if m := _DIAGNOSIS.search(text):
            add("DEFINITIVE_DIAGNOSIS", "BLOCK", f"Definitive diagnosis wording: '{m.group(0)[:60]}'")
        if m := _DISCOURAGE.search(text):
            add("DISCOURAGES_CARE", "BLOCK", f"Discourages seeking care: '{m.group(0)}'")
        if m := _STOP_MED.search(text):
            add("MEDICATION_ADVICE", "BLOCK", f"Start/stop medication advice: '{m.group(0)}'")
        if m := _FALSE_REASSURE.search(text):
            add("FALSE_REASSURANCE", "REVISE", f"False reassurance: '{m.group(0)}'")
        low = text.lower()
        if risk == RiskLevel.EMERGENCY and "103" not in text:
            add("RISK_MISMATCH", "BLOCK", "EMERGENCY risk but no instruction to call 103")
        if risk == RiskLevel.HIGH and not ("эмч" in low and any(k in low for k in ("өнөөдөр", "яаралтай", "103"))):
            add("RISK_MISMATCH", "REVISE", "HIGH risk but no clear 'see a doctor today / 103' instruction")
        if require_sections:
            missing = [h for h in REQUIRED_HEADERS if h not in text]
            if missing:
                add("MISSING_SECTIONS", "REVISE", f"Missing sections: {missing}")
            if "ерөнхий мэдээлэл" not in low and "эмчийн үзлэг" not in low:
                add("MISSING_DISCLAIMER", "REVISE", "Disclaimer missing")
        body = _URL.sub("", text)
        cyr, lat = len(_CYR.findall(body)), len(_LAT.findall(body))
        if cyr + lat > 0 and cyr / (cyr + lat) < 0.6:
            add("LANGUAGE", "REVISE", "Answer is not mainly in Mongolian Cyrillic")
        if len(text) > 4500:
            add("TOO_LONG", "REVISE", f"Answer too long ({len(text)} chars)")
        if _PII_ECHO.search(text):
            add("PII_ECHO", "REVISE", "Answer contains a phone number / national ID")
        return issues

    # ------------------------------------------------------------- review
    def review(self, draft: DraftResponse, *, triage: TriageResult, retrieval: RetrievalResult | None,
               norm: NormalizedInput, require_sections: bool = True) -> SafetyVerdict:
        self.last_usage = []
        docs = retrieval.docs if retrieval else []
        allowed_ids = {d.doc_id for d in docs}
        allowed_urls = {d.source_url for d in docs}
        issues = self.check_rules(draft.text, risk=triage.risk_level, allowed_doc_ids=allowed_ids,
                                  allowed_urls=allowed_urls, require_sections=require_sections)
        checked = ["rules"]
        if draft.generator == "llm" and docs and not (cited_doc_ids(draft.text) & allowed_ids):
            # an LLM answer must be traceable to at least one retrieved document
            issues.append(SafetyIssue(code="NO_CITATION", severity="REVISE",
                                      detail="Answer cites none of the retrieved documents"))
        if draft.generator == "llm" and not any(i.severity == "BLOCK" for i in issues):
            judge = self._judge(draft, triage, retrieval, norm)
            checked.append("llm_judge")
            if judge is None:
                issues.append(SafetyIssue(code="JUDGE_UNAVAILABLE", severity="REVISE",
                                          detail="LLM safety review failed — LLM draft cannot be released"))
            else:
                issues.extend(SafetyIssue(code=i.code, severity=i.severity, detail=i.detail) for i in judge.issues)
                if judge.verdict == "BLOCK" and not any(i.severity == "BLOCK" for i in issues):
                    issues.append(SafetyIssue(code="JUDGE_BLOCK", severity="BLOCK", detail="LLM judge blocked the draft"))
                elif judge.verdict == "REVISE" and not issues:
                    issues.append(SafetyIssue(code="JUDGE_REVISE", severity="REVISE", detail="LLM judge requested revision"))
        if any(i.severity == "BLOCK" for i in issues):
            verdict = "BLOCK"
        elif issues:
            verdict = "REVISE"
        else:
            verdict = "SAFE"
        return SafetyVerdict(verdict=verdict, issues=issues, checked_by=checked)

    def _judge(self, draft, triage, retrieval, norm) -> JudgeOutput | None:
        if not self.llm.available:
            return None
        docs = "\n".join(
            f"<document id='{d.doc_id}'>\n" + "\n".join(f"## {s.title}\n{s.text}" for s in d.sections) + "\n</document>"
            for d in (retrieval.docs if retrieval else [])
        )
        user = (
            f"<user_message>{norm.redacted}</user_message>\n"
            f"<triage_risk>{triage.risk_level.value}</triage_risk>\n"
            f"<retrieved_documents>\n{docs}\n</retrieved_documents>\n"
            f"<draft_answer>\n{draft.text}\n</draft_answer>"
        )
        try:
            out, usage = self.llm.generate_structured(system=self.prompt, messages=[{"role": "user", "content": user}],
                                                      model=self.model, schema=JudgeOutput)
        except LLMError as e:
            log.warning("Safety judge unavailable: %s", e)
            return None
        self.last_usage.append({"agent": "safety", "model": self.model, **usage})
        return out
