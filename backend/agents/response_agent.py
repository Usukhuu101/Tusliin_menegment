"""
AGENT 4 — RESPONSE GENERATION AGENT

Two modes, same 6-section output format:
  * template (default / offline): the answer is ASSEMBLED from the reviewed
    knowledge-base sections. Nothing is invented, so hallucination is impossible.
  * llm (when an API key is configured): the LLM writes a friendlier answer,
    grounded ONLY in the retrieved documents, with [DOC_ID] citations.
    If the LLM fails or keeps failing the safety review, the orchestrator
    falls back to template mode.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..models.schemas import (DraftResponse, NormalizedInput, RetrievalResult, RetrievedDoc, RiskLevel,
                              SafetyIssue, SymptomAnalysis, TriageResult)
from ..nlp.lexicon import ConceptLexicon
from ..services.llm import LLMError, LLMProvider, UsageTracker
from .safety_agent import cited_doc_ids  # one shared citation extractor

DISCLAIMER = "ℹ️ Энэ бол ерөнхий мэдээлэл бөгөөд эмчийн үзлэг, оношийг орлохгүй."

RISK_LEAD = {
    RiskLevel.LOW: "Одоогийн мэдээллээр яаралтай шинж илрээгүй байна. Гэхдээ доорх тохиолдолд эмчид хандаарай:",
    RiskLevel.MODERATE: "Таны хэлсэн шинжийг харгалзан **ойрын 1–2 хоногт өрхийн эмчдээ үзүүлэхийг** зөвлөж байна.",
    RiskLevel.HIGH: "⚠️ Таны хэлсэн шинж анхаарал шаардаж байна. **Өнөөдөр аль болох түрүүн эмчид үзүүлээрэй.** "
                    "Байдал муудвал **103** дугаарт залгаарай.",
    RiskLevel.EMERGENCY: "🚨 **Яг одоо 103 дугаарт залгаарай.**",
}


def _bullets(text: Optional[str], limit: int) -> list[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [ln[2:].strip() for ln in lines if ln.startswith(("- ", "* "))]
    return bullets[:limit]


def _tail_sentence(text: Optional[str]) -> Optional[str]:
    """Non-bullet closing line of a section (e.g. 'Эдгээр шинж илэрвэл ... 103')."""
    if not text:
        return None
    others = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(("- ", "* "))]
    return others[-1] if others else None


def _paragraph(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    para = " ".join(ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("- "))
    if len(para) <= max_chars:
        return para
    cut = para[:max_chars]
    return cut[: cut.rfind(".") + 1] if "." in cut else cut + "…"


class ResponseAgent(UsageTracker):
    def __init__(self, lexicon: ConceptLexicon, llm: LLMProvider, model: str, effort: str, prompt_path: Path):
        UsageTracker.__init__(self)
        self.lexicon = lexicon
        self.llm = llm
        self.model = model
        self.effort = effort
        self.system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    # ------------------------------------------------------------- public
    def generate(self, *, norm: NormalizedInput, analysis: SymptomAnalysis, triage: TriageResult,
                 retrieval: RetrievalResult, feedback: Optional[list[SafetyIssue]] = None,
                 attempt: int = 1, use_llm: bool = True) -> DraftResponse:
        self.last_usage = []
        if use_llm and self.llm.available:
            try:
                return self._generate_llm(norm, analysis, triage, retrieval, feedback, attempt)
            except LLMError:
                pass  # fall through to the deterministic answer
        return self.compose_template(norm=norm, analysis=analysis, triage=triage, retrieval=retrieval)

    # ------------------------------------------------------- template mode
    def _summary_lines(self, norm: NormalizedInput, analysis: SymptomAnalysis) -> list[str]:
        f = analysis.facts
        lines = [f"- {self.lexicon.label(c)}" for c in analysis.symptoms]
        details = []
        if f.duration_text:
            details.append(f"хугацаа: {f.duration_text}")
        if f.severity is not None:
            details.append(f"хүч: {f.severity}/10")
        if f.temperature_c is not None:
            details.append(f"биеийн халуун: {f.temperature_c:g}°C")
        if f.age_months is not None and f.age_months < 24:
            details.append(f"нас: {f.age_months:g} сартай")
        elif f.age_years is not None:
            details.append(f"нас: {f.age_years:g}")
        if f.bp_systolic:
            details.append(f"даралт: {f.bp_systolic}/{f.bp_diastolic}")
        context = [self.lexicon.label(c) for c in analysis.concepts if self.lexicon.type_of(c) == "context"]
        if details:
            lines.append("- " + ", ".join(details))
        if context:
            lines.append("- нэмэлт нөхцөл: " + ", ".join(context))
        return lines

    def compose_template(self, *, norm: NormalizedInput, analysis: SymptomAnalysis, triage: TriageResult,
                         retrieval: RetrievalResult) -> DraftResponse:
        docs: list[RetrievedDoc] = retrieval.docs
        d1 = docs[0]
        d2 = docs[1] if len(docs) > 1 else None
        cite1, cite2 = f"[{d1.doc_id}]", (f"[{d2.doc_id}]" if d2 else "")
        out: list[str] = []

        # 🩺
        summary = self._summary_lines(norm, analysis) if analysis.symptoms else []
        if summary:
            out.append("🩺 **Таны хэлсэн шинж**\n" + "\n".join(summary))
        else:
            out.append(f"🩺 **Таны асуулт**\n- «{norm.display_text[:160]}»")

        # 🔎
        info = [_paragraph(d1.section("Ерөнхий мэдээлэл"), 520) + f" {cite1}"]
        if d2:
            info.append(_paragraph(d2.section("Ерөнхий мэдээлэл"), 260) + f" {cite2}")
        if analysis.intent == "symptom_check":
            info.append("_Таны хэлсэн шинжээр тодорхой онош тавих боломжгүй — доорх нь ерөнхий мэдээлэл юм._")
        out.append("🔎 **Ерөнхий мэдээлэл**\n" + "\n\n".join(i for i in info if i.strip()))

        # 💡
        care = [f"- {b}" for b in _bullets(d1.section("Гэртээ юу хийж болох вэ"), 6)]
        if d2:
            care += [f"- {b}" for b in _bullets(d2.section("Гэртээ юу хийж болох вэ"), 3) if f"- {b}" not in care]
        head = "💡 **Одоохондоо юу хийж болох вэ?**"
        if analysis.intent == "medication":
            head += ("\n💊 Эмийн нэр, тун, хэрэглэх хугацааны талаар би зөвлөгөө өгөх боломжгүй — "
                     "**эмчлэгч эмч эсвэл эм зүйчээсээ** заавал асуугаарай.")
        out.append(head + "\n" + "\n".join(care) + f" {cite1}{(' ' + cite2) if d2 else ''}")

        # ⚠️
        warn_text = d1.section("Анхаарах шинж")
        warn = [f"- {b}" for b in _bullets(warn_text, 6)]
        tail = _tail_sentence(warn_text) or "Эдгээр шинж илэрвэл даруй 103 дугаарт залгаарай."
        out.append("⚠️ **Анхаарах шинж**\n" + "\n".join(warn) + f"\n**{tail}** {cite1}")

        # 🏥
        lead = RISK_LEAD[triage.risk_level]
        reasons = [f.label_mn for f in triage.red_flags if f.risk != RiskLevel.LOW and f.rule_id != "LLM_SIGNAL"]
        when = [f"- {b}" for b in _bullets(d1.section("Эмчид хэзээ хандах вэ"), 5)]
        block = "🏥 **Хэзээ эмчид хандах вэ?**\n" + lead
        if reasons:
            block += "\n_Анхаарсан шалтгаан: " + "; ".join(reasons[:3]) + "_"
        if triage.modifiers:
            block += "\n_Эрсдэлтэй бүлэг: " + ", ".join(triage.modifiers) + "_"
        out.append(block + "\n" + "\n".join(when) + f" {cite1}")

        # 📚
        src = [f"- [{d.doc_id}] {d.title} — {d.source} ({d.source_url})" for d in docs]
        out.append("📚 **Эх сурвалж**\n" + "\n".join(src))
        out.append(DISCLAIMER)
        text = "\n\n".join(out)
        return DraftResponse(text=text, cited_doc_ids=sorted(cited_doc_ids(text)), generator="template")

    # ------------------------------------------------------------ LLM mode
    def _generate_llm(self, norm, analysis, triage, retrieval, feedback, attempt) -> DraftResponse:
        docs_xml = "\n".join(
            f'<document id="{d.doc_id}" title="{d.title}" source="{d.source}" tier="{d.source_tier}" url="{d.source_url}">\n'
            + "\n".join(f"## {s.title}\n{s.text}" for s in d.sections)
            + "\n</document>"
            for d in retrieval.docs
        )
        sa = {
            "intent": analysis.intent,
            "symptoms": [self.lexicon.label(c) for c in analysis.symptoms],
            "context": [self.lexicon.label(c) for c in analysis.concepts if self.lexicon.type_of(c) == "context"],
            "facts": analysis.facts.model_dump(exclude_none=True),
        }
        flags = "; ".join(f.label_mn for f in triage.red_flags) or "байхгүй"
        user = (
            f"<user_message>\n{norm.redacted}\n</user_message>\n"
            f"<transliterated>\n{norm.display_text}\n</transliterated>\n"
            f"<symptom_analysis>\n{json.dumps(sa, ensure_ascii=False)}\n</symptom_analysis>\n"
            f"<triage>\nrisk_level: {triage.risk_level.value}\nred_flags: {flags}\n"
            f"recommended_action: {triage.recommended_action}\n</triage>\n"
            f"<retrieved_documents>\n{docs_xml}\n</retrieved_documents>"
        )
        if feedback:
            user += "\n<safety_feedback>\n" + "\n".join(f"- {i.code}: {i.detail}" for i in feedback) + "\n</safety_feedback>"
        res = self.llm.generate(system=self.system_prompt, messages=[{"role": "user", "content": user}],
                                model=self.model, effort=self.effort)
        self.last_usage.append({"agent": "response", "model": res.model, "attempt": attempt, **res.usage})
        text = res.text
        if "ℹ️" not in text:
            text += "\n\n" + DISCLAIMER
        return DraftResponse(text=text, cited_doc_ids=sorted(cited_doc_ids(text)), generator="llm", attempt=attempt)
