"""
MULTI-AGENT ORCHESTRATOR — deterministic Python code decides the route,
never an LLM. One call to `handle()` = one conversation turn.

  input -> Input Processor
        -> Agent 1 Triage (rules pass)        --EMERGENCY--> emergency template
        -> Agent 2 Symptom Analysis
        -> Agent 1 Triage (merge pass)        --EMERGENCY--> emergency template
        -> follow-up questions? (risk <= MODERATE, missing info, max 2 rounds)
        -> Agent 3 RAG                        --insufficient--> safe fallback + gap log
        -> Agent 4 Response
        -> Agent 5 Safety  --REVISE--> regenerate (max N) --still unsafe--> template / fallback
        -> store redacted turn + trace, return

Every agent's work is recorded as an AgentStep (shown in the UI under
"🧠 Агентуудын ажиллагаа") so the team can see — and demo — what each agent did.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Optional

from ..agents.rag_agent import RAGAgent
from ..agents.response_agent import DISCLAIMER, RISK_LEAD, ResponseAgent
from ..agents.safety_agent import SafetyAgent
from ..agents.symptom_agent import SYMPTOM_TYPES, SymptomAgent, mentions_health_topic
from ..agents.triage_agent import TriageAgent
from ..config import Settings
from ..database.db import Database
from ..models.schemas import (AgentStep, ChatResponse, DraftResponse, Facts, NormalizedInput, RetrievalResult, RiskLevel,
                              SafetyVerdict, SessionState, SourceOut, SymptomAnalysis, TriageResult, TurnTrace,
                              max_risk, risk_rank)
from ..nlp.input_processor import InputProcessor
from ..nlp.lexicon import ConceptLexicon
from ..rag.embeddings import get_embedding_provider
from ..rag.indexer import ensure_index, lexicon_paths
from .llm import get_llm_provider

log = logging.getLogger(__name__)
APP_VERSION = "0.1.0"
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
TOPIC_TYPES = {"symptom", "red_flag", "condition", "medication"}   # what a user asks ABOUT
CARRY_OVER_TYPES = {"red_flag", "context", "exposure"}             # kept when the topic changes

TITLES = {
    "input": "Оролт боловсруулагч",
    "triage": "Агент 1 · Триаж (дүрэм)",
    "symptom": "Агент 2 · Шинж тэмдгийн шинжилгээ",
    "triage_merge": "Агент 1 · Триаж (нэгтгэсэн)",
    "rag": "Агент 3 · Мэдлэгийн сан (RAG)",
    "response": "Агент 4 · Хариулт бичих",
    "safety": "Агент 5 · Аюулгүй байдал",
}
INTENT_MN = {"symptom_check": "шинж тэмдэг", "medication": "эмийн асуулт", "general_info": "ерөнхий мэдээлэл",
             "greeting": "мэндчилгээ", "thanks": "талархал", "out_of_scope": "хамрах хүрээнээс гадуур",
             "unclear": "тодорхойгүй"}

GREETING = (
    "Сайн байна уу! 👋 Би **эрүүл мэндийн ерөнхий мэдээлэл** өгөх туслах чатбот байна.\n\n"
    "Танд ямар шинж тэмдэг илэрч байгаа, эсвэл ямар өвчний талаар мэдэхийг хүсч байгаагаа бичээрэй. Жишээ нь:\n"
    "- «Өчигдрөөс толгой өвдөөд байна»\n- «Хүүхэд маань 38.5 халуураад байна»\n- «Даралт ихсэлт гэж юу вэ?»\n\n"
    "ℹ️ Би эмч биш тул онош тавих, эм бичих боломжгүй. **Яаралтай үед 103 дугаарт залгаарай.**"
)
THANKS = ("Баярлалаа! 🙏 Бие тань удахгүй сайжрахыг хүсье. Шинж тэмдэг хүндэрвэл эмчид хандаарай, "
          "яаралтай үед **103** дугаарт залгаарай.")
OUT_OF_SCOPE = ("Уучлаарай, би зөвхөн **эрүүл мэндийн ерөнхий мэдээлэл** өгөх боломжтой. "
                "Шинж тэмдэг, өвчин, анхны тусламж, эрүүл мэндийн талаар асуугаарай. Яаралтай үед **103** дугаарт залгаарай.")
CLARIFY = ("Таны асуултыг бүрэн ойлгож чадсангүй 🙏 Ямар шинж тэмдэг байгаа (**хаана, хэзээнээс, хэр хүчтэй**) эсвэл "
           "ямар өвчний талаар мэдэхийг хүсч байгаагаа тодорхой бичнэ үү.\n\n"
           "Жишээ нь: «Өчигдрөөс толгой өвдөөд байна», «Улаанбурхан гэж юу вэ?»\n\n"
           "Амьсгал давчдах, цээжээр өвдөх зэрэг яаралтай шинж илэрвэл **103** дугаарт залгаарай.")
FALLBACK_LEAD = {
    RiskLevel.LOW: "Өрхийн эмч эсвэл эм зүйчдээ хандаж зөвлөгөө аваарай.",
    RiskLevel.MODERATE: RISK_LEAD[RiskLevel.MODERATE],
    RiskLevel.HIGH: RISK_LEAD[RiskLevel.HIGH],
}


class ChatOrchestrator:
    def __init__(self, s: Settings, db: Optional[Database] = None):
        self.s = s
        self.lexicon = ConceptLexicon(lexicon_paths(s))
        self.processor = InputProcessor(self.lexicon)
        self.llm = get_llm_provider(s.llm_provider)
        self.embedder = get_embedding_provider(s.embedding_provider, s.embedding_model)
        self.index, self.index_info = ensure_index(s, self.embedder, self.lexicon)
        self.triage = TriageAgent(s.rules_dir / "red_flags.yaml", s.rules_dir / "emergency_templates.yaml")
        self.symptom = SymptomAgent(self.lexicon, s.protocols_dir / "symptom_protocols.yaml", self.llm,
                                    s.extraction_model, s.prompts_dir / "symptom_extraction.md", s.max_followup_questions)
        self.rag = RAGAgent(self.index, self.lexicon, s.cosine_threshold)
        self.response = ResponseAgent(self.lexicon, self.llm, s.response_model, s.response_effort,
                                      s.prompts_dir / "system_response.md")
        self.safety = SafetyAgent(self.llm, s.safety_model, s.prompts_dir / "safety_judge.md")
        self.db = db or Database(s.db_path)
        purged = self.db.purge_older_than(s.retention_days)
        if purged:
            log.info("Retention: deleted %d old turns", purged)
        self.versions = {
            "app": APP_VERSION, "lexicon": self.lexicon.version, "rules": self.triage.version,
            "kb_index": self.index.name, "prompts": "1.0", "mode": self.mode,
        }

    @property
    def mode(self) -> str:
        return f"llm:{self.llm.name}" if self.llm.available else "offline"

    # ------------------------------------------------------------ step log
    @staticmethod
    def _step(trace: TurnTrace, agent: str, detail: str, t_start: float, status: str = "ok") -> None:
        trace.steps.append(AgentStep(agent=agent, title=TITLES[agent], status=status, detail=detail,
                                     ms=round((time.perf_counter() - t_start) * 1000, 1)))

    @staticmethod
    def _skip(trace: TurnTrace, agents: list[str], reason: str) -> None:
        for a in agents:
            trace.steps.append(AgentStep(agent=a, title=TITLES[a], status="skipped", detail=reason))

    def _labels(self, concepts: list[str]) -> str:
        return ", ".join(self.lexicon.label(c) for c in concepts) or "—"

    def _start_new_topic(self, session: SessionState) -> None:
        """The previous question was answered and the user now raises a NEW topic.
        Forget the old complaint (its symptoms, duration, severity ...) so that
        "гэдэс өвдөөд байна" after a headache answer is not read as "headache +
        stomach ache". Who the patient is (age, child, pregnancy), what they were
        exposed to and any red flag stay in focus.
        SAFETY: nothing is lost for triage — the old topic moves to `background_*`,
        which Agent 1 still evaluates, so cross-turn combinations (chest pain now,
        breathlessness later) still trigger the emergency rules exactly as before."""
        keep = [c for c in session.concepts if self.lexicon.type_of(c) in CARRY_OVER_TYPES]
        dropped = [c for c in session.concepts if c not in keep]
        session.background_concepts = list(dict.fromkeys(session.background_concepts + dropped))
        session.background_facts = session.background_facts.merged_with(session.facts)
        session.concepts = keep
        session.negated_concepts = []
        old = session.facts
        session.facts = Facts(age_years=old.age_years, age_months=old.age_months, sex=old.sex)
        session.followup_rounds = 0          # a new complaint starts a new clarification round
        session.asked_slots = []

    # ================================================================ turn
    def handle(self, message: str, session_id: Optional[str] = None) -> ChatResponse:
        t0 = time.perf_counter()
        sid = session_id if session_id and _SESSION_RE.match(session_id) else uuid.uuid4().hex
        session = self.db.get_session(sid) or SessionState(session_id=sid)
        trace = TurnTrace(input={}, versions=self.versions)
        usage: list[dict] = []

        # ---- Input processor
        t = time.perf_counter()
        norm = self.processor.process(message)
        trace.input = {
            "redacted": norm.redacted, "display": norm.display_text, "script": norm.script, "pii": norm.pii_types,
            "concepts": [c.model_dump() for c in norm.concepts], "facts": norm.facts.model_dump(exclude_none=True),
        }
        parts = [f"бичиг: {norm.script}"]
        if norm.script in ("latin", "mixed"):
            parts.append(f"кирилл: «{norm.display_text[:60]}»")
        parts.append(f"ойлголт: {self._labels(norm.positive_concepts)}")
        if norm.negated_concepts:
            parts.append(f"үгүйсгэсэн: {self._labels(norm.negated_concepts)}")
        if norm.pii_types:
            parts.append(f"нуусан хувийн мэдээлэл: {', '.join(norm.pii_types)}")
        self._step(trace, "input", "; ".join(parts), t)

        current_symptoms = [c for c in norm.positive_concepts if self.lexicon.type_of(c) in SYMPTOM_TYPES]
        current_topics = [c for c in norm.positive_concepts if self.lexicon.type_of(c) in TOPIC_TYPES]
        if any(c not in session.concepts for c in current_topics) and session.last_message_type in ("ANSWER", "FALLBACK"):
            self._start_new_topic(session)

        # ---- Agent 1, pass 1 (rules only, current message)
        t = time.perf_counter()
        tri1 = self.triage.assess(norm.positive_concepts, session.triage_facts(session.facts.merged_with(norm.facts)),
                                  bool(current_symptoms))
        trace.triage_rules = tri1.model_dump()
        self._step(trace, "triage", self._triage_detail(tri1), t,
                   "alert" if tri1.risk_level == RiskLevel.EMERGENCY else "ok")
        if tri1.risk_level == RiskLevel.EMERGENCY:
            self._skip(trace, ["symptom", "triage_merge", "rag"], "Яаралтай — шууд 103 загвар руу шилжив")
            return self._emergency(tri1, norm, session, trace, None, t0, usage)

        # ---- Agent 2
        t = time.perf_counter()
        analysis = self.symptom.analyze(norm, session)
        usage += self.symptom.last_usage
        trace.symptoms = analysis.model_dump()
        self._step(trace, "symptom", self._symptom_detail(analysis), t)

        # ---- Agent 1, pass 2 (session-merged + LLM signal; LLM can only raise)
        t = time.perf_counter()
        tri_concepts = session.triage_concepts(analysis.concepts + analysis.llm_red_flags)
        tri = self.triage.assess(
            tri_concepts, session.triage_facts(analysis.facts),
            any(self.lexicon.type_of(c) in SYMPTOM_TYPES for c in tri_concepts),
            llm_risk=analysis.llm_risk, llm_flags=[self.lexicon.label(c) for c in analysis.llm_red_flags],
        )
        trace.triage_final = tri.model_dump()
        self._step(trace, "triage_merge", self._triage_detail(tri), t,
                   "alert" if risk_rank(tri.risk_level) >= risk_rank(RiskLevel.HIGH) else "ok")
        if tri.risk_level == RiskLevel.EMERGENCY:
            self._skip(trace, ["rag"], "Яаралтай — шууд 103 загвар руу шилжив")
            return self._emergency(tri, norm, session, trace, analysis, t0, usage)

        # ---- simple intents
        if analysis.intent in ("greeting", "thanks", "out_of_scope"):
            text = {"greeting": GREETING, "thanks": THANKS, "out_of_scope": OUT_OF_SCOPE}[analysis.intent]
            mtype = {"greeting": "GREETING", "thanks": "THANKS", "out_of_scope": "OUT_OF_SCOPE"}[analysis.intent]
            self._skip(trace, ["rag", "response", "safety"], f"«{INTENT_MN[analysis.intent]}» — бэлэн хариу")
            return self._finish(text, mtype, tri, norm, session, analysis, trace, t0, usage)

        # ---- follow-up questions (Agent 2 asks; never when risk is HIGH)
        if (analysis.intent == "symptom_check" and analysis.followup_questions
                and risk_rank(tri.risk_level) <= risk_rank(RiskLevel.MODERATE)
                and session.followup_rounds < self.s.max_followup_rounds):
            labels = ", ".join(self.lexicon.label(c) for c in analysis.symptoms) or "таны шинж"
            qs = "\n".join(f"{i}. {q}" for i, q in enumerate(analysis.followup_questions, 1))
            text = (f"🩺 Ойлголоо — **{labels}**. Танд илүү тохирох мэдээлэл өгөхийн тулд хэдэн зүйл тодруулъя:\n\n{qs}\n\n"
                    "ℹ️ Хэрэв амьсгал давчдах, цээжээр хүчтэй өвдөх, ухаан балартах зэрэг шинж илэрвэл "
                    "хүлээлгүйгээр **103** дугаарт залгаарай.")
            session.pending_slots = analysis.missing_slots
            session.asked_slots = list(dict.fromkeys(session.asked_slots + analysis.missing_slots))
            session.followup_rounds += 1
            self._skip(trace, ["rag", "response", "safety"],
                       f"Агент 2 эхлээд {len(analysis.followup_questions)} тодруулах асуулт асууж байна")
            return self._finish(text, "FOLLOWUP", tri, norm, session, analysis, trace, t0, usage,
                                followups=analysis.followup_questions)
        session.pending_slots = []

        # ---- Agent 3
        t = time.perf_counter()
        retrieval = self.rag.retrieve(analysis, norm)
        trace.retrieval = {
            "query": retrieval.query, "concepts": retrieval.query_concepts, "sufficient": retrieval.sufficient,
            "reason": retrieval.reason, "index": retrieval.index_name,
            "docs": [{"id": d.doc_id, "score": d.score, "overlap": d.concept_overlap, "cos": d.best_cosine}
                     for d in retrieval.docs],
        }
        found = ", ".join(f"{d.doc_id} «{d.title}» ({d.score:.2f})" for d in retrieval.docs) or "олдсонгүй"
        self._step(trace, "rag", f"{found}; {'хангалттай нотолгоо' if retrieval.sufficient else 'хангалтгүй — ' + retrieval.reason}",
                   t, "ok" if retrieval.sufficient else "fallback")
        if not retrieval.sufficient:
            self.db.add_knowledge_gap(session_id=sid, query=norm.redacted, concepts=retrieval.query_concepts,
                                      top_score=retrieval.top_score, reason=retrieval.reason)
            self._skip(trace, ["response"], "Найдвартай мэдээлэл алга — зохиохгүй, аюулгүй хариу + цоорхойн бүртгэл")
            # "Шүд өвдөөд байна" is perfectly clear — we just have no document for it.
            # Only a genuinely vague message gets "please describe it more".
            if analysis.intent == "unclear" and not analysis.concepts and not mentions_health_topic(norm):
                return self._finish(CLARIFY, "CLARIFY", tri, norm, session, analysis, trace, t0, usage)
            return self._finish(self._fallback_text(tri), "FALLBACK", tri, norm, session, analysis, trace, t0, usage)

        # ---- Agents 4 + 5
        draft = self._answer_with_safety(norm, analysis, tri, retrieval, trace, usage)
        if draft is None:
            return self._finish(self._fallback_text(tri), "FALLBACK", tri, norm, session, analysis, trace, t0, usage)
        return self._finish(draft.text, "ANSWER", tri, norm, session, analysis, trace, t0, usage, retrieval=retrieval)

    # ============================================================ details
    def _triage_detail(self, tri: TriageResult) -> str:
        flags = "; ".join(f.label_mn for f in tri.red_flags[:3]) or "улаан туг илрээгүй"
        extra = f"; эрсдэлтэй бүлэг: {', '.join(tri.modifiers)}" if tri.modifiers else ""
        src = " (LLM дохио нэмэгдсэн)" if tri.source == "rules+llm" else ""
        return f"эрсдэл: {tri.risk_level.value}{src} — {flags}{extra}"

    def _symptom_detail(self, a: SymptomAnalysis) -> str:
        f = a.facts
        facts = [x for x in (
            f"хугацаа {f.duration_text}" if f.duration_text else "",
            f"хүч {f.severity}/10" if f.severity is not None else "",
            f"халуун {f.temperature_c:g}°C" if f.temperature_c is not None else "",
            f"нас {f.age_years:g}" if f.age_years is not None else "",
            f"даралт {f.bp_systolic}/{f.bp_diastolic}" if f.bp_systolic else "",
        ) if x]
        parts = [f"зорилго: {INTENT_MN.get(a.intent, a.intent)}", f"шинж: {self._labels(a.symptoms)}"]
        if facts:
            parts.append(", ".join(facts))
        if a.missing_slots:
            parts.append(f"дутуу мэдээлэл: {', '.join(s.split(':')[0] for s in a.missing_slots)}")
        parts.append("арга: дүрэм + LLM" if a.source == "rules+llm" else "арга: дүрэм (офлайн)")
        return "; ".join(parts)

    @staticmethod
    def _verdict_detail(v: SafetyVerdict) -> str:
        who = " + ".join("дүрэм" if c == "rules" else "LLM шүүгч" for c in v.checked_by)
        issues = "; ".join(f"{i.code}" for i in v.issues[:4])
        return f"{v.verdict} ({who})" + (f" — {issues}" if issues else "")

    # ================================================================ helpers
    def _answer_with_safety(self, norm: NormalizedInput, analysis: SymptomAnalysis, tri: TriageResult,
                            retrieval: RetrievalResult, trace: TurnTrace, usage: list[dict]) -> Optional[DraftResponse]:
        t = time.perf_counter()
        draft = self.response.generate(norm=norm, analysis=analysis, triage=tri, retrieval=retrieval)
        usage += self.response.last_usage
        self._step(trace, "response", self._draft_detail(draft), t)
        for attempt in range(self.s.max_regenerations + 1):
            t = time.perf_counter()
            verdict = self.safety.review(draft, triage=tri, retrieval=retrieval, norm=norm)
            usage += self.safety.last_usage
            trace.drafts.append({"generator": draft.generator, "attempt": draft.attempt, "chars": len(draft.text),
                                 "cited": draft.cited_doc_ids})
            trace.safety.append(verdict.model_dump())
            self._step(trace, "safety", self._verdict_detail(verdict), t,
                       "ok" if verdict.verdict == "SAFE" else "alert")
            if verdict.verdict == "SAFE":
                return draft
            judge_down = any(i.code == "JUDGE_UNAVAILABLE" for i in verdict.issues)
            if draft.generator == "llm" and verdict.verdict == "REVISE" and not judge_down \
                    and attempt < self.s.max_regenerations:
                t = time.perf_counter()
                draft = self.response.generate(norm=norm, analysis=analysis, triage=tri, retrieval=retrieval,
                                               feedback=verdict.issues, attempt=attempt + 2)
                usage += self.response.last_usage
                self._step(trace, "response", "Дахин бичив (аюулгүй байдлын санал) — " + self._draft_detail(draft), t)
                continue
            break
        if draft.generator == "llm":   # never release an unsafe LLM draft -> deterministic answer
            t = time.perf_counter()
            draft = self.response.compose_template(norm=norm, analysis=analysis, triage=tri, retrieval=retrieval)
            verdict = self.safety.review(draft, triage=tri, retrieval=retrieval, norm=norm)
            trace.drafts.append({"generator": "template", "attempt": "fallback", "chars": len(draft.text)})
            trace.safety.append(verdict.model_dump())
            self._step(trace, "safety", "LLM ноорог татгалзсан → хянагдсан KB загвар: " + self._verdict_detail(verdict),
                       t, "fallback")
            if verdict.verdict == "SAFE":
                return draft
        return None

    def _draft_detail(self, d: DraftResponse) -> str:
        how = (f"LLM ({self.llm.model_for(self.s.response_model)})" if d.generator == "llm"
               else "хянагдсан KB текстээс угсарсан (офлайн)")
        return f"{how}; иш татсан: {', '.join(d.cited_doc_ids) or '—'}; {len(d.text)} тэмдэгт"

    def _fallback_text(self, tri: TriageResult) -> str:
        lead = FALLBACK_LEAD.get(tri.risk_level, FALLBACK_LEAD[RiskLevel.HIGH])
        return (
            "Уучлаарай, энэ асуултад хариулах **найдвартай, баталгаажсан мэдээлэл** миний мэдлэгийн санд одоогоор "
            "хангалтгүй байна. Буруу мэдээлэл өгөхгүйн тулд таамаглахгүй байя.\n\n"
            f"🏥 {lead}\n\n"
            "⚠️ Амьсгал давчдах, цээжээр өвдөх, ухаан балартах, их цус алдах зэрэг яаралтай шинж илэрвэл "
            "**103** дугаарт залгаарай.\n\n"
            "📝 Таны асуултыг (хувийн мэдээлэлгүйгээр) мэдлэгийн санг өргөжүүлэхэд ашиглахаар тэмдэглэлээ.\n\n"
            + DISCLAIMER
        )

    def _emergency(self, tri: TriageResult, norm: NormalizedInput, session: SessionState, trace: TurnTrace,
                   analysis: Optional[SymptomAnalysis], t0: float, usage: list[dict]) -> ChatResponse:
        t = time.perf_counter()
        text = self.triage.render_emergency(tri)
        self._step(trace, "response", f"Урьдчилан батлагдсан яаралтай загвар «{tri.template}» (LLM-гүй)", t, "alert")
        t = time.perf_counter()
        verdict = self.safety.review(DraftResponse(text=text, generator="template"), triage=tri, retrieval=None,
                                     norm=norm, require_sections=False)
        trace.safety.append(verdict.model_dump())
        self._step(trace, "safety", self._verdict_detail(verdict) + "; 103 заавар байгаа эсэхийг шалгав", t)
        if verdict.verdict == "BLOCK":   # should never happen — templates are pre-approved
            text = "🚨 **Яг одоо 103 дугаарт залгаж яаралтай тусламж авна уу.**"
        return self._finish(text, "EMERGENCY", tri, norm, session, analysis, trace, t0, usage)

    def _finish(self, text: str, mtype: str, tri: TriageResult, norm: NormalizedInput, session: SessionState,
                analysis: Optional[SymptomAnalysis], trace: TurnTrace, t0: float, usage: list[dict],
                retrieval: Optional[RetrievalResult] = None, followups: Optional[list[str]] = None) -> ChatResponse:
        # update session memory
        if analysis is not None:
            session.concepts, session.negated_concepts = analysis.concepts, analysis.negated_concepts
            session.facts = analysis.facts
        else:
            session.concepts = list(dict.fromkeys(session.concepts + norm.positive_concepts))
            session.facts = session.facts.merged_with(norm.facts)
        if mtype != "FOLLOWUP":
            session.pending_slots = []
        session.turn_count += 1
        session.max_risk = max_risk(session.max_risk, tri.risk_level)
        session.last_message_type = mtype

        latency = int((time.perf_counter() - t0) * 1000)
        trace.message_type = mtype
        trace.llm_usage = usage
        trace.timings_ms = {"total": latency}
        self.db.save_session(session)
        turn_id = self.db.add_turn(
            session_id=session.session_id, turn_no=session.turn_count, user_text=norm.redacted, reply=text,
            message_type=mtype, risk_level=tri.risk_level.value, mode=self.mode, latency_ms=latency,
            trace=trace.model_dump(),
        )
        sources = []
        notice = None
        if retrieval and mtype == "ANSWER":
            sources = [SourceOut(doc_id=d.doc_id, title=d.title, source=d.source, url=d.source_url,
                                 tier=d.source_tier, status=d.status, score=d.score) for d in retrieval.docs]
            if any(d.status != "ACTIVE" for d in retrieval.docs):
                notice = ("Энэ хариултад ашигласан мэдлэгийн сангийн мэдээлэл эмчийн хяналтаар хараахан "
                          "батлагдаагүй (REVIEW төлөв) — зөвхөн туршилтын зорилгоор.")
        return ChatResponse(
            session_id=session.session_id, turn_id=turn_id, reply=text, message_type=mtype,
            risk_level=tri.risk_level, red_flags=[f.label_mn for f in tri.red_flags if f.risk != RiskLevel.LOW],
            sources=sources, followup_questions=followups or [], mode=self.mode, content_notice=notice,
            latency_ms=latency, agents=trace.steps,
        )
