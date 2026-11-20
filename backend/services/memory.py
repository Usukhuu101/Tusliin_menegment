"""
Conversation memory — what the user told the bot EARLIER IN THIS CHAT.

    "Би 28 настай, жирэмсэн"  ->  later  "Парацетамол ууж болох уу?"
    the answer says "🧠 Энэ ярианд өмнө хэлсэн: 28 настай, жирэмсэн" and tells the
    user to mention it to the pharmacist; the LLM (if on) sees it too.

Stored in SessionState.memory (SQLite). Only REDACTED text is kept — phone numbers,
national IDs and e-mails are masked by the input processor before anything is saved.
"Миний яриаг устгах" deletes it together with the session.

Used for:
  * personalised answers (template + LLM + the LLM safety judge sees it too)
  * "тэр эм" / "үүнийг" -> the medicine asked about before
  * "Миний нас хэд гэж хэлсэн бэ?"  -> RECALL (what I remember)
  * "Би 28 настай, жирэмсэн" (no question) -> NOTED (acknowledged, remembered)
Memory never lowers risk: triage already sees earlier concepts and facts on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models.schemas import Facts, HistoryItem, NormalizedInput, SessionState
from ..nlp.lexicon import ConceptLexicon

MEDICINE_PREFIXES = ("drug_", "eml_")
HISTORY_LIMIT = 10
LABEL = {"allergy_context": "харшилтай"}
TOPIC_TYPES = ("symptom", "red_flag", "condition")

# all patterns run on FOLDED text (ө->о, ү->у, й->и, е->э)
_QUESTION = re.compile(r"\?|\b(?:уу|юу|вэ|бэ)\b|гэж\s*ю|яаж|хэрхэн|ямар|хэзээ|хаана|\bхэд\b")
_PERSONAL = re.compile(r"\b(?:би|миний|надад|намаиг|манаи|маань|минь|оороо)\b|\w(?:таи|тэи|тои)\b")
_RECALL = re.compile(
    r"санаж\s+(?:баина|баигаа|бн|бна|бгаа|баи)|санадаг\s+(?:уу|юу)|юу\s+санаж"
    r"|миний\s+(?:нас|хуис|тухаи)|намаиг\s+(?:санаж|юу)"
    r"|(?:юу|хэд|ямар)\s+гэж\s+(?:хэлсэн|бичсэн)|омно\s+(?:нь\s+)?(?:юу|хэлсэн|бичсэн)\s+(?:юу|зуил|бэ|вэ|мэдээлэл)"
)
# "тэр эм", "энэ эмийн", "уг эмийг", "үүнийг", "түүнийг"
_REFERS_BACK = re.compile(r"\b(?:тэр|энэ|уг|тус)\s+эм|\b(?:уун|туун)ии[гн]?\b|\b(?:уун|туун)ээс\b")
_ALLERGY = re.compile(r"([А-Яа-яЁёӨөҮүA-Za-z-]{3,})\s+(харшилтай|харшилтэй|харшил өгдөг|харшилтай байдаг)", re.I)


def _add(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _to_end(items: list[str], value: str) -> None:
    if value in items:
        items.remove(value)
    items.append(value)


def _profile_facts(session: SessionState) -> Facts:
    return session.background_facts.merged_with(session.facts)


# ------------------------------------------------------------------ write
def remember(session: SessionState, norm: NormalizedInput, lexicon: ConceptLexicon, reply_type: str) -> None:
    """Called once per turn (orchestrator._finish)."""
    mem = session.memory
    personal = bool(_PERSONAL.search(norm.folded_text)) and not _QUESTION.search(norm.folded_text)
    for c in norm.positive_concepts:
        t = lexicon.type_of(c)
        if t == "context":
            _add(mem.contexts, c)
        elif t == "condition" and personal:          # "Би чихрийн шижинтэй", not "Чихрийн шижин гэж юу вэ"
            _add(mem.conditions, c)
        elif t in ("symptom", "red_flag"):
            _add(mem.symptoms, c)
        elif c.startswith(MEDICINE_PREFIXES):
            _to_end(mem.medicines, c)
    for c in norm.negated_concepts:                   # "жирэмсэн биш", "харшилгүй"
        for bucket in (mem.contexts, mem.conditions):
            if c in bucket:
                bucket.remove(c)
    for m in _ALLERGY.finditer(norm.display_text):
        _add(mem.allergies, m.group(0).lower()[:60])
    topics = [lexicon.label(c) for c in norm.positive_concepts
              if lexicon.type_of(c) in TOPIC_TYPES or c.startswith(MEDICINE_PREFIXES) or c == "side_effect"]
    mem.history.append(HistoryItem(user=norm.display_text[:300], reply_type=reply_type, topics=topics[:4]))
    mem.history = mem.history[-HISTORY_LIMIT:]


# ------------------------------------------------------------------ read
def profile_items(session: SessionState, lexicon: ConceptLexicon, exclude: set[str] = frozenset(),
                  skip_age: bool = False) -> list[str]:
    """Short phrases about the PERSON: "28 настай", "жирэмсэн", "чихрийн шижин", «пенициллинд харшилтай»."""
    mem, f = session.memory, _profile_facts(session)
    items = []
    if not skip_age:
        if f.age_months is not None and f.age_months < 24:
            items.append(f"{f.age_months:g} сартай")
        elif f.age_years is not None:
            items.append(f"{f.age_years:g} настай")
    if f.sex:
        items.append("эмэгтэй" if f.sex == "female" else "эрэгтэй")
    for c in mem.contexts:
        if (c in exclude or (c == "allergy_context" and mem.allergies)
                or (c == "child" and "infant" in mem.contexts)):        # "нярай" already says it
            continue
        items.append(LABEL.get(c, lexicon.label(c)))
    items += [lexicon.label(c) for c in mem.conditions if c not in exclude]
    items += [f"«{a}»" for a in mem.allergies]
    return items


def caution_items(session: SessionState, lexicon: ConceptLexicon, exclude: set[str] = frozenset()) -> list[str]:
    """Remembered facts that matter when a MEDICINE is discussed (pregnancy, child, old age,
    chronic disease, allergy). Plain adult age alone is not a caution."""
    f = _profile_facts(session)
    age = []
    if f.age_months is not None and f.age_months < 24:
        age = [f"{f.age_months:g} сартай"]
    elif f.age_years is not None and (f.age_years < 18 or f.age_years >= 65):
        age = [f"{f.age_years:g} настай"]
    return age + profile_items(session, lexicon, exclude, skip_age=True)


def referenced_medicine(norm: NormalizedInput, session: SessionState) -> str | None:
    """'Тэр эмийн гаж нөлөө' -> the medicine asked about before (if this message names none)."""
    if not _REFERS_BACK.search(norm.folded_text) or not session.memory.medicines:
        return None
    if any(c.startswith(MEDICINE_PREFIXES) for c in norm.positive_concepts):
        return None
    return session.memory.medicines[-1]


def is_recall_question(norm: NormalizedInput) -> bool:
    return bool(_RECALL.search(norm.folded_text))


def is_self_description(norm: NormalizedInput, lexicon: ConceptLexicon) -> bool:
    """'Би 28 настай, жирэмсэн' — facts about the person, no question, nothing to look up."""
    if _QUESTION.search(norm.folded_text):
        return False
    f = norm.facts
    has_profile = (f.age_years is not None or f.age_months is not None or f.sex is not None
                   or any(lexicon.type_of(c) == "context" for c in norm.positive_concepts)
                   or bool(_ALLERGY.search(norm.display_text)))
    return has_profile


@dataclass
class MemoryContext:
    """Memory as the response / safety agents see it for ONE turn."""
    profile: list[str] = field(default_factory=list)    # earlier facts NOT repeated in this message
    cautions: list[str] = field(default_factory=list)   # subset relevant to medicines
    medicines: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.profile or self.medicines or self.symptoms or self.history)

    def prompt_block(self) -> str:
        if not self:
            return ""
        lines = ["Энэ ярианд хэрэглэгч ӨМНӨ хэлсэн (хариултаа тохируулахад ашигла; энэ нь эмнэлгийн баримт биш):"]
        if self.profile:
            lines.append("- хүний тухай: " + ", ".join(self.profile))
        if self.symptoms:
            lines.append("- өмнө хэлсэн шинж: " + ", ".join(self.symptoms))
        if self.medicines:
            lines.append("- өмнө асуусан эм (сүүлийнх нь хамгийн сүүлд): " + ", ".join(self.medicines))
        if self.history:
            lines.append("Сүүлийн яриа (эхнийх нь хамгийн хуучин):")
            lines += [f"{i}. {h}" for i, h in enumerate(self.history, 1)]
        return "\n".join(lines)


def build_context(session: SessionState, norm: NormalizedInput, lexicon: ConceptLexicon,
                  history_turns: int = 6) -> MemoryContext:
    """Built BEFORE this turn is remembered, so `profile` is what was said earlier."""
    here = set(norm.positive_concepts)
    skip_age = norm.facts.age_years is not None or norm.facts.age_months is not None
    mem = session.memory
    return MemoryContext(
        profile=profile_items(session, lexicon, here, skip_age=skip_age),
        cautions=caution_items(session, lexicon, here),
        medicines=[lexicon.label(c) for c in mem.medicines if c not in here],
        symptoms=[lexicon.label(c) for c in mem.symptoms if c not in here],
        history=[f"«{h.user[:160]}» → {h.reply_type}" + (f" ({', '.join(h.topics)})" if h.topics else "")
                 for h in mem.history[-history_turns:]],
    )


# ------------------------------------------------------------------ replies
def recall_text(session: SessionState, lexicon: ConceptLexicon) -> str:
    mem = session.memory
    profile = profile_items(session, lexicon)
    symptoms = [lexicon.label(c) for c in mem.symptoms]
    medicines = [lexicon.label(c) for c in mem.medicines]
    topics = list(dict.fromkeys(t for h in mem.history for t in h.topics if t not in symptoms + medicines))
    if not (profile or symptoms or medicines or topics):
        return ("🧠 Энэ ярианд та өөрийнхөө талаар (нас, жирэмслэлт, архаг өвчин, харшил гэх мэт) одоогоор юу ч "
                "хэлээгүй байна. Хэлбэл энэ ярианы туршид санаж, хариултаа түүнд тохируулна.")
    out = ["🧠 **Энэ ярианд таны хэлсэн мэдээлэл:**"]
    if profile:
        out.append("- Хэлсэн нөхцөл: " + ", ".join(profile))
    if symptoms:
        out.append("- Хэлсэн шинж тэмдэг: " + ", ".join(symptoms))
    if medicines:
        out.append("- Асуусан эм: " + ", ".join(medicines))
    if topics:
        out.append("- Ярилцсан сэдэв: " + ", ".join(topics[:8]))
    out.append("\nДараагийн хариултууддаа эдгээрийг харгалзана. Устгахыг хүсвэл доорх «Миний яриаг устгах»-ыг дарна уу.")
    return "\n".join(out)


def noted_text(session: SessionState, norm: NormalizedInput, lexicon: ConceptLexicon) -> str:
    """Reply to a self-description. Uses the memory AFTER this message is remembered."""
    items = profile_items(session, lexicon)
    got = ", ".join(items) if items else "таны хэлсэн мэдээлэл"
    return (f"📝 Ойлголоо, энэ ярианы туршид санаж авлаа: **{got}**.\n\n"
            "Одоо асуух зүйлээ бичнэ үү — жишээ нь шинж тэмдэг, өвчин эсвэл эмийн талаар. "
            "Хариултаа үүнд тохируулна.\n\n"
            "ℹ️ Яаралтай шинж (амьсгал давчдах, цээжээр хүчтэй өвдөх, ухаан балартах) илэрвэл **103** дугаарт залгаарай.")
