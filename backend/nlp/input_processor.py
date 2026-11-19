"""
INPUT PROCESSOR (no LLM).

1. Unicode clean-up
2. Remove personal information (phone, national ID/РД, e-mail, names)
3. Latin chat-script -> Cyrillic (display) and folded form (matching)
4. Concept matching with the lexicon (incl. negation)
5. Deterministic fact extraction: duration, pain score, temperature, age,
   blood pressure, sex.
"""
from __future__ import annotations

import re

from ..models.schemas import ConceptHit, Facts, NormalizedInput
from .lexicon import ConceptLexicon
from .text_utils import NUMBER_WORDS, fold, normalize_unicode, script_of, to_display_cyrillic

# ------------------------------------------------------------------ PII
_PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[ИМЭЙЛ]"),
    ("national_id", re.compile(r"\b[А-ЯӨҮЁа-яөүё]{2}\s?\d{8}\b"), "[РД]"),
    ("phone", re.compile(r"(?<![\d/.,])(?:\+?976[\s-]?)?[5-9]\d{3}[\s-]?\d{4}(?![\d/])"), "[УТАС]"),
    ("name", re.compile(r"(миний нэр(?:\s+бол)?|намайг)\s+([А-ЯӨҮЁ][а-яөүё]+)", re.IGNORECASE), r"\1 [НЭР]"),
]

# ---------------------------------------------------------------- facts
_NUM = r"(\d+(?:[.,]\d+)?|нэг|хоэр|гурван|гурав|дорвон|доров|таван|тав|зургаан|зургаа|долоон|долоо|наиман|наим|эсон|эс|арван|арав)"
_UNIT = r"(долоо\s*хоног|хоног(?!то)|одор|удур|цаг|минут|сар(?!та)|жил)"
_DURATION_RE = re.compile(r"\b" + _NUM + r"\s*" + _UNIT)
_UNIT_DAYS = {"хоног": 1, "одор": 1, "удур": 1, "цаг": 1 / 24, "минут": 1 / 1440, "сар": 30, "жил": 365}
_RELATIVE = [
    (re.compile(r"уржигдар"), 2.0, "уржигдраас"),
    (re.compile(r"очигд(о|р)"), 1.0, "өчигдрөөс"),
    (re.compile(r"оноо\s*(оглоо|шоно)|оноодор|оглоо"), 0.5, "өнөөдрөөс"),
    (re.compile(r"\bсаяхан\b|\bсая\b|\bдоноос\b"), 0.1, "саяхнаас"),
    (re.compile(r"since yesterday|yesterday"), 1.0, "өчигдрөөс"),
]
_EN_DURATION = re.compile(r"(\d+)\s*(day|days|week|weeks|hour|hours)")

_SEVERITY_NUM = [
    re.compile(r"\b(10|[1-9])\s*(?:/\s*10|оноо|балл|point)"),
    re.compile(r"10\s*-?(?:аас|оос)\s*(10|[1-9])\b"),
]
_SEVERITY_WORDS = [
    (re.compile(r"тэсэхии?н аргагуи|тэвчихии?н аргагуи|маш хучтэи|хэт хучтэи|аимаар|аимшигтаи"), 8),
    (re.compile(r"\bхучтэи\b|\bих овд|\bихэд\b"), 7),
    (re.compile(r"дунд зэрэг"), 5),
    (re.compile(r"бага зэрэг|гаигуи|\bсул\b|жаахан"), 3),
]

_TEMP_RES = [
    re.compile(r"\b(3[4-9]|4[0-2])(?:[.,](\d))?\s*(?:°|градус|хэм\b|c\b|с\b)"),
    re.compile(r"(?:халуун|температур|халуур)\D{0,15}\b(3[5-9]|4[0-2])(?:[.,](\d))?\b"),
    re.compile(r"\b(3[5-9]|4[0-2])[.,](\d)\b"),
]
_AGE_YEARS = re.compile(r"\b(\d{1,3})\s*(?:нас|наст|years? old|yo\b)")
_AGE_MONTHS = re.compile(r"\b(\d{1,2})\s*(?:сарта|сарын\s*(?:хуухэд|няраи)|months? old)")
_AGE_DAYS = re.compile(r"\b(\d{1,2})\s*(?:хоногто)")
_BP = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b")


def _to_number(tok: str) -> float | None:
    tok = tok.replace(",", ".")
    if tok in NUMBER_WORDS:
        return float(NUMBER_WORDS[tok])
    try:
        return float(tok)
    except ValueError:
        return None


def extract_facts(folded: str) -> Facts:
    f = Facts()

    # duration
    m = _DURATION_RE.search(folded)
    if m:
        n = _to_number(m.group(1))
        unit = re.sub(r"\s+", " ", m.group(2))
        days = (n * 7) if unit.startswith("долоо") else (n * _UNIT_DAYS.get(unit, 1) if n is not None else None)
        if days is not None:
            f.duration_days = round(days, 3)
            f.duration_text = m.group(0)
    if f.duration_days is None:
        for rx, days, text in _RELATIVE:
            if rx.search(folded):
                f.duration_days, f.duration_text = days, text
                break
    if f.duration_days is None:
        m = _EN_DURATION.search(folded)
        if m:
            n = float(m.group(1))
            unit = m.group(2)
            f.duration_days = n * (7 if unit.startswith("week") else (1 / 24 if unit.startswith("hour") else 1))
            f.duration_text = m.group(0)

    # severity 1-10
    for rx in _SEVERITY_NUM:
        m = rx.search(folded)
        if m:
            f.severity = int(m.group(1))
            break
    if f.severity is None:
        for rx, score in _SEVERITY_WORDS:
            if rx.search(folded):
                f.severity = score
                break

    # temperature
    for rx in _TEMP_RES:
        m = rx.search(folded)
        if m:
            f.temperature_c = float(m.group(1) + "." + (m.group(2) or "0"))
            break

    # age
    m = _AGE_YEARS.search(folded)
    if m and 0 < int(m.group(1)) <= 120:
        f.age_years = float(m.group(1))
    m = _AGE_MONTHS.search(folded)
    if m:
        f.age_months = float(m.group(1))
        f.age_years = round(f.age_months / 12, 2)
    m = _AGE_DAYS.search(folded)
    if m and f.age_months is None:
        f.age_months = round(int(m.group(1)) / 30, 2)
        f.age_years = 0.0

    # blood pressure (e.g. 180/110) — "8/10" is excluded by the ranges
    for m in _BP.finditer(folded):
        sys_, dia = int(m.group(1)), int(m.group(2))
        if 70 <= sys_ <= 260 and 40 <= dia <= 160 and sys_ > dia:
            f.bp_systolic, f.bp_diastolic = sys_, dia
            break

    # sex (only if volunteered)
    if re.search(r"\bэрэгтэи\b|\bmale\b", folded):
        f.sex = "male"
    elif re.search(r"\bэмэгтэи\b|\bfemale\b", folded):
        f.sex = "female"
    return f


def redact_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for kind, rx, repl in _PII_PATTERNS:
        if rx.search(text):
            found.append(kind)
            text = rx.sub(repl, text)
    return text, found


class InputProcessor:
    def __init__(self, lexicon: ConceptLexicon):
        self.lexicon = lexicon

    def process(self, text: str) -> NormalizedInput:
        clean = normalize_unicode(text)
        redacted, pii = redact_pii(clean)
        display = to_display_cyrillic(redacted)
        folded = fold(display)
        hits = [
            ConceptHit(concept=m.concept, label_mn=m.label_mn, type=m.type, negated=m.negated)
            for m in self.lexicon.match(redacted)
        ]
        return NormalizedInput(
            original=clean,
            redacted=redacted,
            display_text=display,
            folded_text=folded,
            script=script_of(clean),
            pii_types=pii,
            concepts=hits,
            facts=extract_facts(folded),
        )
