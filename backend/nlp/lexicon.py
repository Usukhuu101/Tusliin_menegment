"""
Concept lexicon: maps many ways of writing a health concept to ONE id.

    "толгой өвдөөд", "tolgoi ovdood", "headache"  ->  headache

The lexicon lives in data/nlp/concept_lexicon.yaml (data, not code), so the
team can add synonyms without touching Python. The same lexicon feeds the
input processor, triage rules, RAG query expansion and the symptom agent.

Pattern syntax (in YAML):
  "толгой өвд"   -> two stems, in this order, within a few words.
                   Each stem is a PREFIX, so "өвд" matches өвдөх/өвдөөд/өвдсөн.
  "эм$"          -> "$" = the word must match exactly (short stems like "эм"
                   would otherwise match "эмч", "эмнэлэг", ...).
Negation: "халуураагүй", "толгой өвдөхгүй", "... байхгүй" are detected and the
concept is reported as negated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .text_utils import fold, folded_tokens, levenshtein, transliterate_latin

WINDOW = 4  # max distance (in words) between consecutive stems of one pattern
NEGATION_WORDS = {"угуи", "баихгуи", "бхгуи", "биш", "алга"}
NEG_SUFFIX = "гуи"
# Frequent NON-medical words that are one letter away from a medical stem.
# Tokens starting with these (folded) are never fuzzy-matched.
#   сүүлийн/сүүлд/сүүлээр ("recently") vs суулгах (diarrhea); агаар (air) vs угаар (CO)
NO_FUZZY_PREFIXES = ("суули", "суулд", "суулэ", "агаар", "дугаар", "халуун")
_LATIN_RE = re.compile(r"[a-z]")


@dataclass
class Concept:
    id: str
    label_mn: str
    type: str
    patterns: list[list[str]] = field(default_factory=list)   # folded stems
    english: list[str] = field(default_factory=list)
    # Safety-critical concepts (e.g. suicidal thoughts) are never "negated away":
    # a false alarm is far cheaper than a missed crisis.
    ignore_negation: bool = False


@dataclass
class ConceptMatch:
    concept: str
    label_mn: str
    type: str
    negated: bool
    start: int
    end: int


def _compile_pattern(pattern: str, latin: bool = False) -> list[str]:
    words = pattern.strip().split()
    stems = []
    for w in words:
        exact = w.endswith("$")
        core = w[:-1] if exact else w
        if latin:
            core = transliterate_latin(core)
        core = fold(core)
        stems.append(core + ("$" if exact else ""))
    return stems


def _fuzzy_prefix(token: str, stem: str) -> bool:
    n = len(stem)
    # On short stems a different LAST letter usually means a different word
    # ("суули-йн" vs "суулг-ах"), not a typo -> not a match. An inserted letter
    # before the last one is still accepted ("давчи-д-жээ" vs "давчд").
    if n <= 5 and token[: n - 1] == stem[:-1] and stem[-1] not in token[n - 1 : n + 1]:
        return False
    return min(levenshtein(token[:k], stem, 1) for k in (n - 1, n, n + 1) if k <= len(token)) <= 1


class ConceptLexicon:
    def __init__(self, paths: list[Path]):
        self.concepts: dict[str, Concept] = {}
        self.version = "unknown"
        for p in paths:
            if p.exists():
                self._load(p)
        # every stem of 4+ letters: a token that starts with one of these is a word
        # we KNOW, so it must not be fuzzy-matched to a different stem
        # (prevents "халуун"=hot -> "халуур"=fever, "агаар"=air -> "угаар"=CO).
        self._known_stems = {s for c in self.concepts.values() for pat in c.patterns for s in pat
                             if not s.endswith("$") and len(s) >= 4}
        self._known_cache: dict[str, bool] = {}
        # one pre-compiled regex per concept for its English phrases (speed)
        self._english_re = {
            c.id: re.compile(r"\b(?:" + "|".join(re.escape(p) for p in sorted(set(c.english), key=len, reverse=True)) + r")\b")
            for c in self.concepts.values() if c.english
        }

    def _is_known_word(self, token: str) -> bool:
        if token not in self._known_cache:
            self._known_cache[token] = any(token.startswith(s) for s in self._known_stems)
        return self._known_cache[token]

    def _stem_match(self, token: str, stem: str) -> bool:
        if stem.endswith("$"):
            return token == stem[:-1]
        if token.startswith(stem):
            return True
        # one typo allowed (incl. swapped letters) — only for long stems, same first
        # letter, and only when the token is not already a known lexicon word
        if (len(stem) >= 5 and len(token) >= len(stem) - 1 and token[0] == stem[0]
                and not token.startswith(NO_FUZZY_PREFIXES) and not self._is_known_word(token)):
            return _fuzzy_prefix(token, stem)
        return False

    # ------------------------------------------------------------------ load
    def _load(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if self.version == "unknown":          # the main lexicon (loaded first) defines the version
            self.version = str(data.get("version", self.version))
        for cid, spec in (data.get("concepts") or {}).items():
            c = self.concepts.get(cid) or Concept(
                id=cid, label_mn=spec.get("label_mn", cid), type=spec.get("type", "symptom"),
                ignore_negation=bool(spec.get("ignore_negation", False)),
            )
            for pat in spec.get("mn", []) or []:
                c.patterns.append(_compile_pattern(pat))
            for pat in spec.get("latin", []) or []:
                c.patterns.append(_compile_pattern(pat, latin=True))
            c.english.extend(e.lower() for e in spec.get("en", []) or [])
            self.concepts[cid] = c

    # ----------------------------------------------------------------- query
    def label(self, concept_id: str) -> str:
        c = self.concepts.get(concept_id)
        return c.label_mn if c else concept_id

    def type_of(self, concept_id: str) -> str:
        c = self.concepts.get(concept_id)
        return c.type if c else "unknown"

    def ids_of_type(self, *types: str) -> set[str]:
        return {cid for cid, c in self.concepts.items() if c.type in types}

    # ----------------------------------------------------------------- match
    def match(self, text: str) -> list[ConceptMatch]:
        tokens = folded_tokens(text)
        lower = text.lower()
        has_latin = bool(_LATIN_RE.search(lower))     # English phrases only need checking then
        found: dict[str, ConceptMatch] = {}
        for c in self.concepts.values():
            best: ConceptMatch | None = None
            for stems in c.patterns:
                for start, end in self._find(tokens, stems):
                    neg = False if c.ignore_negation else self._is_negated(tokens, end, stems[-1])
                    m = ConceptMatch(c.id, c.label_mn, c.type, neg, start, end)
                    # a positive mention wins over a negated one
                    if best is None or (best.negated and not neg):
                        best = m
            eng = self._english_re.get(c.id) if has_latin else None
            if eng is not None:
                for mm in eng.finditer(lower):
                    neg = (not c.ignore_negation) and bool(
                        re.search(r"\b(no|not|without|never)\s+(\w+\s+){0,2}$", lower[: mm.start()])
                    )
                    m = ConceptMatch(c.id, c.label_mn, c.type, neg, -1, -1)
                    if best is None or (best.negated and not neg):
                        best = m
            if best:
                found[c.id] = best
        return list(found.values())

    def _find(self, tokens: list[str], stems: list[str]):
        for i, tok in enumerate(tokens):
            if not self._stem_match(tok, stems[0]):
                continue
            pos, ok = i, True
            for stem in stems[1:]:
                nxt = next(
                    (j for j in range(pos + 1, min(len(tokens), pos + 1 + WINDOW)) if self._stem_match(tokens[j], stem)),
                    None,
                )
                if nxt is None:
                    ok = False
                    break
                pos = nxt
            if ok:
                yield i, pos

    @staticmethod
    def _is_negated(tokens: list[str], end: int, last_stem: str) -> bool:
        last_tok = tokens[end]
        if last_tok.endswith(NEG_SUFFIX) and not last_stem.rstrip("$").endswith(NEG_SUFFIX):
            return True
        return any(t in NEGATION_WORDS for t in tokens[end + 1 : end + 3])
