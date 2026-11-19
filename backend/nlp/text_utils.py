"""
Low-level Mongolian text helpers.

Key idea: "folding". Mongolian is typed in many ways (Cyrillic, Latin chat
script, with/without ө/ү, with typos). For MATCHING we map every input into
one simplified Cyrillic alphabet:

    ө -> о, ү -> у, ё -> е -> э, й -> и, ы -> и, ъ/ь removed

Latin text is first transliterated into Cyrillic, then folded. So
"tolgoi owdood", "толгой өвдөөд" and "толгои овдоод" all become
"толгои овдоод".

Folded text is only used internally for matching/search, never shown to users.
"""
from __future__ import annotations

import re
import unicodedata

_FOLD_MAP = str.maketrans({
    "ө": "о", "ү": "у", "ё": "э", "е": "э", "й": "и", "ы": "и", "ъ": "", "ь": "",
    "Ө": "о", "Ү": "у", "Ё": "э", "Е": "э", "Й": "и", "Ы": "и", "Ъ": "", "Ь": "",
})

# Latin chat-script -> Cyrillic (multi-letter combinations first)
_LATIN_MULTI = [
    ("shch", "щ"), ("kh", "х"), ("ch", "ч"), ("sh", "ш"), ("ts", "ц"),
    ("ya", "я"), ("yu", "ю"), ("yo", "ё"), ("ye", "е"),
]
_LATIN_SINGLE = {
    "a": "а", "b": "б", "c": "ц", "d": "д", "e": "э", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "ө", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "х",
    "y": "ү", "z": "з", "ö": "ө", "ü": "ү",
}

# Very common chat abbreviations (Latin and Cyrillic) -> full word
CHAT_WORDS = {
    "bn": "байна", "bna": "байна", "bnaa": "байна", "baina": "байна", "bnu": "байна уу",
    "бн": "байна", "бна": "байна", "бнаа": "байна",
    "bh": "байх", "bhgui": "байхгүй", "baihgui": "байхгүй", "бхгүй": "байхгүй",
    "ugui": "үгүй", "ygui": "үгүй", "vgui": "үгүй", "tiim": "тийм",
    "ym": "юм", "yum": "юм", "uu": "уу", "vv": "үү", "yu": "юу", "ve": "вэ", "we": "вэ",
    "sn": "сайн", "sain": "сайн", "snuu": "сайн уу",
}

_CYR_RE = re.compile(r"[а-яёөү]", re.IGNORECASE)
_LAT_RE = re.compile(r"[a-z]", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-zа-яёөүöü0-9]+", re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    # Latin look-alikes typed inside Cyrillic words are common; keep as is,
    # transliteration handles them. Collapse whitespace.
    return re.sub(r"\s+", " ", text).strip()


def fold(text: str) -> str:
    return text.lower().translate(_FOLD_MAP)


def transliterate_latin(word: str) -> str:
    """Best-effort Latin -> Cyrillic for one word (not folded)."""
    w = word.lower()
    if w in CHAT_WORDS:
        return CHAT_WORDS[w]
    out, i = [], 0
    while i < len(w):
        for latin, cyr in _LATIN_MULTI:
            if w.startswith(latin, i):
                out.append(cyr)
                i += len(latin)
                break
        else:
            out.append(_LATIN_SINGLE.get(w[i], w[i]))
            i += 1
    return "".join(out)


def script_of(text: str) -> str:
    cyr = len(_CYR_RE.findall(text))
    lat = len(_LAT_RE.findall(text))
    if cyr == 0 and lat == 0:
        return "empty"
    if lat == 0:
        return "cyrillic"
    if cyr == 0:
        return "latin"
    return "mixed"


def raw_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def to_display_cyrillic(text: str) -> str:
    """Transliterate Latin words to Cyrillic for display / LLM input.
    Cyrillic words and numbers are kept. English medical words stay as-is
    when the text is mixed (e.g. 'headache болоод байна')."""
    parts = re.split(r"(\s+|[^\w]+)", text)
    mixed = script_of(text) == "mixed"
    out = []
    for p in parts:
        if p and _LAT_RE.search(p) and not _CYR_RE.search(p):
            out.append(p if mixed and len(p) > 3 and p.lower() not in CHAT_WORDS else transliterate_latin(p))
        else:
            out.append(p)
    return "".join(out)


def folded_tokens(text: str) -> list[str]:
    """Tokens for matching: Latin tokens transliterated, everything folded."""
    toks = []
    for tok in raw_tokens(text):
        if _LAT_RE.search(tok) and not _CYR_RE.search(tok):
            expanded = transliterate_latin(tok)
            toks.extend(fold(expanded).split())
        else:
            toks.append(fold(tok))
    return toks


def levenshtein(a: str, b: str, limit: int = 2) -> int:
    """Edit distance where swapping two neighbouring letters counts as ONE typo
    (Damerau / optimal string alignment). Returns limit+1 when larger."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(rows[-1][j] + 1, cur[j - 1] + 1, rows[-1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], rows[-2][j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        rows.append(cur)
    return rows[-1][-1]


# Number words (folded) -> int, used for durations like "хоёр хоног"
NUMBER_WORDS = {
    "нэг": 1, "хоэр": 2, "гурав": 3, "гурван": 3, "доров": 4, "дорвон": 4, "тав": 5,
    "таван": 5, "зургаа": 6, "зургаан": 6, "долоо": 7, "долоон": 7, "наим": 8,
    "наиман": 8, "эс": 9, "эсон": 9, "арав": 10, "арван": 10,
}
