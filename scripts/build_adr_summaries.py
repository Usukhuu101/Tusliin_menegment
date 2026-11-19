"""
Turn the MMRA adverse-drug-reaction (ADR) spreadsheets in data/raw/ into
chatbot knowledge — WITHOUT exposing individual patients.

Outputs:
  data/cleaned/adr_records.csv                one row per unique report, NO age/sex/facility/dates
  data/cleaned/adr_summary_by_drug.json       aggregated counts per drug
  data/nlp/generated_drug_lexicon.yaml        drug names -> concepts (so users can ask by name)
  data/knowledge_base/medications/generated/MED_ADR_*.md   status: REVIEW (needs pharmacist review)

Run:  python scripts/build_adr_summaries.py [--min-reports 5] [--top 15]
Then: python -m backend.rag.indexer
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.nlp.lexicon import ConceptLexicon  # noqa: E402
from backend.nlp.text_utils import fold, levenshtein  # noqa: E402

RAW = ROOT / "data" / "raw" / "regulatory_mmra"
CLEAN = ROOT / "data" / "cleaned"
KB_OUT = ROOT / "data" / "knowledge_base" / "medications" / "generated"
LEX_OUT = ROOT / "data" / "nlp" / "generated_drug_lexicon.yaml"
ADR_FILE_HINTS = ("ADR", "гаж нөлөө", "opendata")

HEADER_KEYS = {
    "reg_no": lambda h: "бүртгэлийн" in h,
    "inn": lambda h: "олон улсын нэршил" in h,
    "trade": lambda h: "худалдааны нэршил" in h,
    "reaction": lambda h: ("гаж нөлөө" in h and "шинж" in h) or "илэрсэн гаж нөлөө" in h,
    "severity": lambda h: "хүнд, хөнгөний" in h,
    "outcome": lambda h: "төгсгөл" in h,
    "atc": lambda h: h.strip() in ("atc", "atc ангилал", "atc код"),
    "naranjo": lambda h: "наранжо" in h,
    "date": lambda h: "ирүүлсэн огноо" in h,
}
_LAT2CYR_SKIP = re.compile(r"[a-z]", re.I)


def norm_header(v) -> str:
    return re.sub(r"\s+", " ", str(v).replace("\n", " ")).strip().lower()


DRUG_INFO = lambda h: "илэрсэн эмий" in h  # noqa: E731  combined "drug information" column


def _is_data_row(vals) -> bool:
    return bool(re.match(r"^\d", str(vals[0]).strip())) if vals and not pd.isna(vals[0]) else False


def is_header_row(vals) -> bool:
    hs = [norm_header(v) for v in vals if not pd.isna(v)]
    hits = sum(1 for h in hs for t in list(HEADER_KEYS.values()) + [DRUG_INFO] if t(h))
    return hits >= 2 or (bool(hs) and hs[0] == "№" and len(hs) >= 4)


def detect_columns(rows: list[list]) -> dict[str, int]:
    """Detect column positions from a header row + up to 2 sub-header rows.
    Sheets change layout part-way down, so this is called for EVERY header block."""
    cols: dict[str, int] = {}
    drug_info = None
    for vals in rows[:3]:
        if rows.index(vals) > 0 and _is_data_row(vals):
            break
        for c, v in enumerate(vals):
            if pd.isna(v):
                continue
            h = norm_header(v)
            if DRUG_INFO(h) and drug_info is None:
                drug_info = c
            for key, test in HEADER_KEYS.items():
                if key not in cols and test(h):
                    cols[key] = c
    if "inn" not in cols and "trade" not in cols and drug_info is not None:
        cols["inn"] = drug_info
    return cols


def drug_key(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"\(.*?\)|/.*?/", " ", s)
    s = re.split(r"[,+;]", s)[0]
    s = re.sub(r"[^a-zа-яёөү\s-]", " ", s)
    words = [w for w in s.split() if len(w) > 2 and w not in {"натри", "натрийн", "калийн", "гидрохлорид", "сульфат", "тарилгын", "уусмал"}]
    return words[0] if words else ""


def cell(row, idx):
    if idx is None or idx >= len(row):
        return ""
    v = row[idx]
    return "" if pd.isna(v) else re.sub(r"\s+", " ", str(v)).strip()


def extract_records() -> list[dict]:
    seen_files, records, seen_keys = set(), [], set()
    for path in sorted(RAW.glob("*.xlsx")):
        if not any(h.lower() in path.name.lower() for h in ADR_FILE_HINTS):
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen_files:            # exact duplicate file
            continue
        seen_files.add(digest)
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            if sheet.upper() == "PSUR":
                continue
            rows = xl.parse(sheet, header=None).values.tolist()
            cols: dict[str, int] = {}
            year = int(sheet) if re.fullmatch(r"20\d\d", sheet.strip()) else None
            in_header = False
            for i, vals in enumerate(rows):
                if is_header_row(vals) or (in_header and not _is_data_row(vals)):
                    if not in_header:              # a NEW header block starts after data rows
                        cols = detect_columns(rows[i:i + 3])
                    in_header = True
                    continue
                in_header = False
                if "reaction" not in cols or ("inn" not in cols and "trade" not in cols):
                    continue
                first = cell(vals, 0)
                nonempty = [v for v in vals if not pd.isna(v)]
                if len(nonempty) == 1 and re.fullmatch(r"20\d\d(\.0)?", str(nonempty[0]).strip()):
                    year = int(float(nonempty[0]))      # year separator rows inside 2017-2022 sheet
                    continue
                reaction = cell(vals, cols.get("reaction"))
                inn, trade = cell(vals, cols.get("inn")), cell(vals, cols.get("trade"))
                if not reaction or not (inn or trade) or not re.match(r"^\d", first or "x"):
                    continue
                reg = re.sub(r"\s+", "", cell(vals, cols.get("reg_no")))
                key = drug_key(inn) or drug_key(trade)
                dedup = reg if re.match(r"^(MN-MMRA-\d+|\d{6,})$", reg) else hashlib.md5(f"{key}|{reaction}".encode()).hexdigest()
                if dedup in seen_keys or not key:
                    continue
                seen_keys.add(dedup)
                records.append({
                    "drug_key": key, "_k_inn": drug_key(inn), "_k_trade": drug_key(trade),
                    "inn": inn, "trade_name": trade, "atc": cell(vals, cols.get("atc"))[:8],
                    "reaction_text": reaction, "severity": cell(vals, cols.get("severity")),
                    "outcome": cell(vals, cols.get("outcome")), "causality": cell(vals, cols.get("naranjo")),
                    "year": year or _year_from(cell(vals, cols.get("date"))) or "",
                    "source_file": path.name, "sheet": sheet,
                })
    # Some sheets have trade/INN columns swapped (e.g. "Ультравист" vs "Йопромид").
    # Canonical key = whichever of the two names is used more often as an INN overall.
    inn_freq = Counter(r["_k_inn"] for r in records if r["_k_inn"])
    for r in records:
        cands = [k for k in (r.pop("_k_inn"), r.pop("_k_trade")) if k]
        if cands:
            r["drug_key"] = max(cands, key=lambda k: (inn_freq[k], k == cands[0]))
    # Merge spelling variants ("никотины" / "никитоны", "иопромид" / "йопромид")
    freq = Counter(r["drug_key"] for r in records)
    canon: dict[str, str] = {}
    for k, _ in freq.most_common():
        target = next((c for c in canon.values() if len(k) >= 6 and levenshtein(fold(k), fold(c), 1) <= 1), None)
        canon[k] = target or k
    for r in records:
        r["drug_key"] = canon[r["drug_key"]]
    return records


def _year_from(text: str):
    m = re.match(r"^(20\d\d)", text.replace(" ", ""))
    return int(m.group(1)) if m else None


def severity_bucket(s: str) -> str:
    s = s.lower()
    if "ноцтой" in s:
        return "ноцтой"
    if "хүнд" in s:
        return "хүнд"
    if "хөнгөн" in s:
        return "хөнгөн"
    return "тодорхойгүй"


def display_name(names: Counter, key: str) -> str:
    ordered = [n for n, _ in names.most_common()]
    matching = [n for n in ordered if drug_key(n) == key and not _LAT2CYR_SKIP.search(n)]
    cyr = [n for n in ordered if not _LAT2CYR_SKIP.search(n)]
    best = (matching or cyr or ordered)[0]
    best = re.split(r"[,(/]", best)[0].strip()
    return best[:1].upper() + best[1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reports", type=int, default=5)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    lexicon = ConceptLexicon([ROOT / "data" / "nlp" / "concept_lexicon.yaml"])
    records = extract_records()
    CLEAN.mkdir(parents=True, exist_ok=True)
    with (CLEAN / "adr_records.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    by_drug: dict[str, dict] = defaultdict(lambda: {"n": 0, "names": Counter(), "concepts": Counter(),
                                                    "severity": Counter(), "years": Counter(), "atc": Counter()})
    overall = Counter()
    for r in records:
        d = by_drug[r["drug_key"]]
        d["n"] += 1
        d["names"][r["inn"] or r["trade_name"]] += 1
        d["severity"][severity_bucket(r["severity"])] += 1
        if r["year"]:
            d["years"][int(r["year"])] += 1
        if r["atc"]:
            d["atc"][r["atc"][:1].upper()] += 1
        for m in lexicon.match(r["reaction_text"]):
            if not m.negated and m.type in ("symptom", "red_flag"):
                d["concepts"][m.concept] += 1
                overall[m.concept] += 1
        if "анафилакс" in r["reaction_text"].lower():
            d["concepts"]["anaphylaxis"] += 1
            overall["anaphylaxis"] += 1

    summary = {
        k: {"reports": v["n"], "display_name": display_name(v["names"], k),
            "names": dict(v["names"].most_common(5)), "top_reactions": dict(v["concepts"].most_common(6)),
            "severity": dict(v["severity"]), "years": dict(sorted(v["years"].items())), "atc": dict(v["atc"])}
        for k, v in sorted(by_drug.items(), key=lambda kv: -kv[1]["n"])
    }
    years = sorted({int(r["year"]) for r in records if r["year"]})
    (CLEAN / "adr_summary_by_drug.json").write_text(json.dumps({
        "total_reports": len(records), "unique_drugs": len(summary), "years": [years[0], years[-1]] if years else [],
        "top_reactions_overall": dict(overall.most_common(12)), "drugs": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- generated lexicon + KB docs for the most-reported drugs
    label = lambda c: "анафилакси (хүнд харшил)" if c == "anaphylaxis" else lexicon.label(c)  # noqa: E731
    top = [(k, v) for k, v in summary.items() if v["reports"] >= args.min_reports][: args.top]
    KB_OUT.mkdir(parents=True, exist_ok=True)
    for old in KB_OUT.glob("MED_ADR_*.md"):
        old.unlink()
    lex = {"version": "generated", "concepts": {}}
    yr = f"{years[0]}–{years[-1]}" if years else ""
    for i, (key, v) in enumerate(top, 1):
        cid = f"drug_{i:03d}"
        names = {key} | {re.split(r"[,(/]", n)[0].strip().lower() for n in v["names"]}
        lex["concepts"][cid] = {"label_mn": v["display_name"], "type": "medication",
                                "mn": sorted(n for n in names if n and len(n) > 3 and not _LAT2CYR_SKIP.search(n)),
                                "en": sorted(n for n in names if n and len(n) > 3 and _LAT2CYR_SKIP.search(n))}
        reactions = ", ".join(f"{label(c)} ({n})" for c, n in v["top_reactions"].items()) or "тодорхойгүй"
        serious = v["severity"].get("хүнд", 0) + v["severity"].get("ноцтой", 0)
        doc = f"""---
id: MED_ADR_{i:03d}
title: "{v['display_name']}: Монголд бүртгэгдсэн гаж нөлөө"
category: medication
concepts: [{cid}, side_effect, medication]
source: "Монгол Улсын эмийн хяналтын байгууллага (MMRA) — гаж нөлөөний бүртгэл"
source_tier: 1
source_url: "data/raw/regulatory_mmra (MMRA ADR reports)"
published_date: null
last_reviewed: null
reviewer: null
version: "1.0"
language: mn
status: REVIEW
notes: "AUTO-GENERATED by scripts/build_adr_summaries.py from {v['reports']} de-duplicated reports. Aggregated counts only. Requires pharmacist review."
---
## Ерөнхий мэдээлэл
Монголын эмийн гаж нөлөөний бүртгэлд ({yr} он) {v['display_name']} эмтэй холбоотой {v['reports']} тохиолдол мэдээлэгдсэн байна. Хамгийн их мэдээлэгдсэн шинж: {reactions}. Эдгээрээс {serious} тохиолдлыг хүнд эсвэл ноцтой гэж үнэлсэн. Бүртгэгдсэн тохиолдол нь эм тухайн шинжийг заавал үүсгэсэн гэсэн үг биш бөгөөд эмийн аюулгүй байдлыг бүрэн тодорхойлохгүй.

## Гэртээ юу хийж болох вэ
- Энэ эмийг хэрэглэж байгаад шинэ шинж илэрвэл эмчлэгч эмч эсвэл эм зүйчдээ даруй мэдэгдэх
- Эмийн хэрэглээг өөрийн дураар өөрчлөхөөсөө өмнө эмчтэйгээ зөвлөлдөх
- Өмнө нь энэ эмэнд харшил өгч байсан бол эмнэлэгт очих бүртээ хэлэх
- Гаж нөлөөг эмнэлэг, эмийн сангаар дамжуулан эмийн хяналтын байгууллагад мэдээлж болно

## Анхаарах шинж
- Уруул, нүүр, хоолой хавдах
- Амьсгал давчдах, хоолой бачуурах
- Толгой эргэж ухаан балартах, арьс цайх
- Арьс хуурч цэврүүтэх
Эдгээр шинж илэрвэл даруй 103 дугаарт залгаарай.

## Эмчид хэзээ хандах вэ
- Эм хэрэглэсний дараа тууралт, загатнаа, дотор муухайрах зэрэг шинж илэрвэл
- Гаж нөлөө удаан үргэлжлэх эсвэл давтагдвал
"""
        (KB_OUT / f"MED_ADR_{i:03d}_{re.sub(r'[^a-zа-яөү0-9]+', '_', key)}.md").write_text(doc, encoding="utf-8")
    LEX_OUT.write_text("# AUTO-GENERATED by scripts/build_adr_summaries.py — do not edit by hand\n"
                       + yaml.safe_dump(lex, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"unique ADR reports: {len(records)} | drugs: {len(summary)} | years: {yr}")
    print("top reactions overall:", ", ".join(f"{label(c)}={n}" for c, n in overall.most_common(8)))
    print(f"generated {len(top)} medication documents -> {KB_OUT.relative_to(ROOT)}")
    for i, (k, v) in enumerate(top, 1):
        print(f"  MED_ADR_{i:03d}  {v['display_name']:<28} reports={v['reports']}")


if __name__ == "__main__":
    main()
