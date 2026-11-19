"""
Move the original data files from data/ into data/raw/<category>/ and write
data/raw/manifest.csv (md5 hash, size, duplicate detection).

- Nothing is deleted. Exact duplicates are only flagged in the manifest.
- Safe to run multiple times (already-moved files are skipped).

Run:  python scripts/organize_raw_data.py
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

# (substring in filename -> (category folder, dataset description))
RULES: list[tuple[str, str, str]] = [
    ("ADR", "regulatory_mmra", "Adverse drug reaction (ADR) case reports"),
    ("гаж нөлөө", "regulatory_mmra", "Adverse drug reaction (ADR) case reports"),
    ("opendata", "regulatory_mmra", "MMRA open data: ADR + PSUR sheets"),
    ("PSUR", "regulatory_mmra", "Periodic safety update report (PSUR) summaries"),
    ("аюулгүй байдал", "regulatory_mmra", "Periodic safety update report (PSUR) summaries"),
    ("lab", "regulatory_mmra", "Drug testing laboratory statistics"),
    ("тусгай зөвшөөр", "regulatory_mmra", "Pharma license statistics"),
    ("үндэсний үйлдвэр", "regulatory_mmra", "Domestic pharmaceutical manufacturers"),
    ("Халдварт", "statistics", "Infectious disease incidence per 10,000 by aimag/month"),
    ("Нас баралт", "statistics", "Mortality counts by aimag/month"),
    ("68-18ff", "statistics", "Mortality counts 2023 by aimag/month (csv copy)"),
    ("94-34c2", "statistics", "Mortality counts 2023 by aimag/month (json copy)"),
    ("72-430e", "statistics", "Infectious disease incidence 2020-2022 (xlsx)"),
    ("a7-fa72", "statistics", "Infectious disease incidence 2022 (json copy)"),
    ("dc-5bf3", "statistics", "Infectious disease incidence 2023, first 6 months (xlsx)"),
    ("f3-b49f", "statistics", "Infectious disease incidence 2023, first 6 months (csv copy)"),
    ("who_mongolia_health_data", "who", "WHO Mongolia news articles, full text (7 articles, Oct-Nov 2026, English)"),
    ("who_mongolia_markdown", "who", "WHO Mongolia news articles, cleaned summaries (same 7 articles)"),
    ("medlineplus", "medlineplus", "MedlinePlus health-topic synonyms / cross-references (English)"),
]
SOURCES = {
    "regulatory_mmra": "MMRA (MN-MMRA ids)",
    "statistics": "Health statistics (to be confirmed)",
    "who": "WHO Mongolia (who.int/mongolia)",
    "medlineplus": "MedlinePlus, U.S. National Library of Medicine (medlineplus.gov)",
}


def classify(name: str) -> tuple[str, str]:
    for key, folder, desc in RULES:
        if key.lower() in name.lower():
            return folder, desc
    return "other", "Unclassified"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in sorted(DATA.iterdir()):
        if f.is_file() and f.suffix.lower() in {".xlsx", ".csv", ".json", ".md"} and f.name.lower() != "readme.md":
            folder, _ = classify(f.name)
            target_dir = RAW / folder
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f.name
            if target.exists():
                print(f"skip (already exists): {target.relative_to(ROOT)}")
                continue
            shutil.move(str(f), str(target))
            moved += 1
    print(f"moved {moved} file(s) into data/raw/")

    # Build manifest over everything in data/raw
    rows, seen = [], {}
    for f in sorted(p for p in RAW.rglob("*") if p.is_file() and p.name != "manifest.csv"):
        h = md5(f)
        folder, desc = classify(f.name)
        rows.append({
            "file": str(f.relative_to(RAW)).replace("\\", "/"),
            "category": folder,
            "dataset": desc,
            "md5": h,
            "size_bytes": f.stat().st_size,
            "duplicate_of": seen.get(h, ""),
            "source": SOURCES.get(folder, "unknown"),
        })
        seen.setdefault(h, rows[-1]["file"])
    with (RAW / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    dups = sum(1 for r in rows if r["duplicate_of"])
    print(f"manifest: {len(rows)} files, {dups} exact duplicate(s) -> data/raw/manifest.csv")


if __name__ == "__main__":
    main()
