"""
Import WHO Mongolia news (data/raw/who/) into a cleaned, de-duplicated index.

Input:  data/raw/who/who_mongolia_health_data.json   (full text, scraped)
        data/raw/who/who_mongolia_markdown11.md      (clean titles + summaries)
Output: data/cleaned/who_news.json

Cleaning:
  * correct titles (the scraper sometimes captured a sub-heading such as
    "Notes to editors") using the markdown version, matched by URL
  * remove repeated sentences (one article body appears twice) and photo credits
  * date from the URL (dd-mm-yyyy), topic tags, and which KB documents cite it

News is NOT medical guidance and is English, so it is not indexed directly.
Team members write reviewed Mongolian KB documents from it (status REVIEW) and
cite the article URL — `used_in_kb` shows which articles are already covered.

Run:  python scripts/import_who_news.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "who"
OUT = ROOT / "data" / "cleaned" / "who_news.json"
KB = ROOT / "data" / "knowledge_base"

TOPICS = {
    "measles": "measles", "hpv": "hpv", "air pollution": "air_pollution", "coal": "air_pollution",
    "hypertension": "hypertension", "tobacco": "tobacco", "e-cigarette": "tobacco", "violence": "violence",
    "mental health": "stress", "ageing": "elderly", "immunization": "vaccination", "vaccin": "vaccination",
    "artificial intelligence": "ai_in_health", "climate": "climate",
}
BAD_TITLES = {"notes to editors", "learning together across the region"}


def md_sections(text: str) -> dict[str, dict]:
    """url -> {title, summary} from the markdown file."""
    out = {}
    for block in re.split(r"^## \d+\.\s+", text, flags=re.M)[1:]:
        title = block.splitlines()[0].strip()
        m = re.search(r"\*\*URL:\*\*\s*(\S+)", block)
        if m:
            body = block.split(m.group(0), 1)[1]
            body = re.sub(r"\n-{3,}\s*$", "", body).strip()
            out[m.group(1).strip()] = {"title": title, "summary": body}
    return out


def dedupe(text: str) -> str:
    text = re.sub(r"[^\n]*(Photo credit:|©)[^\n]*\n?", "", text)
    seen, kept = set(), []
    for sent in re.split(r"(?<=[.!?”\"])\s+", text.replace("\n", " ")):
        key = re.sub(r"\W+", "", sent.lower())
        if key and key not in seen:
            seen.add(key)
            kept.append(sent.strip())
    return " ".join(kept)


def main() -> None:
    articles = json.loads((RAW / "who_mongolia_health_data.json").read_text(encoding="utf-8"))
    md = md_sections((RAW / "who_mongolia_markdown11.md").read_text(encoding="utf-8"))
    kb_files = {p: p.read_text(encoding="utf-8") for p in KB.rglob("*.md")}
    rows = []
    for i, a in enumerate(articles, 1):
        url = a["url"].strip()
        title = a["title"].strip()
        fixed = False
        if title.lower() in BAD_TITLES and url in md:
            title, fixed = md[url]["title"], True
        clean = dedupe(a["content"])
        d = re.search(r"/(\d{2})-(\d{2})-(\d{4})-", url)
        low = clean.lower()
        rows.append({
            "id": f"WHO_NEWS_{i:03d}",
            "title": title,
            "title_corrected": fixed,
            "url": url,
            "date": f"{d.group(3)}-{d.group(2)}-{d.group(1)}" if d else None,
            "source": a.get("source", "WHO Mongolia"),
            "language": "en",
            "topics": sorted({t for k, t in TOPICS.items() if k in low}),
            "used_in_kb": sorted(p.stem.split("_")[0] + "_" + p.stem.split("_")[1] + "_" + p.stem.split("_")[2]
                                 for p, txt in kb_files.items() if url in txt),
            "chars_raw": len(a["content"]),
            "chars_clean": len(clean),
            "summary_md": md.get(url, {}).get("summary", ""),
            "content": clean,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} WHO articles -> {OUT.relative_to(ROOT)}")
    for r in rows:
        flag = " (title fixed)" if r["title_corrected"] else ""
        saved = r["chars_raw"] - r["chars_clean"]
        print(f"  {r['id']} {r['date'] or 'no-date   '} {r['title'][:60]:<60}{flag}")
        print(f"      topics={r['topics']} used_in_kb={r['used_in_kb'] or '—'} removed={saved} chars")


if __name__ == "__main__":
    main()
