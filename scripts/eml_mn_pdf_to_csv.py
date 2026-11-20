"""
Convert Mongolia's 9th Essential Medicines List (Зайлшгүй шаардлагатай эмийн IX жагсаалт,
Minister of Health order А/625, 2020-12-25; PDF from the WHO national-EML repository)
into a UTF-8 CSV — one row per dosage form / strength line.

    python scripts/eml_mn_pdf_to_csv.py [IN.pdf] [OUT.csv]
    defaults: data/raw/essential_medicines/mongolia_neml_2020.pdf -> data/cleaned/eml_mn_2020.csv

Then:  python scripts/build_eml_kb.py

Needs `pip install pdfplumber` (only for this one-off conversion, not for the app).

Based on the original eml_mn_pdf_to_csv.py, with these fixes:
  * a row split across a page break ("... (нугасаар" + "мэдээгүйжүүлэх)") is merged back
  * numbered items without a dosage form (blood components, № 211-214) are kept
  * Latin look-alike letters inside Cyrillic words are fixed ("кeтамин" -> "кетамин")
  * the chapter ("2. Өвдөлт намдаах ...") is kept next to the section ("2.1 Опийн бус бэлдмэл")
  * parsing stops at the age/precaution table that follows the list
The last column holds the pharmacy dispensing mark. Its legend is printed on p.67:
  ЖГ  = нийтийн үйлчилгээтэй эмийн санд жоргүй олгоно (no prescription)
  Ж   = нийтийн үйлчилгээтэй эмийн санд жороор олгоно (prescription)
  Ж*  = зөвхөн амбулаторийн үйлчлүүлэгчдэд үйлчлэх эмийн санд жороор олгоно
  СЖ  = жороор олгоно (сэтгэцэд нөлөөт эм)
  МЖ  = мансууруулах эмийн жороор олгоно
  ""  = not required in public pharmacies (a blank cell = "not mandatory at this level")
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "raw" / "essential_medicines" / "mongolia_neml_2020.pdf"
DEFAULT_OUT = ROOT / "data" / "cleaned" / "eml_mn_2020.csv"

LEVELS = ["bag_feldsher", "soum_village_hc", "aimag_district_hosp", "specialised_centre"]
FIELDS = ["no", "inn_mn", "form_strength", *LEVELS, "public_pharmacy_mark", "chapter", "section", "pdf_page"]
END_MARKER = "Анхааруулга, нас болон тунгийн хязгаартай"

_LATIN_LOOKALIKE = str.maketrans("aeopcxykmtABEKMHOPCTXY", "аеорсхукмтАВЕКМНОРСТХУ")
_CYR = re.compile(r"[а-яёөүА-ЯЁӨҮ]")


def fix_lookalikes(text: str) -> str:
    """Latin a/e/o/p/c/x... typed inside a Cyrillic word -> Cyrillic. Latin words stay Latin."""
    return re.sub(r"\S+", lambda m: m.group(0).translate(_LATIN_LOOKALIKE) if _CYR.search(m.group(0)) else m.group(0), text)


def norm(x) -> str:
    x = re.sub(r"\s+", " ", (x or "").replace("\n", " ")).strip()
    return fix_lookalikes(x)


def norm_header(x: str) -> str:
    # "1.1.1Амьсгалаар ..." -> "1.1.1 Амьсгалаар ..."
    return re.sub(r"^(\d+(?:\.\d+)*\.?)\s*(?=\D)", r"\1 ", x).strip()


def convert(pdf: Path, out: Path) -> list[dict]:
    import pdfplumber  # imported here so the rest of the project does not need it

    rows: list[dict] = []
    no = inn = chapter = section = ""
    done = False
    with pdfplumber.open(pdf) as p:
        for pi, page in enumerate(p.pages, 1):
            if done:
                break
            for tb in page.extract_tables():
                for raw in tb:
                    r = [norm(c) for c in raw] + [""] * 10
                    # None = a cell merged with the one ABOVE (same marks as the previous form row);
                    # "" = a really empty cell (e.g. the tail of a row split by a page break)
                    inherits = len(raw) >= 8 and all(raw[i] is None for i in range(3, 8))
                    if r[0] and not r[1] and not r[2]:                     # a heading row
                        head = norm_header(r[0])
                        if head.startswith(END_MARKER):
                            done = True
                            break
                        section = head
                        if re.match(r"^\d+\.\s*(?!\d)", head):            # "2. ..." = chapter
                            chapter = head
                        continue
                    if r[2] == "Тун хэмжээ, хэлбэр":                     # repeated column header
                        continue
                    numbered = bool(re.fullmatch(r"\d{1,4}", r[0]))
                    marks = r[3:8]
                    if not r[0] and not r[1] and r[2] and not any(marks) and rows:
                        if inherits:                                       # another form, same marks
                            rows.append({**rows[-1], "form_strength": r[2], "pdf_page": pi})
                        else:                                              # continuation of the row above
                            rows[-1]["form_strength"] = f"{rows[-1]['form_strength']} {r[2]}".strip()
                        continue
                    if not r[2] and not numbered:
                        continue
                    if numbered:
                        no, inn = r[0], r[1]
                    elif r[1]:
                        inn = r[1]
                    rows.append({"no": no, "inn_mn": inn, "form_strength": r[2],
                                 **{LEVELS[i]: r[3 + i] for i in range(4)},
                                 "public_pharmacy_mark": r[7], "chapter": chapter, "section": section,
                                 "pdf_page": pi})
                if done:
                    break
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    rows = convert(pdf, out)
    print(len(rows), "rows;", len({r["no"] for r in rows if r["no"]}), "distinct item numbers ->", out)


if __name__ == "__main__":
    main()
