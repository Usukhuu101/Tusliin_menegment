"""
Turn Mongolia's Essential Medicines List (data/cleaned/eml_mn_2020.csv, made by
scripts/eml_mn_pdf_to_csv.py) into chatbot knowledge:

  data/knowledge_base/medications/eml/MED_EML_*.md   one document per medicine (status: REVIEW)
  data/nlp/generated_eml_lexicon.yaml                medicine names -> concepts

What a document says (all from the official list, MOH order А/625, 2020-12-25):
  * which therapeutic group the medicine is listed in
  * prescription status in public pharmacies (ЖГ / Ж / Ж* / СЖ / МЖ — legend on p.67 of the PDF)
  * dosage FORMS (tablet, syrup, injection ...), the official age / precaution note, care levels
What it deliberately does NOT say: strengths or doses. The chatbot never gives doses
(see backend/agents/safety_agent.py, MEDICATION_DOSE), so numbers like "200мг" are dropped.

The documents use `retrieval: concept`: they are only retrieved when the user names
the medicine. Without that, a question like "Сүрьеэ гэж юу вэ" would be "answered"
with an anti-TB drug's list entry instead of an honest fallback.

Run:  python scripts/build_eml_kb.py
Then: python -m backend.rag.indexer      (or just restart the app — it re-indexes)
"""
from __future__ import annotations

import csv
import re

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.nlp.text_utils import fold, levenshtein  # noqa: E402

CSV_IN = ROOT / "data" / "cleaned" / "eml_mn_2020.csv"
KB_OUT = ROOT / "data" / "knowledge_base" / "medications" / "eml"
LEX_OUT = ROOT / "data" / "nlp" / "generated_eml_lexicon.yaml"

SOURCE = "Эрүүл мэндийн сайдын 2020 оны А/625 тушаал — Зайлшгүй шаардлагатай эмийн IX жагсаалт"
SOURCE_URL = ("https://cdn.who.int/media/docs/default-source/essential-medicines/"
              "national-essential-medicines-lists-(neml)/wpro_neml/mongolia_neml_2020.pdf")


LEVELS = OrderedDict([
    ("bag_feldsher", "багийн эмчийн салбар"),
    ("soum_village_hc", "сум/тосгоны эрүүл мэндийн төв"),
    ("aimag_district_hosp", "аймаг/дүүргийн нэгдсэн эмнэлэг"),
    ("specialised_centre", "тусгай мэргэжлийн төв/төрөлжсөн мэргэшлийн эмнэлэг"),
])
# legend printed on p.67 of the PDF
MARK_SHORT = {"ЖГ": "жоргүй", "Ж": "жороор", "Ж*": "зөвхөн амбулаторийн үйлчлүүлэгчдэд үйлчилдэг эмийн санд жороор",
              "СЖ": "сэтгэцэд нөлөөт эмийн жороор", "МЖ": "мансууруулах эмийн тусгай жороор"}
RX_MARKS = {"Ж", "Ж*", "СЖ", "МЖ"}

# "Анхааруулга, нас болон тунгийн хязгаартай хэрэглэх эмүүд" table (PDF p.65-67), copied by hand
# because it is not a real table in the PDF. Keyed by the name as it appears in the list.
# ретинол is left out on purpose: its note is a dose.
AGE_NOTES = {
    "ацетилсалицилын хүчил": "12 наснаас дээш",
    "атропин": "3 сараас дээш",
    "бензилбензоат": "2 наснаас дээш",
    "бетаметазон": "арьсанд түрхэх хэлбэрийг нярайд хэрэглэхгүй (гидрокортизоноор орлуулна)",
    "ибупрофен": "3 сараас дээш (тарилгын уусмал хэлбэрт хамаарахгүй)",
    "лоперамид": "4 наснаас дээш",
    "метоклопрамид": "нярайд хэрэглэхгүй",
    "мизопростол": "зөвхөн эмнэлгийн нөхцөлд, эмчийн хяналтад хэрэглэнэ",
    "мифепристон": "зөвхөн эмнэлгийн нөхцөлд, эмчийн хяналтад хэрэглэнэ",
    "ондансетрон": "1 сараас дээш",
    "парацетамол": "үрэвслийн эсрэг эмчилгээнд хэрэглэхгүй",
    "пропофол": "1 сараас дээш",
    "сульфаметоксазол + триметоприм": "6 сараас дээш",
    "тетракаин": "нярайд хэрэглэхгүй",
    "цефазолин": "1 сараас дээш",
    "цефтриаксон": "нярайд залруулсан нас нь (жирэмсний хугацааг оруулан) 41 долоо хоногоос дээш бол",
    "ципрофлоксацин": "12 наснаас дээш",
    "флуоксетин": "8 наснаас дээш",
    "эфавиренз": "3 наснаас дээш эсвэл 10 кг-аас дээш жинтэй хүүхдэд",
    "мөнгөний сульфадиазин": "2 сараас дээш",
}

# PDF cell wraps that split a word ("эта\nмбутол") and obvious misprints of the same item
NAME_FIXES = {"эта мбутол": "этамбутол", "сукцин ат": "сукцинат", "цитофлавин )": "цитофлавин)",
              # misprints, checked against the raw PDF cells (item 568 itself prints "азатиоприн"; the
              # strengths of "лозартан+гидрохлорид" are those of losartan + hydrochlorothiazide).
              # The CSV keeps the PDF text; the documents use the corrected name.
              "азатиопирин": "азатиоприн", "пириндоприл": "периндоприл", "тербинафаин": "тербинафин",
              "спиринолактон": "спиронолактон", "броминидин": "бримонидин", "комипрамин": "кломипрамин",
              "дезогестерол": "дезогестрел", "этинилэтрадиол": "этинилэстрадиол", "бупивикайн": "бупивакаин",
              "ропивокайн": "ропивакаин", "флюросцен": "флуоресцеин", "флюроурацил": "флуороурацил",
              "ацетозоламид": "ацетазоламид", "клавулоны": "клавуланы",
              "лозартан+гидрохлорид": "лозартан+гидрохлортиазид"}
# vaccines are printed as disease names in the genitive ("улаан бурхан", "гепатит Б - ийн")
DISPLAY_NAMES = {
    "улаан бурхан": "Улаанбурхны вакцин",
    "гепатит А": "Гепатит А-гийн вакцин",
    "гепатит Б - ийн": "Гепатит Б-ийн вакцин",
    "гепатит Б - ийн иммуноглобулин": "Гепатит Б-ийн иммуноглобулин",
    "улаан бурхан - гахайн хавдар - улаануудын": "Улаанбурхан, гахайн хавдар, улаануудын хавсарсан вакцин",
    "сахуу - хөхүүл ханиад - татран - В гепатит - хемофилюс инфлюенза В-ийн":
        "Сахуу, хөхүүл ханиад, татран, В гепатит, хемофилюс инфлюенза В-ийн хавсарсан вакцин",
    "сахуу - татрангийн": "Сахуу, татрангийн хавсарсан вакцин",
}
# extra names people use (trade / common names); only well-established equivalents
ALIASES = {
    "эпинефрин (адреналин)": ["адреналин"],
    "ацетилсалицилын хүчил": ["аспирин", "ацетилсалицилийн хүчил"],
    "сульфаметоксазол + триметоприм": ["бисептол", "тримоксазол"],
    "сальбутамол": ["вентолин"],
    "дротаверин": ["ношпа", "но$ шпа"],          # the tokenizer splits "но-шпа" into "но" + "шпа"
    "амоксициллин +клавуланы хүчил": ["амоксиклав", "аугментин", "амоксициллин клавулан"],
    "аскорбины хүчил": ["аскорбин"],
    "вальпроны хүчил (натрийн вальпроат)": ["вальпроат"],
    "аденозинтрифосфат (АТФ)": ["атф$"],
    "метилтионинийн хлорид (метилен хөх)": ["метилен хөх"],
    "N-метилглюкамин сукцинат+натрийн хлорид+калийн хлорид+магнийн хлорид (реамберин)": ["реамберин"],
    "моноаммони глициризинат + глицин+DL- метионин (глицерон)": ["глицерон"],
    "инозин+никотинамид+рибофлавин+сукцинат (цитофлавин)": ["цитофлавин"],
    "циклоферон /меглюмина акридонацетат/": ["циклоферон"],
    "липиодол/этиодол": ["липиодол", "этиодол"],
    "трипан хөх нүдний уусмал (trypan blue ophthalmic solution)": ["трипан хөх"],
    "л-орнитин - л - аспартат": ["орнитин аспартат"],
    "натрийн хлорид": ["физиологийн уусмал"],
    "туберкулин, цэвэршүүлсэн протейны уусал": ["туберкулин", "манту"],
    "феназон, лидокаин гидрохлорид": ["отипакс"],
    "идэвхжүүлсэн нүүрс": ["идэвхт нүүрс"],
    # trade / common names used in Mongolia for a medicine that is on the list
    "метформин": ["глюкофаж"], "хлорамфеникол": ["левомицетин"], "метронидазол": ["трихопол"],
    "ципрофлоксацин": ["ципролет"], "азитромицин": ["сумамед"], "амброксол": ["лазолван"],
    "ацетилцистейн": ["ацц$"], "дидрогестерон": ["дюфастон"], "аминофиллин": ["эуфиллин"],
    "метоклопрамид": ["церукал"], "домперидон": ["мотилиум"], "лоперамид": ["имодиум"],
    "панкреатин": ["креон", "мезим"], "фосфолипид": ["эссенциале"], "силимарин": ["карсил"],
    "урсодеоксихолийн хүчил": ["урсосан"], "повидон иод": ["бетадин"], "калийн перманганат": ["марганцовка"],
    "устөрөгчийн хэт исэл": ["перекись"],
}
# Vaccines, sera and phages: the DISEASE word must be in the genitive ("томуугийн вакцин"), so that
# "Томуу туссан, вакцин хийлгээгүй" (a patient with flu) is not answered with the vaccine's list entry.
# Hepatitis A and B are told apart by their letter (Mongolian writes hepatitis B as "В гепатит").
PATTERN_OVERRIDES = {
    "сүрьеэгийн (БЦЖ)": ["сүрьеэгийн вакцин", "бцж$"],
    "гепатит А": ["гепатит а$ вакцин", "а$ гепатит вакцин"],
    "гепатит Б - ийн": ["гепатит б$ вакцин", "гепатит в$ вакцин", "в$ гепатит вакцин", "б$ гепатит вакцин"],
    "улаан бурхан": ["улаанбурхн вакцин", "улаан бурхн вакцин"],
    "улаан бурхан - гахайн хавдар - улаануудын": ["гахайн хавдр вакцин", "улаануудын вакцин"],
    "сахуу - хөхүүл ханиад - татран - В гепатит - хемофилюс инфлюенза В-ийн":
        ["пентавалент", "хөхүүл ханиадн вакцин", "хемофилюс вакцин"],
    "сахуу - татрангийн": ["сахуу татрангийн вакцин"],
    "татрангийн вакцин": ["татрангийн вакцин"],
    "халдварт саагийн": ["саагийн вакцин"],
    "томуугийн": ["томуугийн вакцин", "томуугаас вакцин"],
    "боомын вакцин": ["боомын$ вакцин"],
    "пневмококкын эсрэг вакцин": ["пневмококкын вакцин"],
    "хачигт энцефалитын эсрэг хатааж хөлдөөсөн вакцин": ["энцефалитын вакцин"],
    "балнадын": ["балнадын вакцин"],
    "галзуугийн": ["галзуугийн вакцин", "галзуугаас вакцин"],
    "менингококкийн": ["менингококкийн вакцин"],
    "тарваган тахлын": ["тахлын вакцин"],
    "шар чичрэгийн вакцин": ["шар$ чичрэгийн вакцин"],
    "хүний хөхөнцөр вирүсийн эсрэг вакцин": ["хөхөнцөр вирүсийн вакцин"],
    "татрангийн эсрэг ийлдэс": ["татрангийн ийлдэс"],
    "галзуугийн эсрэг иммуноглобулин": ["галзуугийн иммуноглобулин"],
    "боомын эсрэг иммуноглобулин": ["боомын иммуноглобулин"],
    "татрангийн эсрэг иммуноглобулин": ["татрангийн иммуноглобулин"],
    "хиамдалын ийлдэс (А,В,Е төрөл)": ["хиамдалын ийлдэс"],
    "сахуугийн ийлдэс": ["сахуугийн ийлдэс"],
    "гепатит Б - ийн иммуноглобулин": ["гепатит иммуноглобулин"],
    "хачигт энцефалитын иммуноглобулин": ["энцефалитын иммуноглобулин"],
    # "залгиур" alone is also the pharynx: "цусан суулгалт гараад залгиур өвдөж байна"
    "сальмонеллёзын нян залгиур": ["сальмонеллёзын нян$ залгиур"],
    "балнадын нян залгиур": ["балнадын нян$ залгиур"],
    "цусан суулгын нян залгиур": ["цусан суулгын нян$ залгиур"],
}
# lab results ("натри, кали, хлорид бага") use the bare names; the medicines use the genitive
CATION_GENITIVES = {"натрийн", "калийн", "магнийн", "кальцийн"}
# the next word shows the body's own substance is meant, not the medicine:
# "инсулины эсэргүүцэл", "адреналин ихсээд", "фолийн хүчлийн дутагдал", "окситоцин даавар"
BLOCK_NEXT = ["эсэргүүц", "хамаар", "даавар", "дутагдал", "түвшин", "хордлого", "ихсэ", "ихэс", "багас", "шинжилгээ"]
# Names that are everyday words in health questions ("цусан дахь глюкоз", "хүчилтөрөгч дутагдах",
# "амин хүчил", "иодын дутагдал") — matching them would turn a symptom question into a medicine-list
# answer. They are matched only by the narrow phrases in NARROW_ALIASES; without one no document is
# written, because it could never be retrieved.
NARROW_ALIASES = {
    "хүчилтөрөгч": ["эмнэлгийн хүчилтөрөгч", "хүчилтөрөгчийн баллон"],
    "глюкоз (декстроз)": ["глюкозын уусмал", "глюкоз дусал", "декстроз"],
    "альбумин": ["альбумин тариа", "альбумин дусал", "альбуминий уусмал"],
    "олон найрлагат давсны бэлдмэл": ["регидрон", "ус давсны уусмал", "уулгаж сэлбэх"],
    "диализийн уусмал": ["диализ уусмал"],
    "тестостерон": ["тестостерон тариа", "тестостерон эм$"],
    "ашигтай микрофлорууд": ["ашигтай микрофлор", "пробиотик"],
    "хлорт нэгдлүүд": ["хлорт нэгдэл", "хлорт ариутгал"],
    "иодын бэлдмэл": ["иодын бэлдмэл", "иодтой эм$", "калийн иодид", "иодид", "иодомарин"],
}
NO_LEXICON = {"иодын бэлдмэл", "хүчилтөрөгч", "глюкоз (декстроз)", "глюкоз (декстроз) + натрийн хлоридын холимог уусмал",
              "альбумин", "амин хүчлийн нийлмэл бэлдмэл", "тосны хүчлийн нийлмэл бэлдмэл",
              "амин хүчлийн нийлмэл бэлдмэл + тосны хүчлийн нийлмэл бэлдмэл + эрдэс бодис",
              "амин хүчил (элэгний дутмагшилд)", "кето амин хүчил (бөөрний дутагдалд)", "диализийн уусмал",
              "тестостерон", "хүний (хэвийн) иммуноглобулин", "хүний иммуноглобулин (хэвийн)", "ашигтай микрофлорууд",
              "эс залгиурууд", "ашигтай микрофлорууд + эс залгиурууд", "олон найрлагат давсны бэлдмэл",
              "хлорт нэгдлүүд", "амин хүчил"}

SALT_WORDS = {"гидрохлорид", "бромид", "сульфат", "фумарат", "фуроат", "моногидрат", "натри", "валериат", "ацетат"}
STOP_WORDS = {"эсрэг", "бэлдмэл", "бэлдмэлүүд", "нийлмэл", "уусмал", "уусал", "холимог", "цэвэршүүлсэн", "хүний",
              "хэвийн", "нүдний", "вакцин", "чөлөөлөгч", "хатааж", "хөлдөөсөн", "төрөл", "нян",
              "иммуноглобулин", "ийлдэс", "залгиур"}
# A disease name alone is NOT the product: "цусан суулгалт" (bloody diarrhoea, a red flag) must not
# become "цусан суулгын нян залгиур"; "хачигт энцефалит гэж юу вэ" must not become the immunoglobulin.
PRODUCT_WORDS = ("вакцин", "иммуноглобулин", "ийлдэс", "залгиур")
VACCINE_SECTIONS = ("19.3.1", "19.3.2")

FORMS = [  # (label, regex) — most specific first; matched text is removed before the next one
    ("нүдний тосон түрхлэг", r"нүдний\s+тосон"),
    ("нүдний дусаалга", r"нүдний\s+дус\w*"),
    ("чихний дусаалга", r"чихний\s+дус\w*"),
    ("хамрын цацлага", r"хамрын\s+цацлага"),
    ("ингаляцийн цацлага", r"ингаляци\w*\s+цацлага"),
    ("ингаляцийн нунтаг", r"ингаляци\w*\s+нунтаг"),
    ("тарилгын нунтаг", r"тарилг\w*\s+нунтаг"),
    ("тарилгын шингэн", r"тарилг\w*\s+шингэн"),
    ("тарилгын уусмал", r"(?:депо-)?тарилг\w*\s+у?ус\w*мал"),   # incl. misprints "тарилгйн", "усумал"
    ("тарилга", r"тарилга"),
    ("дусаах шингэн", r"дуслын\s+шингэн"),
    ("үтрээний шахмал", r"үтрээний\s+шахмал"),
    ("үтрээний капсул", r"үтрээний\s+капсул"),
    ("уусдаг шахмал", r"уусдаг\s+шахмал"),
    ("зөөлөн капсул", r"зөөлөн\s+капсул"),
    ("шахмал", r"шахмал"),
    ("капсул", r"капсул"),
    ("тосон түрхлэг", r"(?:шингэн\s+)?тосон\s+түрхлэг|мазь"),
    ("крем", r"крем"),
    ("гель", r"гель"),
    ("лаа", r"\bлаа\b"),
    ("сироп", r"сироп"),
    ("уух уусмал", r"уух\s+(?:усан\s+)?уусмал|дуслаар\s+ууж\s+хэрэглэх\s+тосон\s+уусмал"),
    ("гадуур хэрэглэх уусмал", r"гадуур\s+хэрэглэх\s+уусмал|хэсэг\s+газарт\s+хэрэглэх"),
    ("хэвлийд сэлбэх уусмал", r"хэвлийд\s+сэлбэх\s+уусмал"),
    ("усан цийдмэг", r"усан\s+цийдмэг"),
    ("хөвмөл бэлтгэх нунтаг", r"хөвмөл\w*\s+(?:бэлтгэх\s+)?нунтаг"),
    ("утлагын хөвмөл", r"утлагад\s+зориулсан\s+хөвмөл"),
    ("хөвмөл", r"хөвмөл\w*"),
    ("цацлага", r"цацлага"),
    ("мөхлөг", r"мөхлөг"),
    ("нунтаг", r"нунтаг"),
    ("суулгац", r"суулгац"),
    ("амьсгалаар хэрэглэх", r"амьсгалаар\s+хэрэглэх"),
    ("уусмал", r"уусмал"),
]

_TRANSLIT = dict(zip("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "yo", "j", "z", "i", "i", "k", "l", "m", "n", "o", "o", "p", "r",
                      "s", "t", "u", "u", "f", "h", "ts", "ch", "sh", "sh", "", "y", "", "e", "yu", "ya"]))


@dataclass
class Medicine:
    key: str
    names: list[str] = field(default_factory=list)      # spellings as printed in the list
    rows: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.names[0]


# ------------------------------------------------------------------ helpers
def fix_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.replace("*", "")).strip()
    for bad, good in NAME_FIXES.items():
        name = name.replace(bad, good)
    return name


def strip_brackets(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)|/[^/]*/", " ", name)).strip()


def name_key(name: str) -> str:
    n = strip_brackets(fix_name(name)).lower()
    n = re.sub(r"\bxr\b", "", n)
    parts = sorted(p for p in (re.sub(r"[\s\-]+", "", x) for x in n.split("+")) if p)
    return fold("+".join(parts))


def display_name(m: Medicine) -> str:
    if m.name in DISPLAY_NAMES:
        return DISPLAY_NAMES[m.name]
    n = re.sub(r"\s*\(Х\)", "", m.name)                       # "(Х)" is an unexplained table mark
    n = re.sub(r"\s*\+\s*", " + ", n).replace(" XR", "")
    n = re.sub(r"\s+-\s+(?=(?:г)?ийн\b|ын\b)", "-", n)          # "гепатит Б - ийн" -> "гепатит Б-ийн"
    if is_vaccine(m) and "вакцин" not in n:
        n = n.replace(" - ", ", ") + " вакцин"
    return n[:1].upper() + n[1:]


def is_vaccine(m: Medicine) -> bool:
    return any(r["section"].startswith(VACCINE_SECTIONS) for r in m.rows)


def heading_text(h: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", h).strip()


def slug(text: str) -> str:
    s = "".join(_TRANSLIT.get(ch, ch) for ch in strip_brackets(text).lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:40].rstrip("_")


GENITIVE = ("гийн", "ийн", "ний", "ын", "ны")
# Mongolian drops a vowel when a suffix is added: бурхан -> бурхны, хавдар -> хавдрын
STEM_FIX = {"бурхан": "бурх", "хавдар": "хавд", "ханиад": "ханиа", "тахал": "тах"}


def word_stem(word: str) -> str | None:
    w = word.lower().strip("-,.")
    if "-" in w:
        w = w.split("-")[-1]              # "n-метилглюкамин" -> "метилглюкамин", "в-ийн" -> "ийн"
    if not w or re.search(r"\d", w) or len(w) == 1 or w in STOP_WORDS or w in GENITIVE:
        return None
    if w.startswith(("хүчил", "хүчл")):
        return "хүч"                      # хүчил / хүчлийн — only used after another stem
    if w in CATION_GENITIVES:
        return w + "$"                    # "натрийн хлорид" yes, lab "натри, кали, хлорид" no
    if w in STEM_FIX:
        return STEM_FIX[w]
    for suf in GENITIVE:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w if len(w) >= 4 else w + "$"   # very short words must match exactly ("эс" != "эсвэл")


def component_stems(comp: str) -> list[str]:
    words = comp.split()
    if len(words) > 1 and len(words[0]) >= 8 and words[-1].lower() in SALT_WORDS | {"натрийн"}:
        words = words[:-1]                # "тербуталин сульфат" -> "тербуталин"; keep "магнийн сульфат"
    stems = [s for s in (word_stem(w) for w in words) if s]
    return stems[:3]                      # "азотын дутуу исэл": "азотын дутуу" alone is everyday speech


def drug_lexicon_aliases() -> dict[str, list[str]]:
    """Spellings the ADR drug lexicon already knows ("амоксицилин"). A word known to one lexicon
    is never typo-matched by another, so the EML concept must list them too."""
    path = LEX_OUT.parent / "generated_drug_lexicon.yaml"
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    for spec in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("concepts", {}).values():
        pats = [p for p in spec.get("mn", []) if not re.search(r"[\d\-]", p)]
        out.setdefault(name_key(spec.get("label_mn", "")), []).extend(pats)
    return out


DRUG_ALIASES: dict[str, list[str]] = {}


def patterns_for(m: Medicine) -> list[str]:
    pats: list[str] = []
    overrides = [p for nm in m.names for p in PATTERN_OVERRIDES.get(nm, [])]
    if overrides:
        return list(dict.fromkeys(p.lower() for p in overrides))
    for nm in m.names:
        base = strip_brackets(fix_name(nm)).replace(" XR", "")
        comps = [c.strip() for c in re.split(r"\+|,|\s-\s", base) if c.strip()]
        stems = [s for c in comps[:3] for s in component_stems(c)]   # long mixtures: first 3 parts
        if not stems:
            continue
        # the product word is required (see PRODUCT_WORDS)
        product = "вакцин" if is_vaccine(m) else next((w for w in PRODUCT_WORDS if w in base.lower()), None)
        if product:
            stems = [s for s in stems if s != product][:2] + [product]
            if stems[0] == "улаан":                            # "улаанбурхан" is usually one word
                pats.append("".join(s.rstrip("$") for s in stems[:2]) + f" {product}")
        pats.append(" ".join(stems))
    for nm in m.names:
        pats.extend(ALIASES.get(fix_name(nm), []))
    pats.extend(DRUG_ALIASES.get(m.key, []))
    return list(dict.fromkeys(p.lower() for p in pats))


def forms_of(m: Medicine) -> list[str]:
    return forms_of_rows(m.rows)


def forms_of_rows(rows: list[dict]) -> list[str]:
    found: list[str] = []
    for r in rows:
        s = re.sub(r"\([^)]*\)", " ", r["form_strength"].lower())
        for label, rx in FORMS:
            if re.search(rx, s):
                if label not in found:
                    found.append(label)
                s = re.sub(rx, " ", s)
    if "тарилга" in found and any(f.startswith("тарилгын") for f in found):
        found.remove("тарилга")
    return found


def join_mn(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " болон " + items[-1]


# ------------------------------------------------------------------ kind of product
HOSPITAL_FORMS = {"тарилгын уусмал", "тарилгын нунтаг", "тарилгын шингэн", "тарилга", "дусаах шингэн",
                  "амьсгалаар хэрэглэх", "хэвлийд сэлбэх уусмал"}
HOSPITAL_SECTIONS = ("14.", "19.1", "23.", "11.")          # contrast media, diagnostics, dialysis, plasma


def kind_of(m: Medicine, marks: list[str], age_note: str | None) -> str:
    """medicine (taken home) | hospital (given by staff) | vaccine | disinfectant (surfaces)."""
    sections = [r["section"] for r in m.rows]
    if is_vaccine(m):
        return "vaccine"
    if all(s.startswith("15.2") for s in sections):
        return "disinfectant"
    if age_note and "эмнэлгийн нөхцөлд" in age_note:
        return "hospital"
    # dispensed by a pharmacy -> people take it home (insulin is an injection, but used at home)
    if not marks and (set(forms_of(m)) <= HOSPITAL_FORMS or all(s.startswith(HOSPITAL_SECTIONS) for s in sections)):
        return "hospital"
    return "medicine"


# ------------------------------------------------------------------ text
NOT_REQUIRED = "Нийтийн үйлчилгээтэй эмийн санд заавал байлгах эмийн тоонд ороогүй"


def dispensing_sentence(m: Medicine, marks: list[str], kind: str) -> str:
    if not marks:
        tail = {"hospital": "; эмнэлгийн нөхцөлд хэрэглэдэг бэлдмэл.",
                "vaccine": "; вакциныг эрүүл мэндийн байгууллагад хийдэг.",
                "disinfectant": "; эмнэлэг, байгууллагад ариутгалд хэрэглэдэг бодис."}.get(
            kind, " — хаанаас авах талаар эмч, эм зүйчээсээ тодруулаарай.")
        return NOT_REQUIRED + tail
    if marks == ["Ж*"]:
        s = "Зөвхөн амбулаторийн үйлчлүүлэгчдэд үйлчилдэг эмийн санд жороор олгоно (Ж*)."
    elif len(marks) == 1:
        s = f"Нийтийн үйлчилгээтэй эмийн санд {MARK_SHORT[marks[0]]} олгоно ({marks[0]})."
    elif "Ж*" in marks:
        others = [MARK_SHORT[x] for x in marks if x != "Ж*"]
        s = ("Хэлбэр, тун хэмжээнээсээ хамааран заримыг нь зөвхөн амбулаторийн үйлчлүүлэгчдэд үйлчилдэг эмийн санд "
             f"жороор, бусдыг нь нийтийн үйлчилгээтэй эмийн санд {join_mn(others)} олгоно ({'/'.join(marks)}).")
    else:
        s = (f"Нийтийн үйлчилгээтэй эмийн санд хэлбэр, тун хэмжээнээсээ хамааран "
             f"{join_mn([MARK_SHORT[x] for x in marks])} олгоно ({'/'.join(marks)}).")
    blank = [r for r in m.rows if not r["public_pharmacy_mark"]]
    if blank:
        if set(forms_of_rows(blank)) <= HOSPITAL_FORMS:
            s += " Тарилга зэрэг зарим хэлбэрийг эмнэлгийн нөхцөлд хэрэглэдэг."
        else:
            s += " Зарим хэлбэр нь нийтийн эмийн санд заавал байлгах тоонд ороогүй."
    return s


def groups_sentence(m: Medicine) -> str:
    by_chapter: "OrderedDict[str, list[str]]" = OrderedDict()
    for r in m.rows:
        ch, sec = heading_text(r["chapter"]), heading_text(r["section"])
        secs = by_chapter.setdefault(ch, [])
        if sec and sec != ch and sec not in secs:
            secs.append(sec)
    parts = [f"«{ch}» — {', '.join(f'«{s}»' for s in secs)}" if secs else f"«{ch}»" for ch, secs in by_chapter.items()]
    return "Жагсаалтын бүлэг, хэсэг: " + "; ".join(parts) + "."


ALLERGY_SIGNS = ["Уруул, нүүр, хоолой хавдах", "Амьсгал давчдах, хоолой бачуурах", "Толгой эргэж ухаан балартах",
                 "Арьс хуурч цэврүүтэх, бүх биеэр тууралт гарах"]
REPORT_ADR = "Гаж нөлөө илэрвэл эмнэлэг, эмийн сангаар дамжуулан эмийн хяналтын байгууллагад мэдээлж болно"


def advice(kind: str, marks: list[str], age: str | None) -> tuple[list[str], list[str], str, list[str]]:
    """(care bullets, warning bullets, warning tail, when-to-see-a-doctor bullets) for each kind of product."""
    tail = "Эдгээр шинж илэрвэл даруй 103 дугаарт залгаарай."
    if kind == "vaccine":
        return ([
            "Вакциныг эрүүл мэндийн байгууллагын дархлаажуулалтын кабинетад эрүүл мэндийн ажилтан хийнэ",
            "Өөрийн болон хүүхдийнхээ дархлаажуулалтын хуваарийг өрхийн эмчээсээ тодруулах",
            "Өмнөх вакцинд хүчтэй урвал өгч байсан, харшилтай, жирэмсэн эсвэл дархлаа дарангуйлах эмчилгээ "
            "хийлгэж буй бол вакцин хийлгэхийн өмнө эмчид хэлэх",
            "Вакцин хийлгэсний дараа эрүүл мэндийн ажилтны зөвлөснөөр хэсэг хугацаанд эмнэлэгт ажиглуулах",
        ], ALLERGY_SIGNS + ["Таталт өгөх, ухаан алдах"], tail, [
            "Вакцин хийлгэсний дараа халуурах, тарьсан газар хавдаж улайх нь удаан үргэлжлэх эсвэл улам дордвол",
            "Хүүхэд тайван бус уйлах, хооллохгүй байх нь үргэлжилбэл",
            "Дархлаажуулалтын хуваарь, хоцорсон вакцины талаар мэдэхийг хүсвэл өрхийн эмчдээ",
            "Вакцины дараах урвалыг эрүүл мэндийн байгууллагад мэдээлж болно",
        ])
    if kind == "disinfectant":
        return ([
            "Энэ бодисыг гадаргуу, багаж ариутгахад хэрэглэдэг — уух, залгих, арьсанд түрхэхгүй",
            "Савлагаан дээрх заавар, хамгаалах хэрэгслийн (бээлий) шаардлагыг дагах",
            "Хүүхэд хүрэхгүй, хүнсний зүйлээс тусад нь хадгалах",
        ], ["Санамсаргүй залгисан", "Уураар амьсгалаад амьсгал давчдах, хоолой бачуурах",
            "Нүдэнд орж хүчтэй хорсох, хараа бүдгэрэх"],
            "Эдгээр тохиолдолд даруй 103 дугаарт залгаарай.", [
            "Арьс, нүдэнд хүрсний дараа улайлт, хорсол арилахгүй бол",
            "Энэ бодисыг аюулгүй хэрэглэх талаар асуух зүйл байвал эмч, эм зүйчдээ",
        ])
    if kind == "hospital":
        return ([
            "Энэ бэлдмэлийг эмнэлгийн нөхцөлд эрүүл мэндийн ажилтан хэрэглэнэ — гэртээ өөрөө хэрэглэдэг эм биш",
            "Харшил, жирэмсэн эсвэл хөхүүл эсэх, хэрэглэж буй эмээ эмч, сувилагчдаа заавал хэлэх",
            "Өмнө нь ийм бэлдмэлд урвал өгч байсан бол эмнэлэгт очих бүртээ хэлэх",
            "Тун, хэрэглэх аргыг зөвхөн эмч тогтооно",
        ], ALLERGY_SIGNS, tail, [
            "Эмнэлгээс гарсны дараа тууралт, загатнаа, амьсгал давчдах зэрэг шинж илэрвэл",
            "Шинж тэмдэг сайжрахгүй эсвэл улам дордвол",
            "Энэ бэлдмэлийн талаар асуух зүйл байвал эмчлэгч эмчдээ",
            REPORT_ADR,
        ])
    care = ["Эмийг зөвхөн эмч, эм зүйчийн зааврын дагуу, савлагаан дээрх зааврыг уншиж хэрэглэх"]
    if set(marks) & RX_MARKS:
        care.append("Жороор олгодог эмийг эмчийн жоргүйгээр, өөрийн дураар авч хэрэглэхгүй байх")
    elif marks:
        care.append("Жоргүй олгодог эмийг ч гэсэн эм зүйчээс зөвлөгөө авч хэрэглэх")
    else:
        care.append("Энэ эмийг эмчийн томилгоо, хяналтаар хэрэглэх")
    if set(marks) & {"СЖ", "МЖ"}:
        care.append("Сэтгэцэд нөлөөт болон мансууруулах эмийг бусдад дамжуулж болохгүй — зөвхөн жор бичигдсэн хүн хэрэглэнэ")
    care.append("Хэрэглэж буй бусад эм, харшил, жирэмсэн эсвэл хөхүүл эсэхээ эмч, эм зүйчдээ заавал хэлэх")
    if age and re.search(r"нас|сар|нярай|долоо хоног", age):
        care.append("Хүүхэд, нярайд хэрэглэхийн өмнө насны хязгаарыг эмч, эм зүйчээс заавал тодруулах")
    care.append("Эмийн тун, хэрэглэх хугацааг зөвхөн эмч, эм зүйч тогтооно")
    care.append("Эмийг хүүхэд хүрэхгүй газар хадгалах")
    return (care, ALLERGY_SIGNS + ["Хүүхэд санамсаргүй залгисан эсвэл хэт их хэмжээгээр хэрэглэсэн"], tail, [
        "Эм хэрэглэсний дараа тууралт, загатнаа, дотор муухайрах зэрэг шинж илэрвэл",
        "Шинж тэмдэг сайжрахгүй эсвэл улам дордвол",
        "Энэ эм танд тохирох эсэх, жор шаардлагатай эсэхийг мэдэхийг хүсвэл өрхийн эмч эсвэл эм зүйчдээ",
        REPORT_ADR,
    ])


def age_note_for(m: Medicine) -> str | None:
    return next((AGE_NOTES[k] for k in AGE_NOTES if name_key(k) == m.key), None)


def build_doc(m: Medicine, doc_id: str, concept: str | None) -> str:
    name = display_name(m)
    marks = list(dict.fromkeys(r["public_pharmacy_mark"] for r in m.rows if r["public_pharmacy_mark"]))
    levels = [label for col, label in LEVELS.items() if any(r[col] == "+" for r in m.rows)]
    forms = forms_of(m)
    age = age_note_for(m)
    kind = kind_of(m, marks, age)
    first_chapter = heading_text(m.rows[0]["chapter"])

    info = [f"{name} нь Монгол Улсын Зайлшгүй шаардлагатай эмийн IX жагсаалтын (Эрүүл мэндийн сайдын 2020 оны "
            f"А/625 тушаал) «{first_chapter}» бүлэгт багтдаг.",
            dispensing_sentence(m, marks, kind)]
    if age:
        info.append(f"Жагсаалтын анхааруулга: {age}.")
    if forms:
        info.append(f"Жагсаалтад орсон хэлбэр: {', '.join(forms)}.")
    info.append(groups_sentence(m))
    if levels:
        info.append(f"Заавал байх ёстой шатлал: {', '.join(levels)}.")
    info.append("Энэ нь эмийн жагсаалтын мэдээлэл бөгөөд хэрэглэх заавар биш. Жагсаалт шинэчлэгдсэн байж "
                "болзошгүй тул одоогийн нөхцөлийг эм зүйч эсвэл эмчээс тодруулаарай.")
    care, warn, warn_tail, when = advice(kind, marks, age)

    items = ", ".join(dict.fromkeys(r["no"] for r in m.rows))
    pages = ", ".join(dict.fromkeys(r["pdf_page"] for r in m.rows))
    meta = {
        "id": doc_id,
        "title": f"{name} — зайлшгүй шаардлагатай эмийн жагсаалт",
        "category": "medication",
        "concepts": [concept] if concept else [],
        "retrieval": "concept",
        "source": SOURCE,
        "source_tier": 1,
        "source_url": SOURCE_URL,
        "published_date": "2020-12-25",
        "last_reviewed": None,
        "reviewer": None,
        "version": "1.0",
        "language": "mn",
        "status": "REVIEW",
        "notes": (f"AUTO-GENERATED by scripts/build_eml_kb.py from list item(s) {items} (PDF p. {pages}); "
                  f"kind: {kind}. Strengths/doses deliberately omitted. Requires pharmacist review."),
    }
    front = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000)
    body = [
        "## Ерөнхий мэдээлэл", " ".join(info), "",
        "## Гэртээ юу хийж болох вэ", *[f"- {c}" for c in care], "",
        "## Анхаарах шинж", *[f"- {w}" for w in warn], warn_tail, "",
        "## Эмчид хэзээ хандах вэ", *[f"- {w}" for w in when],
    ]
    return f"---\n{front}---\n" + "\n".join(body) + "\n"


# ------------------------------------------------------------------ main
def load_medicines() -> list[Medicine]:
    rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8-sig")))
    meds: "OrderedDict[str, Medicine]" = OrderedDict()
    for r in rows:
        if not r["form_strength"]:
            continue                      # blood components (№ 211-214) are not medicines people ask about
        nm = fix_name(r["inn_mn"])
        k = name_key(nm)
        # same medicine, misprinted ("азатиопирин" / "азатиоприн", "ацетозоламид" / "ацетазоламид")
        near = next((mk for mk in meds if len(k) >= 8 and "+" not in k and "+" not in mk
                     and levenshtein(k, mk, 1) <= 1), None)
        m = meds.setdefault(near or k, Medicine(key=near or k))
        if nm not in m.names:
            m.names.append(nm)
        m.rows.append(r)
    return list(meds.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    DRUG_ALIASES.update(drug_lexicon_aliases())
    meds = load_medicines()
    KB_OUT.mkdir(parents=True, exist_ok=True)
    for old in KB_OUT.glob("MED_EML_*.md"):   # files, not the folder: OneDrive/Explorer may hold the folder open
        old.unlink()
    lexicon: dict[str, dict] = {}
    used_ids: set[str] = set()
    unused_age = {name_key(k): k for k in AGE_NOTES}
    skipped = []
    for m in meds:
        no = int(m.rows[0]["no"])
        doc_id, n = f"MED_EML_{no:03d}", 2
        while doc_id in used_ids:
            doc_id, n = f"MED_EML_{no:03d}_{n}", n + 1
        used_ids.add(doc_id)
        unused_age.pop(m.key, None)
        if any(x in NO_LEXICON for x in m.names):
            pats = [p for x in m.names for p in NARROW_ALIASES.get(x, [])]
        else:
            pats = patterns_for(m)
        if not pats:                     # nothing could ever retrieve it (retrieval: concept)
            skipped.append(display_name(m))
            continue
        concept = f"eml_{slug(display_name(m))}"
        while concept in lexicon:
            concept += "_x"
        lexicon[concept] = {"label_mn": display_name(m), "type": "medication", "mn": list(dict.fromkeys(pats))}
        fname = f"{doc_id}_{re.sub(r'[^0-9a-zа-яёөү]+', '_', display_name(m).lower()).strip('_')[:40]}.md"
        (KB_OUT / fname).write_text(build_doc(m, doc_id, concept), encoding="utf-8")
    if unused_age:
        raise SystemExit(f"AGE_NOTES keys that match no medicine (spelling?): {sorted(unused_age.values())}")

    header = ("# AUTO-GENERATED by scripts/build_eml_kb.py from the Essential Medicines List — do not edit by hand\n"
              "# fuzzy_min_len: a typo is forgiven only in long names (\"морфин\" must not match \"морин\")\n"
              "# block_next: \"инсулины эсэргүүцэл\", \"адреналин ихсээд\" are about the body, not the medicine\n")
    LEX_OUT.write_text(header + yaml.safe_dump({"version": "generated", "fuzzy_min_len": 8, "block_next": BLOCK_NEXT,
                                                "concepts": lexicon},
                                               allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    merged = [m.names for m in meds if len(m.names) > 1]
    print(f"{len(meds)} medicines; {len(lexicon)} documents -> {KB_OUT.relative_to(ROOT)} and lexicon concepts -> "
          f"{LEX_OUT.relative_to(ROOT)}")
    print(f"no document (no safe name to match): {skipped}")
    print("merged spellings:", merged)


if __name__ == "__main__":
    main()
