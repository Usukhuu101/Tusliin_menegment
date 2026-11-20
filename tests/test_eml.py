"""Essential Medicines List (ЗШЭ IX жагсаалт, А/625) -> lexicon + one KB document per medicine."""
from __future__ import annotations

import csv

import pytest

from backend.agents.safety_agent import _DIAGNOSIS, _DISCOURAGE, _DOSE, _STOP_MED
from backend.agents.symptom_agent import SymptomAgent
from backend.config import settings
from backend.models.schemas import SymptomAnalysis
from backend.nlp.lexicon import ConceptLexicon
from backend.rag.kb_loader import parse_document
from backend.services.llm import LLMError, LLMProvider

EML_DIR = settings.kb_dir / "medications" / "eml"
EML_CSV = settings.kb_dir.parent / "cleaned" / "eml_mn_2020.csv"


def eml_concepts(processor, text):
    return {c for c in processor.process(text).positive_concepts if c.startswith("eml_")}


# ------------------------------------------------------------------ data
def test_csv_has_the_whole_list():
    rows = list(csv.DictReader(EML_CSV.open(encoding="utf-8-sig")))
    assert len(rows) >= 650 and len({r["no"] for r in rows}) >= 560
    assert {r["public_pharmacy_mark"] for r in rows} <= {"", "Ж", "ЖГ", "Ж*", "СЖ", "МЖ"}
    assert all(r["chapter"] and r["section"] for r in rows)
    assert not any("кeтамин" in r["inn_mn"] for r in rows)          # Latin "e" fixed


def test_every_document_is_valid_and_never_states_a_dose():
    paths = sorted(EML_DIR.glob("MED_EML_*.md"))
    assert len(paths) >= 450
    for p in paths:
        doc = parse_document(p)
        assert not doc.warnings, (p.name, doc.warnings)
        assert doc.meta.retrieval == "concept" and doc.meta.status == "REVIEW" and doc.meta.source_tier == 1
        for rx in (_DOSE, _STOP_MED, _DIAGNOSIS, _DISCOURAGE):
            m = rx.search(doc.full_text)
            assert m is None, (p.name, m.group(0))


def test_ibuprofen_document_states_prescription_status_and_age_note():
    doc = parse_document(next(EML_DIR.glob("MED_EML_019_*.md")))
    info = dict(doc.sections)["Ерөнхий мэдээлэл"]
    assert "жороор" in info and "жоргүй" in info and "3 сараас дээш" in info


# ------------------------------------------------------------------ lexicon
@pytest.mark.parametrize("text,concept", [
    ("Ибупрофен жороор олгодог уу", "eml_ibuprofen"), ("ibuprofen", "eml_ibuprofen"),
    ("ибупрафен", "eml_ibuprofen"),                                   # one typo in a long name
    ("Парацетамолоор халуун буулгаж болох уу", "eml_paratsetamol"),
    ("Аспирин жоргүй авч болох уу", "eml_atsetilsalitsilyn_huchil"),
    ("Но-шпа", "eml_drotaverin"), ("Метформин", "eml_metformin"), ("Глюкофаж", "eml_metformin"),
    ("Улаанбурхны вакцин", "eml_ulaanburhny_vaktsin"),
    ("амоксицилин", "eml_amoksitsillin"),          # spelling known to the ADR lexicon is shared
    ("Физиологийн уусмал", "eml_natriin_hlorid"), ("Инсулин тарьдаг", "eml_insulin"),
])
def test_medicine_names_are_recognised(processor, text, concept):
    assert concept in eml_concepts(processor, text)


@pytest.mark.parametrize("text,concept", [
    ("гепатит А вакцин", "eml_gepatit_a_giin_vaktsin"), ("В гепатитын вакцин", "eml_gepatit_b_iin_vaktsin"),
])
def test_hepatitis_a_and_b_vaccines_are_told_apart(processor, text, concept):
    assert eml_concepts(processor, text) == {concept}


@pytest.mark.parametrize("text", [
    "Морин хуур", "Цусан суулгалт гараад байна", "Цусан дахь глюкоз ихэссэн", "Хүчилтөрөгч дутагдаад байна",
    "Сүрьеэ гэж юу вэ", "Томуу туссан", "Улаанбурхан гэж юу вэ", "Витамин Д", "Арьс цайраад байна",
    "Давс ихтэй хоол", "Толгой өвдөөд байна", "Амин хүчил хаанаас авах вэ",
    # found by the lexicon audit (667 messages without a medicine name)
    "Панкреатит өвчин хэр аюултай вэ", "Иодын дутагдал гэж юу вэ", "Хачигт энцефалит гэж юу вэ",
    "Цусан шинжилгээнд натри, кали, хлорид бага", "Азотын хэмжээ дутуу гарсан",
    "Цусан суулгалт гараад залгиур өвдөж байна", "Хорионы гонадотропин даавар ихэссэн", "Диоксины хордлого",
    "Колитын шинж тэмдэг", "Глицерин гарт түрхэж болох уу", "Гастроинтестинал цус алдалт",
    "Томуу туссан, вакцин хийлгээгүй", "Хүүхэд маань улаанбурхан туссан, вакцин хийлгэсэн байсан",
    "Боомт хотод вакцин хийлгэсэн", "Инсулины эсэргүүцэл гэж юу вэ", "Адреналин ихсээд зүрх дэлсээд байна",
    "Фолийн хүчлийн дутагдал", "Окситоцин даавар",
])
def test_no_medicine_concept_without_a_medicine_name(processor, text):
    assert eml_concepts(processor, text) == set()


def test_generated_lexicon_typo_tolerance_is_limited_to_long_names(tmp_path):
    main = tmp_path / "main.yaml"
    main.write_text("version: t\nconcepts: {}\n", encoding="utf-8")
    gen = tmp_path / "gen.yaml"
    gen.write_text("fuzzy_min_len: 8\nconcepts:\n  m: {label_mn: морфин, type: medication, mn: [морфин]}\n"
                   "  p: {label_mn: парацетамол, type: medication, mn: [парацетамол]}\n", encoding="utf-8")
    lex = ConceptLexicon([main, gen])
    found = lambda t: {m.concept for m in lex.match(t)}  # noqa: E731
    assert found("морин хуур") == set()                  # 6-letter name: exact prefix only
    assert found("морфинтэй") == {"m"}
    assert found("парацетомол") == {"p"}                 # 11-letter name: one typo forgiven


# ------------------------------------------------------------------ pipeline
@pytest.mark.parametrize("msg,doc", [
    ("Ибупрофен жороор олгодог уу", "MED_EML_019"), ("Метформин", "MED_EML_367"),
    ("paracetamol", "MED_EML_024"), ("Омепразол ямар эм вэ", "MED_EML_328"),
])
def test_medicine_questions_are_answered_from_the_list(orchestrator, msg, doc):
    r = orchestrator.handle(msg)
    assert r.message_type == "ANSWER" and r.sources[0].doc_id == doc
    assert "103" in r.reply and "ℹ️" in r.reply


def test_answer_never_cites_a_different_medicine(orchestrator):
    # used to add the cefotaxime side-effect summary as the "second" document
    r = orchestrator.handle("Ибупрофен жороор олгодог уу")
    assert [s.doc_id for s in r.sources] == ["MED_EML_019", "MED_MED_001"]


@pytest.mark.parametrize("msg", ["Сүрьеэ гэж юу вэ", "Бамбай булчирхайн өвчин гэж юу вэ"])
def test_list_entries_do_not_answer_disease_questions(orchestrator, msg):
    # "Сүрьеэ эмчлэх эм" / "Бамбай булчирхайн даавар" are group names inside list entries;
    # those entries are concept-only, so the honest fallback stays.
    r = orchestrator.handle(msg)
    assert r.message_type in ("FALLBACK", "CLARIFY") and not r.sources


class _CapturePrompt(LLMProvider):
    name, available = "capture", True

    def __init__(self):
        self.prompt = ""

    def generate(self, **kw):
        raise LLMError("not used")

    def generate_structured(self, *, messages, **kw):
        self.prompt = messages[0]["content"]
        raise LLMError("captured")


def test_llm_extraction_prompt_does_not_list_500_medicine_names(lexicon, processor):
    llm = _CapturePrompt()
    agent = SymptomAgent(lexicon, settings.protocols_dir / "symptom_protocols.yaml", llm, "m",
                         settings.prompts_dir / "symptom_extraction.md")
    agent._llm_enrich(processor.process("толгой өвдөөд байна"), SymptomAnalysis(intent="symptom_check"))
    assert "- headache:" in llm.prompt and "eml_" not in llm.prompt
