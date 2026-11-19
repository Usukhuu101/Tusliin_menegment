"""Mongolian NLP: script handling, typos, negation, facts, PII."""
import pytest

from backend.nlp.input_processor import redact_pii
from backend.nlp.text_utils import fold, levenshtein, transliterate_latin


def concepts(processor, text):
    return set(processor.process(text).positive_concepts)


@pytest.mark.parametrize("text", [
    "толгой өвдөөд байна", "tolgoi owdood bna", "tolgoi ovgood bn", "headache болоод байна", "тологй өвдөөд байна",
])
def test_headache_variants_map_to_one_concept(processor, text):
    assert "headache" in concepts(processor, text)


def test_fold_and_transliteration():
    assert fold("Өвдөөд") == "овдоод"
    assert fold(transliterate_latin("owdood")) == "овдоод"
    assert transliterate_latin("bna") == "байна"


def test_swap_counts_as_one_typo():
    assert levenshtein("тологи", "толгои") == 1


def test_negation(processor):
    n = processor.process("Толгой өвдөж байна, халуураагүй")
    assert "headache" in n.positive_concepts
    assert "fever" in n.negated_concepts


def test_crisis_language_is_never_negated(processor):
    assert "suicidal" in concepts(processor, "амьдрах хүсэл алга")


@pytest.mark.parametrize("text,absent", [
    ("Маргааш цаг агаар ямар байх вэ", "co_exposure"),      # агаар (air) != угаар (CO)
    ("Халуун цай асгаад түлэгдсэн", "fever"),               # халуун (hot) != халуурах (fever)
    ("Эмч рүү очих уу", "medication"),                      # эмч (doctor) != эм (medicine)
])
def test_no_false_positive(processor, text, absent):
    assert absent not in concepts(processor, text)


def test_facts(processor):
    f = processor.process("Өчигдрөөс толгой өвдөөд байна, 7/10, 45 настай, даралт 150/95, 38.5 хэм").facts
    assert f.duration_days == 1 and f.severity == 7 and f.age_years == 45
    assert (f.bp_systolic, f.bp_diastolic) == (150, 95)
    assert f.temperature_c == 38.5


def test_duration_words(processor):
    assert processor.process("2 долоо хоногоос ханиалгаж байна").facts.duration_days == 14
    assert processor.process("хоёр хоног халуурч байна").facts.duration_days == 2
    assert processor.process("3 сартай хүүхэд").facts.age_months == 3


def test_pii_redaction():
    text, kinds = redact_pii("Миний утас 99112233, РД УБ99112233, mail a@b.mn")
    assert "99112233" not in text and "a@b.mn" not in text
    assert set(kinds) >= {"phone", "national_id", "email"}
