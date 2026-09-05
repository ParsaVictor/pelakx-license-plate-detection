"""Grammar-layer tests — these run with zero ML dependencies installed."""

from __future__ import annotations

import pytest

from pelakx.grammar import identify_country, parse, registry
from pelakx.grammar.normalize import fold_digits, normalize, tokenize
from pelakx.grammar.spec import CountrySpec, GrammarError, Layout


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_loads_shipped_countries():
    codes = registry.codes()
    assert "IR" in codes and "GB" in codes and "US" in codes
    assert registry.load_errors() == []


def test_registry_skips_template():
    assert "XX" not in registry.codes()


def test_unknown_country_raises():
    with pytest.raises(KeyError):
        registry.get("ZZ")


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("۱۲۳۴۵۶۷۸۹۰", "1234567890"), ("٤٥٦", "456"), ("123", "123")],
)
def test_fold_digits(raw, expected):
    assert fold_digits(raw) == expected


def test_normalize_strips_separators_and_uppercases():
    assert normalize("ab-12 cd.e", script="latin") == "AB12CDE"


def test_normalize_folds_arabic_variants():
    assert normalize("ك ي", script="arabic") == "کی"


def test_normalize_strips_noise_words():
    out = normalize("12ب345ایران11", script="arabic", noise_words=("ایران",))
    assert out == "12ب34511"


def test_normalize_ignores_single_char_noise_words():
    # "D" is a real German city code, so a 1-char noise word must be ignored.
    assert normalize("DAB123", script="latin", noise_words=("D",)) == "DAB123"


def test_tokenize_matches_multichar_letters_longest_first():
    assert tokenize("12الف345", ["ا", "الف", "ب"]) == ["1", "2", "الف", "3", "4", "5"]


# ---------------------------------------------------------------------------
# Iranian plates
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ir() -> CountrySpec:
    return registry.get("IR")


def test_iran_clean_plate(ir):
    read = parse("12ب34511", ir, ocr_confidence=0.9)
    assert read is not None
    assert read.canonical == "12ب34511"
    assert read.layout_id == "civilian"
    assert read.fields == {"left": "12", "letter": "ب", "right": "345", "province": "11"}
    assert read.repairs == 0
    assert read.valid


def test_iran_persian_digits_and_country_word(ir):
    read = parse("۱۲ ب ۳۴۵ ایران ۱۱", ir, ocr_confidence=0.9)
    assert read.canonical == "12ب34511"
    assert read.repairs == 0


def test_iran_confusion_repair(ir):
    """A digit slot that received a Latin 'I' should be repaired to '1'."""
    clean = parse("12ب34511", ir, ocr_confidence=0.9)
    repaired = parse("12ب345I1", ir, ocr_confidence=0.9)
    assert repaired.canonical == clean.canonical
    assert repaired.repairs == 1
    assert repaired.confidence < clean.confidence  # repairs must cost something


def test_iran_multichar_letter(ir):
    read = parse("12الف34511", ir, ocr_confidence=0.9)
    assert read.fields["letter"] == "الف"
    assert ir.describe_letter("الف", "en") == "Government"


def test_iran_motorcycle_layout(ir):
    read = parse("88812345", ir, ocr_confidence=0.9)
    assert read.layout_id == "motorcycle"
    assert read.fields == {"province": "888", "serial": "12345"}


def test_iran_province_validator_rejects_out_of_range(ir):
    """Province 09 is below the legal range, so the reading is flagged."""
    good = parse("12ب34555", ir, ocr_confidence=0.9)
    bad = parse("12ب34509", ir, ocr_confidence=0.9)
    assert good.valid and good.fields["province"] == "55"
    assert bad.confidence < good.confidence


def test_unparseable_text_is_surfaced_not_dropped(ir):
    """Text that survives folding but matches no layout is still reported."""
    read = parse("ZZZZ", ir, ocr_confidence=0.9)
    assert read is not None and read.valid is False and read.layout_id == ""
    assert read.confidence < 0.9  # penalised, but visible to the operator


def test_strict_mode_drops_unparseable(ir):
    assert parse("ZZZZ", ir, ocr_confidence=0.9, strict=True) is None


def test_text_that_folds_to_nothing_returns_none(ir):
    for junk in ("", "   ", "!!!!", "---"):
        assert parse(junk, ir) is None


# ---------------------------------------------------------------------------
# Latin-script countries
# ---------------------------------------------------------------------------
def test_uk_current_format():
    read = parse("AB12CDE", registry.get("GB"), ocr_confidence=0.9)
    assert read.layout_id == "current"
    assert read.display == "AB12 CDE"


def test_uk_repairs_digit_in_letter_slot():
    read = parse("A81ZCDE", registry.get("GB"), ocr_confidence=0.9)
    assert read.canonical == "AB12CDE"
    assert read.repairs == 2


def test_spain_rejects_vowel_in_letter_block():
    """Spanish plates use consonants only, so 'A' cannot be a letter token."""
    read = parse("1234BCA", registry.get("ES"), ocr_confidence=0.9)
    assert read.valid is False


def test_germany_variable_length_regex():
    read = parse("B-MW 1234", registry.get("DE"), ocr_confidence=0.9)
    assert read.fields["city"] == "B"
    assert read.fields["digits"] == "1234"


def test_brazil_prefers_mercosul_over_legacy():
    assert parse("ABC1D23", registry.get("BR"), ocr_confidence=0.9).layout_id == "mercosul"
    assert parse("ABC1234", registry.get("BR"), ocr_confidence=0.9).layout_id == "legacy"


# ---------------------------------------------------------------------------
# auto country detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [("AB12CDE", "GB"), ("1234BCD", "ES"), ("12ب34511", "IR"), ("AA-123-BB", "FR")],
)
def test_identify_country(text, expected):
    read = identify_country(text, registry.all_specs(), ocr_confidence=0.92)
    assert read is not None and read.country == expected


# ---------------------------------------------------------------------------
# spec validation
# ---------------------------------------------------------------------------
def test_layout_requires_slots_or_regex():
    with pytest.raises(GrammarError):
        Layout.from_dict({"id": "broken"})


def test_layout_rejects_unknown_slot_char():
    with pytest.raises(GrammarError):
        Layout.from_dict({"id": "broken", "slots": "DDXD"})


def test_layout_rejects_group_outside_slots():
    with pytest.raises(GrammarError):
        Layout.from_dict({"id": "broken", "slots": "DDD", "groups": {"a": [0, 9]}})


def test_spec_requires_at_least_one_layout():
    with pytest.raises(GrammarError):
        CountrySpec.from_dict({"code": "ZZ", "name_en": "Nowhere", "layouts": []})


def test_every_shipped_country_parses_a_sample_of_its_own_layouts():
    """Round-trip: render each fixed layout's slots into a plate and re-parse it."""
    for spec in registry.all_specs():
        for layout in spec.layouts:
            if not layout.is_fixed:
                continue
            sample = "".join(
                spec.digits[i % len(spec.digits)] if slot == "D" else spec.letters[0]
                for i, slot in enumerate(layout.slots)
            )
            read = parse(sample, spec, ocr_confidence=0.9, strict=True)
            assert read is not None, (
                f"{spec.code}/{layout.id} could not parse its own sample {sample!r}"
            )
