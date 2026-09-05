"""Script-aware text normalization.

OCR engines return text in whatever script/codepoints they were trained on.
Before a plate string can be matched against a country's grammar it has to be
folded into one canonical form:

* Persian / Arabic-Indic digits  ->  ASCII digits
* Arabic letter variants         ->  their Persian equivalents (ي -> ی, ك -> ک)
* Zero-width joiners, tatweel, diacritics -> dropped
* Separators / plate furniture   -> dropped
* Latin text                     -> uppercased

The tokenizer then splits the folded string into *plate tokens*, where a token
is either a single character or one of the country's multi-character letters
(``الف``, ``معلولین``, ``تشریفات`` …), matched longest-first.
"""

from __future__ import annotations

import re
import unicodedata

# --- digit folding ---------------------------------------------------------
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DEVANAGARI_DIGITS = "०१२३४५६७८९"
_BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
_THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"

_DIGIT_MAP: dict[int, str] = {}
for _src in (
    _PERSIAN_DIGITS,
    _ARABIC_DIGITS,
    _DEVANAGARI_DIGITS,
    _BENGALI_DIGITS,
    _THAI_DIGITS,
):
    for _i, _ch in enumerate(_src):
        _DIGIT_MAP[ord(_ch)] = str(_i)

# ASCII -> Persian digits, for rendering back out in the native style
_TO_PERSIAN_DIGITS = {str(i): _PERSIAN_DIGITS[i] for i in range(10)}

# --- Arabic/Persian letter folding ----------------------------------------
_ARABIC_FOLD = {
    "ي": "ی",
    "ى": "ی",
    "ئ": "ی",
    "ك": "ک",
    "ﻙ": "ک",
    "ة": "ه",
    "ۀ": "ه",
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ؤ": "و",
    "ٲ": "ا",
    "ھ": "ه",
    "ہ": "ه",
}
_ARABIC_FOLD_TABLE = {ord(k): v for k, v in _ARABIC_FOLD.items()}

# zero-width joiners, bidi controls, tatweel and Arabic diacritics
_INVISIBLE = re.compile(
    "["
    "​-‏"  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "‪-‮"  # bidi embedding/override
    "⁦-⁩"  # bidi isolates
    "﻿"  # BOM
    "ـ"  # tatweel
    "ً-ْٰ"  # harakat
    "]"
)

# characters that are plate furniture, never part of the reading
_SEPARATORS = re.compile(r"[\s\-‐-―_.,;:/\\|·•*'\"`~^+=()\[\]{}<>?!@#$%&]+")

# One run of Arabic-script letters (Arabic-Indic *digits* are excluded: numbers
# read left-to-right even inside RTL text). Interior spaces are part of the run
# so that several Arabic words stay in right-to-left order relative to each other.
_ARABIC_LETTER = "ؠ-ٟٮ-ۯۺ-ۿݐ-ݿﭐ-﷿ﹰ-﻿"
_ARABIC_RUN = re.compile(f"[{_ARABIC_LETTER}]+(?:[ ‌]+[{_ARABIC_LETTER}]+)*")


def fold_digits(text: str) -> str:
    """Convert any supported digit script to ASCII digits."""
    return text.translate(_DIGIT_MAP)


def to_persian_digits(text: str) -> str:
    """Render ASCII digits back as Persian-Indic digits (for display only)."""
    return "".join(_TO_PERSIAN_DIGITS.get(c, c) for c in text)


def fold_arabic(text: str) -> str:
    """Fold Arabic letter variants onto their Persian equivalents."""
    return text.translate(_ARABIC_FOLD_TABLE)


def normalize(
    text: str,
    *,
    script: str = "latin",
    keep_case: bool = False,
    noise_words: tuple[str, ...] | list[str] = (),
) -> str:
    """Fold `text` into the canonical form used for grammar matching.

    Args:
        text: raw OCR output.
        script: the country's script (``latin``, ``arabic``, …). Latin text is
            uppercased; Arabic-script text gets letter-variant folding.
        keep_case: skip uppercasing (useful when debugging an engine).
        noise_words: printed-on-the-plate words that are never part of the
            reading (``ایران`` on Iranian plates, ``NEW YORK`` on a US plate).
            Stripped longest-first, after folding. Single characters are
            ignored on purpose — ``D`` is a real German city code, not noise.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE.sub("", out)
    out = fold_digits(out)
    if script == "arabic":
        out = fold_arabic(out)
    out = _SEPARATORS.sub("", out)
    if not keep_case and script == "latin":
        out = out.upper()
    for word in sorted((w for w in noise_words if len(w) > 1), key=len, reverse=True):
        folded = fold_arabic(word) if script == "arabic" else word
        if not keep_case and script == "latin":
            folded = folded.upper()
        folded = _SEPARATORS.sub("", folded)
        if folded:
            out = out.replace(folded, "")
    return out.strip()


def tokenize(text: str, letters: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Split `text` into plate tokens, matching multi-character letters first.

    >>> tokenize("12الف34511", ["الف", "ب"])
    ['1', '2', 'الف', '3', '4', '5', '1', '1']
    """
    multi = sorted((letter for letter in letters if len(letter) > 1), key=len, reverse=True)
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        for m in multi:
            if text.startswith(m, i):
                tokens.append(m)
                i += len(m)
                break
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def shape_rtl(text: str, base_dir: str = "L") -> str:
    """Reshape + bidi-reorder Arabic-script text so PIL draws it correctly.

    Args:
        text: the string to render.
        base_dir: paragraph direction, ``"L"`` or ``"R"``.

    **Why the default is ``"L"``.** A plate is read left-to-right even when it
    is written in an RTL script — that is exactly what ``read_order: ltr`` in
    ``configs/countries/ir.yaml`` records. Running ``12 ب 345 | ایران 11``
    through the bidi algorithm with an RTL base direction reorders the numeric
    runs and renders it as ``345 ب 12``: a wrong plate, drawn confidently onto
    the video. Forcing an LTR base keeps the groups in plate order while still
    reversing the Arabic segments internally, which is what reshaped Arabic
    needs.

    Pass ``base_dir="R"`` for prose, not for plates.

    Returns the text unchanged when ``arabic-reshaper`` / ``python-bidi`` are
    not installed, so rendering degrades instead of crashing.
    """
    try:  # pragma: no cover - optional dependency
        import arabic_reshaper
    except Exception:  # pragma: no cover - optional dependency
        return text

    if base_dir == "R":
        # Prose: the full bidi algorithm is exactly right.
        try:  # pragma: no cover - optional dependency
            try:
                from bidi import get_display  # python-bidi >= 0.5
            except ImportError:
                from bidi.algorithm import get_display  # python-bidi < 0.5
            return get_display(arabic_reshaper.reshape(text))
        except Exception:  # pragma: no cover - optional dependency
            return text

    # Plate mode: reorder *within* each Arabic run only, leaving the overall
    # left-to-right sequence of groups exactly as the layout composed it.
    # Running the whole mixed string through bidi — even with an LTR base —
    # moves the digit runs around the Arabic word and silently renders a
    # different plate.
    def _flip(match: re.Match[str]) -> str:
        return arabic_reshaper.reshape(match.group(0))[::-1]

    return _ARABIC_RUN.sub(_flip, text)


def has_rtl(text: str) -> bool:
    """True when the string contains Arabic/Hebrew-script characters."""
    return any("֐" <= ch <= "ࣿ" or "יִ" <= ch <= "﷿" for ch in text)
