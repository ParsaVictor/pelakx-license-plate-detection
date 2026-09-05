"""Country grammar specification — the schema behind ``configs/countries/*.yaml``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SLOT_DIGIT = "D"
SLOT_LETTER = "L"
SLOT_ALNUM = "A"
VALID_SLOTS = frozenset({SLOT_DIGIT, SLOT_LETTER, SLOT_ALNUM})


class GrammarError(ValueError):
    """Raised when a country YAML file is malformed."""


@dataclass(slots=True)
class Validator:
    """A structural check applied after a layout matches."""

    id: str
    type: str
    field: str = ""
    applies_to: tuple[str, ...] = ()
    value: Any = None
    min: int | None = None
    max: int | None = None
    weight: float = 0.1

    def check(self, fields: dict[str, str]) -> bool | None:
        """Return True/False, or None when the validator does not apply."""
        raw = fields.get(self.field)
        if raw is None:
            return None
        if self.type == "int_range":
            try:
                n = int(raw)
            except ValueError:
                return False
            lo = self.min if self.min is not None else -(10**9)
            hi = self.max if self.max is not None else 10**9
            return lo <= n <= hi
        if self.type == "not_equal":
            return raw != str(self.value)
        if self.type == "equals":
            return raw == str(self.value)
        if self.type == "charset_excludes":
            return not any(ch in str(self.value) for ch in raw)
        if self.type == "charset_includes":
            return all(ch in str(self.value) for ch in raw)
        if self.type == "in_set":
            return raw in {str(v) for v in (self.value or [])}
        if self.type == "regex":
            return re.fullmatch(str(self.value), raw) is not None
        raise GrammarError(f"unknown validator type: {self.type!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Validator:
        return cls(
            id=str(d.get("id", "unnamed")),
            type=str(d["type"]),
            field=str(d.get("field", "")),
            applies_to=tuple(d.get("applies_to") or ()),
            value=d.get("value"),
            min=d.get("min"),
            max=d.get("max"),
            weight=float(d.get("weight", 0.1)),
        )


@dataclass(slots=True)
class Layout:
    """One legal plate shape for a country."""

    id: str
    name_en: str = ""
    name_native: str = ""
    slots: str = ""
    regex: str = ""
    groups: dict[str, tuple[int, int]] = field(default_factory=dict)
    display: str = "{serial}"
    display_en: str = ""
    canonical: str = ""
    priority: int = 100
    _compiled: re.Pattern[str] | None = None

    @property
    def is_fixed(self) -> bool:
        """Fixed-length layouts support slot-level confusion repair."""
        return bool(self.slots)

    @property
    def length(self) -> int:
        return len(self.slots)

    def compiled(self) -> re.Pattern[str] | None:
        if self.regex and self._compiled is None:
            self._compiled = re.compile(self.regex)
        return self._compiled

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Layout:
        slots = str(d.get("slots", "") or "")
        regex = str(d.get("regex", "") or "")
        if not slots and not regex:
            raise GrammarError(f"layout {d.get('id')!r} needs either 'slots' or 'regex'")
        bad = set(slots) - VALID_SLOTS
        if bad:
            raise GrammarError(
                f"layout {d.get('id')!r} has invalid slot chars {sorted(bad)}; "
                f"use only {sorted(VALID_SLOTS)}"
            )
        groups: dict[str, tuple[int, int]] = {}
        for name, span in (d.get("groups") or {}).items():
            if slots:
                if not (isinstance(span, (list, tuple)) and len(span) == 2):
                    raise GrammarError(f"group {name!r} must be [start, end]")
                start, end = int(span[0]), int(span[1])
                if not (0 <= start < end <= len(slots)):
                    raise GrammarError(
                        f"group {name!r} span [{start}, {end}) is outside slots of length {len(slots)}"
                    )
                groups[str(name)] = (start, end)
        return cls(
            id=str(d["id"]),
            name_en=str(d.get("name_en", "")),
            name_native=str(d.get("name_native", "")),
            slots=slots,
            regex=regex,
            groups=groups,
            display=str(d.get("display", "{serial}")),
            display_en=str(d.get("display_en", "")),
            canonical=str(d.get("canonical", "")),
            priority=int(d.get("priority", 100)),
        )


@dataclass(slots=True)
class CountrySpec:
    """A fully-parsed country grammar."""

    code: str
    name_en: str
    name_native: str = ""
    iso3: str = ""
    script: str = "latin"
    text_direction: str = "ltr"
    read_order: str = "ltr"
    digit_style: str = "latin"
    preferred_engines: tuple[str, ...] = ()
    fallback_engines: tuple[str, ...] = ()
    digits: str = "0123456789"
    letters: tuple[str, ...] = ()
    noise_words: tuple[str, ...] = ()
    layouts: tuple[Layout, ...] = ()
    to_digit: dict[str, str] = field(default_factory=dict)
    to_letter: dict[str, str] = field(default_factory=dict)
    validators: tuple[Validator, ...] = ()
    letter_semantics: dict[str, dict[str, str]] = field(default_factory=dict)
    province_codes: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    # -- derived helpers ----------------------------------------------------
    @property
    def engine_chain(self) -> tuple[str, ...]:
        """Preferred engines first, then fallbacks, de-duplicated."""
        seen: dict[str, None] = {}
        for e in (*self.preferred_engines, *self.fallback_engines):
            seen.setdefault(e, None)
        return tuple(seen)

    @property
    def letter_set(self) -> frozenset[str]:
        return frozenset(self.letters)

    @property
    def digit_set(self) -> frozenset[str]:
        return frozenset(self.digits)

    @property
    def max_length(self) -> int:
        fixed = [layout.length for layout in self.layouts if layout.is_fixed]
        return max(fixed) if fixed else 12

    def token_class(self, token: str) -> str:
        """Classify a token as a digit (D), a letter (L) or unknown (?)."""
        if token in self.digit_set:
            return SLOT_DIGIT
        if token in self.letter_set:
            return SLOT_LETTER
        return "?"

    def sorted_layouts(self) -> tuple[Layout, ...]:
        return tuple(sorted(self.layouts, key=lambda layout: -layout.priority))

    def describe_letter(self, letter: str, lang: str = "en") -> str:
        entry = self.letter_semantics.get(letter)
        return entry.get(lang, "") if entry else ""

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any], *, source_path: str = "") -> CountrySpec:
        try:
            code = str(d["code"]).upper()
            alphabet = d.get("alphabet") or {}
            engines = d.get("ocr_engines") or {}
            confusions = d.get("confusions") or {}
            layouts = tuple(Layout.from_dict(x) for x in (d.get("layouts") or []))
        except KeyError as exc:  # pragma: no cover - defensive
            raise GrammarError(f"{source_path or '<dict>'}: missing key {exc}") from exc
        if not layouts:
            raise GrammarError(f"{source_path or code}: at least one layout is required")
        ids = [layout.id for layout in layouts]
        if len(set(ids)) != len(ids):
            raise GrammarError(f"{source_path or code}: duplicate layout ids in {ids}")
        return cls(
            code=code,
            name_en=str(d.get("name_en", code)),
            name_native=str(d.get("name_native", "")),
            iso3=str(d.get("iso3", "")),
            script=str(d.get("script", "latin")),
            text_direction=str(d.get("text_direction", "ltr")),
            read_order=str(d.get("read_order", "ltr")),
            digit_style=str(d.get("digit_style", "latin")),
            preferred_engines=tuple(engines.get("preferred") or ()),
            fallback_engines=tuple(engines.get("fallback") or ()),
            digits=str(alphabet.get("digits", "0123456789")),
            letters=tuple(str(x) for x in (alphabet.get("letters") or ())),
            noise_words=tuple(str(x) for x in (d.get("noise_words") or ())),
            layouts=layouts,
            to_digit={str(k): str(v) for k, v in (confusions.get("to_digit") or {}).items()},
            to_letter={str(k): str(v) for k, v in (confusions.get("to_letter") or {}).items()},
            validators=tuple(Validator.from_dict(v) for v in (d.get("validators") or [])),
            letter_semantics={
                str(k): {str(kk): str(vv) for kk, vv in (v or {}).items()}
                for k, v in (d.get("letter_semantics") or {}).items()
            },
            province_codes={str(k): str(v) for k, v in (d.get("province_codes") or {}).items()},
            source_path=source_path,
        )
