"""Plate grammar: country specs, script normalization and validated parsing."""

from __future__ import annotations

from pelakx.grammar import normalize, registry
from pelakx.grammar.match import identify_country, parse, parse_raw
from pelakx.grammar.regions import region_to_code
from pelakx.grammar.spec import CountrySpec, GrammarError, Layout, Validator

__all__ = [
    "CountrySpec",
    "GrammarError",
    "Layout",
    "Validator",
    "identify_country",
    "normalize",
    "parse",
    "parse_raw",
    "region_to_code",
    "registry",
]
