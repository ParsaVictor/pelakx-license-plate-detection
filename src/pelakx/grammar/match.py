"""Grammar matching — turn raw OCR text into a validated :class:`PlateRead`.

This is where PelakX differs from a plain "YOLO + OCR" demo. An OCR engine
hands us a noisy string; the country grammar tells us what a *legal* plate for
that country looks like. Matching therefore does four things:

1. **Normalize** the text into one canonical script (:mod:`pelakx.grammar.normalize`).
2. **Tokenize** it, respecting multi-character letters like ``الف``.
3. **Repair** it: when a token lands in a slot it cannot legally occupy, try the
   country's OCR confusion map (``O`` -> ``0``, ``ك`` -> ``ک``, …) under a
   bounded budget, so ``12ب345I1`` becomes ``12ب34511`` instead of being thrown away.
4. **Validate** it against the layout's structural rules (province ranges,
   forbidden letters, checksums) and fold the result into a confidence score.

The score is deliberately interpretable::

    confidence = ocr_confidence
               * (1 - REPAIR_PENALTY * n_repairs)
               + sum(weight of every validator that passed)

clamped to [0, 1]. A plate that parses cleanly *and* satisfies its validators
scores above its raw OCR confidence; a plate that needed three substitutions
and failed a range check scores below it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from pelakx.grammar.normalize import normalize, tokenize
from pelakx.grammar.spec import SLOT_ALNUM, SLOT_DIGIT, SLOT_LETTER, CountrySpec, Layout
from pelakx.types import PlateRead, RawRead

#: confidence lost per confusion substitution applied
REPAIR_PENALTY = 0.06
#: how many substitutions we are willing to try before giving up on a layout
DEFAULT_REPAIR_BUDGET = 2
#: penalty applied when nothing matched and we fall back to a raw reading
UNPARSED_PENALTY = 0.45


@dataclass(slots=True)
class MatchResult:
    """Internal result of matching one layout."""

    layout: Layout
    tokens: list[str]
    fields: dict[str, str]
    repairs: int
    validator_bonus: float
    validators_failed: list[str]

    @property
    def text(self) -> str:
        return "".join(self.tokens)


# ---------------------------------------------------------------------------
# slot-based matching (fixed-length layouts)
# ---------------------------------------------------------------------------
def _fits(token: str, slot: str, spec: CountrySpec) -> bool:
    cls = spec.token_class(token)
    if slot == SLOT_ALNUM:
        return cls in (SLOT_DIGIT, SLOT_LETTER)
    return cls == slot


def _candidates_for(token: str, slot: str, spec: CountrySpec) -> list[str]:
    """Every value `token` could legally take in `slot`, best first."""
    out: list[str] = []
    if _fits(token, slot, spec):
        out.append(token)
    if slot in (SLOT_DIGIT, SLOT_ALNUM):
        sub = spec.to_digit.get(token)
        if sub and sub not in out and _fits(sub, SLOT_DIGIT, spec):
            out.append(sub)
    if slot in (SLOT_LETTER, SLOT_ALNUM):
        sub = spec.to_letter.get(token)
        if sub and sub not in out and _fits(sub, SLOT_LETTER, spec):
            out.append(sub)
    return out


def _match_slots(
    tokens: list[str], layout: Layout, spec: CountrySpec, budget: int
) -> tuple[list[str], int] | None:
    """Fit `tokens` into `layout.slots`, repairing at most `budget` of them.

    Returns the repaired token list and the number of repairs, or None.
    """
    if len(tokens) != layout.length:
        return None

    per_slot: list[list[str]] = []
    for token, slot in zip(tokens, layout.slots, strict=True):
        options = _candidates_for(token, slot, spec)
        if not options:
            return None  # unrepairable token -> this layout is impossible
        per_slot.append(options)

    # Cheapest-first search: try every combination ordered by repair count.
    best: tuple[list[str], int] | None = None
    for combo in product(*per_slot):
        repairs = sum(1 for original, chosen in zip(tokens, combo, strict=True) if original != chosen)
        if repairs > budget:
            continue
        if best is None or repairs < best[1]:
            best = (list(combo), repairs)
        if repairs == 0:
            break
    return best


def _fields_from_slots(tokens: list[str], layout: Layout) -> dict[str, str]:
    if not layout.groups:
        return {"serial": "".join(tokens)}
    return {name: "".join(tokens[start:end]) for name, (start, end) in layout.groups.items()}


# ---------------------------------------------------------------------------
# regex matching (variable-length layouts)
# ---------------------------------------------------------------------------
def _match_regex(text: str, layout: Layout) -> dict[str, str] | None:
    pattern = layout.compiled()
    if pattern is None:
        return None
    m = pattern.fullmatch(text) or pattern.match(text)
    if not m:
        return None
    groups = {k: (v or "") for k, v in m.groupdict().items()}
    return groups or {"serial": m.group(0)}


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------
def _run_validators(
    spec: CountrySpec, layout: Layout, fields: dict[str, str]
) -> tuple[float, list[str]]:
    bonus = 0.0
    failed: list[str] = []
    for validator in spec.validators:
        if validator.applies_to and layout.id not in validator.applies_to:
            continue
        result = validator.check(fields)
        if result is None:
            continue
        if result:
            bonus += validator.weight
        else:
            failed.append(validator.id)
            bonus -= validator.weight
    return bonus, failed


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _render(template: str, fields: dict[str, str], fallback: str) -> str:
    if not template:
        return fallback
    try:
        return template.format(**fields).strip()
    except (KeyError, IndexError):
        return fallback


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def parse(
    text: str,
    spec: CountrySpec,
    *,
    ocr_confidence: float = 1.0,
    engine: str = "",
    repair_budget: int = DEFAULT_REPAIR_BUDGET,
    strict: bool = False,
) -> PlateRead | None:
    """Parse `text` as a plate of country `spec`.

    Args:
        text: raw OCR output.
        spec: the country grammar to match against.
        ocr_confidence: the engine's own confidence, 0..1.
        engine: engine id, recorded on the result.
        repair_budget: max confusion substitutions per layout.
        strict: when True, return ``None`` instead of an ``valid=False``
            fallback reading if no layout matches.

    Returns:
        A :class:`PlateRead`, or ``None`` when nothing matched and
        ``strict=True`` (or the text normalized to nothing).
    """
    folded = normalize(text, script=spec.script, noise_words=spec.noise_words)
    if not folded:
        return None
    tokens = tokenize(folded, spec.letters)

    best: MatchResult | None = None
    best_score = float("-inf")

    for layout in spec.sorted_layouts():
        result: MatchResult | None = None
        if layout.is_fixed:
            fitted = _match_slots(tokens, layout, spec, repair_budget)
            if fitted is not None:
                repaired, repairs = fitted
                fields = _fields_from_slots(repaired, layout)
                bonus, failed = _run_validators(spec, layout, fields)
                result = MatchResult(layout, repaired, fields, repairs, bonus, failed)
        else:
            fields = _match_regex(folded, layout)
            if fields is not None:
                bonus, failed = _run_validators(spec, layout, fields)
                result = MatchResult(layout, tokens, fields, 0, bonus, failed)

        if result is None:
            continue

        # Prefer: fewer repairs, then higher validator bonus, then higher priority.
        score = (
            layout.priority / 1000.0
            - REPAIR_PENALTY * result.repairs
            + result.validator_bonus
            - 0.25 * len(result.validators_failed)
        )
        if score > best_score:
            best, best_score = result, score

    if best is None:
        if strict:
            return None
        # Nothing matched: still surface the reading, flagged invalid, so the
        # operator sees "we saw a plate we could not parse" instead of nothing.
        return PlateRead(
            canonical=folded,
            display=folded,
            country=spec.code,
            layout_id="",
            confidence=max(0.0, ocr_confidence - UNPARSED_PENALTY),
            ocr_confidence=ocr_confidence,
            fields={"serial": folded},
            repairs=0,
            valid=False,
            engine=engine,
            raw_text=text,
        )

    # Repairs scale the OCR confidence down; validator bonuses close a fraction
    # of the remaining gap to 1.0 rather than being added on top, so a
    # grammar-clean plate can approach — but never reach — certainty.
    base = ocr_confidence * (1.0 - REPAIR_PENALTY * best.repairs)
    bonus = best.validator_bonus
    confidence = base + bonus * (1.0 - base) if bonus > 0 else base + bonus
    confidence = max(0.0, min(0.999, confidence))
    canonical = _render(best.layout.canonical, best.fields, best.text)
    display = _render(best.layout.display, best.fields, canonical)

    return PlateRead(
        canonical=canonical,
        display=display,
        country=spec.code,
        layout_id=best.layout.id,
        confidence=confidence,
        ocr_confidence=ocr_confidence,
        fields=dict(best.fields),
        repairs=best.repairs,
        valid=not best.validators_failed,
        engine=engine,
        raw_text=text,
    )


def parse_raw(raw: RawRead, spec: CountrySpec, **kwargs) -> PlateRead | None:
    """Convenience wrapper: parse a :class:`RawRead` straight from an engine."""
    return parse(
        raw.text,
        spec,
        ocr_confidence=raw.confidence,
        engine=raw.engine,
        **kwargs,
    )


def identify_country(
    text: str,
    specs: list[CountrySpec],
    *,
    ocr_confidence: float = 1.0,
    engine: str = "",
) -> PlateRead | None:
    """Auto-detect which country a plate string belongs to.

    Tries every spec and returns the highest-confidence *valid* parse. Useful
    for border crossings, international corridors, or `--country auto`.
    """
    best: PlateRead | None = None
    for spec in specs:
        read = parse(text, spec, ocr_confidence=ocr_confidence, engine=engine, strict=True)
        if read is None:
            continue
        if best is None or read.confidence > best.confidence:
            best = read
    return best
