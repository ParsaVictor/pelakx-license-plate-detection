"""Watchlist / BOLO matching with grammar-aware fuzziness.

An exact string match is the wrong tool here: the whole point of a watchlist
is to still fire when OCR got one character wrong. PelakX matches within a
bounded edit distance and, more usefully, treats *confusable* characters as
free — swapping ``0`` for ``O`` costs nothing, swapping ``0`` for ``7`` costs
a full edit — using the same confusion map the grammar layer uses.

Entries may be exact plates, prefixes (``12ب*``) or regexes (``re:^12ب3``).
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field

from pelakx.grammar.normalize import normalize
from pelakx.grammar.spec import CountrySpec


@dataclass(slots=True)
class WatchHit:
    """One watchlist entry that matched a plate."""

    entry: str
    plate: str
    distance: int
    kind: str  # exact | prefix | regex | fuzzy

    def label(self) -> str:
        return f"watchlist:{self.entry}" + ("" if self.distance == 0 else f"~{self.distance}")


def _confusable_pairs(spec: CountrySpec) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for mapping in (spec.to_digit, spec.to_letter):
        for src, dst in mapping.items():
            if src != dst:
                pairs.add(frozenset((src, dst)))
    return pairs


def confusion_distance(
    a: str, b: str, confusables: set[frozenset[str]], *, swap_cost: float = 0.25
) -> float:
    """Levenshtein distance where confusable substitutions are nearly free."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return float(max(la, lb))
    previous = [float(i) for i in range(lb + 1)]
    for i in range(1, la + 1):
        current = [float(i)]
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0.0
            elif frozenset((a[i - 1], b[j - 1])) in confusables:
                cost = swap_cost
            else:
                cost = 1.0
            current.append(min(previous[j] + 1.0, current[j - 1] + 1.0, previous[j - 1] + cost))
        previous = current
    return previous[lb]


@dataclass(slots=True)
class Watchlist:
    """A set of plates to alert on."""

    spec: CountrySpec
    max_distance: float = 1.0
    exact: set[str] = field(default_factory=set)
    prefixes: list[str] = field(default_factory=list)
    patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list)
    _confusables: set[frozenset[str]] = field(default_factory=set, repr=False)

    @classmethod
    def from_entries(
        cls, entries: list[str], spec: CountrySpec, *, max_distance: float = 1.0
    ) -> Watchlist:
        watchlist = cls(spec=spec, max_distance=max_distance, _confusables=_confusable_pairs(spec))
        for raw in entries or []:
            watchlist.add(raw)
        return watchlist

    def add(self, raw: str) -> None:
        entry = raw.strip()
        if not entry:
            return
        if entry.startswith("re:"):
            # A malformed pattern in a watchlist file must not take the camera down.
            with contextlib.suppress(re.error):
                self.patterns.append((entry, re.compile(entry[3:])))
            return
        folded = normalize(entry, script=self.spec.script, noise_words=self.spec.noise_words)
        if entry.endswith("*"):
            self.prefixes.append(folded.rstrip("*"))
        else:
            self.exact.add(folded)

    def __len__(self) -> int:
        return len(self.exact) + len(self.prefixes) + len(self.patterns)

    def match(self, plate: str) -> WatchHit | None:
        """Best match for `plate`, or None."""
        if not plate or not len(self):
            return None
        folded = normalize(plate, script=self.spec.script, noise_words=self.spec.noise_words)
        if folded in self.exact:
            return WatchHit(folded, folded, 0, "exact")
        for prefix in self.prefixes:
            if prefix and folded.startswith(prefix):
                return WatchHit(f"{prefix}*", folded, 0, "prefix")
        for raw, pattern in self.patterns:
            if pattern.search(folded):
                return WatchHit(raw, folded, 0, "regex")
        if self.max_distance <= 0:
            return None
        best: tuple[float, str] | None = None
        for entry in self.exact:
            if abs(len(entry) - len(folded)) > self.max_distance:
                continue
            distance = confusion_distance(folded, entry, self._confusables)
            if distance <= self.max_distance and (best is None or distance < best[0]):
                best = (distance, entry)
        if best is None:
            return None
        return WatchHit(best[1], folded, int(round(best[0])), "fuzzy")
