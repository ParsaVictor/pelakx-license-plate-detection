"""Temporal consensus — one vehicle, one plate.

A per-frame ALPR demo prints a different string every frame and calls it a
day. PelakX treats every frame as *evidence about one tracked vehicle* and
fuses it, which is where most of the real-world accuracy comes from.

Two votes run at once:

**String vote** — every reading votes for its canonical string, weighted by
``ocr_confidence x crop_quality x grammar_validity``.

**Character vote** — among readings of the modal length, each position votes
independently. This recovers plates that were *never* read correctly in a
single frame: if frame 1 says ``12B34511``, frame 2 says ``12B34S11`` and
frame 3 says ``12B34511``, position 5 still resolves to ``5``.

The winner of the character vote is re-parsed through the country grammar, so
a stitched-together string still has to be a legal plate before it wins.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pelakx.grammar import parse
from pelakx.grammar.normalize import tokenize
from pelakx.grammar.spec import CountrySpec
from pelakx.types import PlateRead

#: weight multiplier for a reading that failed its country's grammar
INVALID_WEIGHT = 0.35
#: how fast confidence saturates with the number of agreeing sightings
EVIDENCE_BASE = 0.6


@dataclass(slots=True)
class Vote:
    read: PlateRead
    weight: float
    frame_index: int = 0


@dataclass(slots=True)
class Candidate:
    """One competing plate string and the evidence behind it."""

    canonical: str
    weight: float
    votes: int
    best_read: PlateRead

    @property
    def share(self) -> float:  # filled in by TrackConsensus.ranked()
        return 0.0


@dataclass(slots=True)
class TrackConsensus:
    """Accumulates plate readings for one tracked vehicle and fuses them."""

    spec: CountrySpec
    min_votes: int = 1
    max_votes: int = 200
    votes: list[Vote] = field(default_factory=list)

    # -- ingestion ----------------------------------------------------------
    def add(self, read: PlateRead | None, *, quality: float = 1.0, frame_index: int = 0) -> None:
        """Record one plate reading as evidence for this track."""
        if read is None or not read.canonical:
            return
        weight = max(0.0, read.confidence) * max(0.0, min(1.0, quality))
        if not read.valid:
            weight *= INVALID_WEIGHT
        if weight <= 0.0:
            return
        if len(self.votes) >= self.max_votes:
            # Keep the strongest evidence rather than the oldest.
            weakest = min(range(len(self.votes)), key=lambda i: self.votes[i].weight)
            if self.votes[weakest].weight >= weight:
                return
            self.votes.pop(weakest)
        self.votes.append(Vote(read, weight, frame_index))

    def __len__(self) -> int:
        return len(self.votes)

    @property
    def total_weight(self) -> float:
        return sum(v.weight for v in self.votes)

    # -- string vote --------------------------------------------------------
    def ranked(self) -> list[Candidate]:
        """Competing plate strings, strongest first."""
        buckets: dict[str, list[Vote]] = defaultdict(list)
        for vote in self.votes:
            buckets[vote.read.canonical].append(vote)
        out = [
            Candidate(
                canonical=key,
                weight=sum(v.weight for v in group),
                votes=len(group),
                best_read=max(group, key=lambda v: v.weight).read,
            )
            for key, group in buckets.items()
        ]
        out.sort(key=lambda c: (-c.weight, -c.votes, c.canonical))
        return out

    # -- character vote -----------------------------------------------------
    def _character_vote(self) -> tuple[str, float] | None:
        """Per-position weighted vote among readings of the modal length.

        Returns the stitched string and its mean per-position agreement, or
        None when there is not enough evidence.
        """
        tokenized: list[tuple[list[str], float]] = [
            (tokenize(v.read.canonical, self.spec.letters), v.weight) for v in self.votes
        ]
        if not tokenized:
            return None

        length_weight: dict[int, float] = defaultdict(float)
        for tokens, weight in tokenized:
            length_weight[len(tokens)] += weight
        modal_length = max(length_weight, key=lambda k: length_weight[k])

        group = [(t, w) for t, w in tokenized if len(t) == modal_length]
        if len(group) < 2:
            return None

        stitched: list[str] = []
        agreements: list[float] = []
        for position in range(modal_length):
            tally: dict[str, float] = defaultdict(float)
            for tokens, weight in group:
                tally[tokens[position]] += weight
            total = sum(tally.values())
            winner = max(tally, key=lambda k: tally[k])
            stitched.append(winner)
            agreements.append(tally[winner] / total if total else 0.0)
        return "".join(stitched), sum(agreements) / len(agreements)

    # -- fusion -------------------------------------------------------------
    def result(self) -> PlateRead | None:
        """The fused reading for this track, or None if there is no evidence."""
        if len(self.votes) < self.min_votes or not self.votes:
            return None

        candidates = self.ranked()
        total = self.total_weight or 1.0
        winner = candidates[0]
        agreement = winner.weight / total
        best = winner.best_read
        n_agreeing = winner.votes

        # Does the character vote propose something the string vote never saw?
        stitched = self._character_vote()
        if stitched is not None:
            text, char_agreement = stitched
            if text != winner.canonical:
                repaired = parse(
                    text,
                    self.spec,
                    ocr_confidence=min(0.99, char_agreement),
                    engine="consensus",
                    strict=True,
                )
                # Only let the stitched string win if it is a *legal* plate and
                # the per-frame vote was genuinely split.
                if repaired is not None and repaired.valid and char_agreement > agreement:
                    best = repaired
                    agreement = char_agreement
                    n_agreeing = len(self.votes)

        mean_conf = (
            sum(
                v.weight * v.read.confidence
                for v in self.votes
                if v.read.canonical == best.canonical
            )
            / max(1e-9, sum(v.weight for v in self.votes if v.read.canonical == best.canonical))
            if any(v.read.canonical == best.canonical for v in self.votes)
            else best.confidence
        )

        # Confidence = how good the readings were, discounted by how much the
        # frames disagreed, then boosted by how many frames agreed.
        confidence = mean_conf * (0.55 + 0.45 * agreement)
        evidence = 1.0 - EVIDENCE_BASE**n_agreeing
        confidence += (1.0 - confidence) * 0.5 * evidence

        return PlateRead(
            canonical=best.canonical,
            display=best.display,
            country=best.country,
            layout_id=best.layout_id,
            confidence=round(min(0.999, max(0.0, confidence)), 4),
            ocr_confidence=round(mean_conf, 4),
            fields=dict(best.fields),
            repairs=best.repairs,
            valid=best.valid,
            engine=f"{best.engine}+vote({len(self.votes)})",
            raw_text=best.raw_text,
        )

    def explain(self) -> dict:
        """Human-readable breakdown of the vote — used by the dashboard."""
        total = self.total_weight or 1.0
        return {
            "n_votes": len(self.votes),
            "total_weight": round(total, 4),
            "candidates": [
                {
                    "canonical": c.canonical,
                    "display": c.best_read.display,
                    "votes": c.votes,
                    "weight": round(c.weight, 4),
                    "share": round(c.weight / total, 4),
                    "valid": c.best_read.valid,
                }
                for c in self.ranked()[:5]
            ],
        }


class VoterPool:
    """One :class:`TrackConsensus` per track id, created on demand."""

    def __init__(self, spec: CountrySpec, *, min_votes: int = 1, max_votes: int = 200) -> None:
        self.spec = spec
        self.min_votes = min_votes
        self.max_votes = max_votes
        self._pool: dict[int, TrackConsensus] = {}

    def votes(self, track_id: int) -> int:
        consensus = self._pool.get(track_id)
        return len(consensus) if consensus else 0

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._pool

    def __len__(self) -> int:
        return len(self._pool)

    def get(self, track_id: int) -> TrackConsensus:
        consensus = self._pool.get(track_id)
        if consensus is None:
            consensus = TrackConsensus(
                self.spec, min_votes=self.min_votes, max_votes=self.max_votes
            )
            self._pool[track_id] = consensus
        return consensus

    def add(
        self, track_id: int, read: PlateRead | None, *, quality: float = 1.0, frame_index: int = 0
    ) -> None:
        self.get(track_id).add(read, quality=quality, frame_index=frame_index)

    def result(self, track_id: int) -> PlateRead | None:
        consensus = self._pool.get(track_id)
        return consensus.result() if consensus else None

    def pop(self, track_id: int) -> PlateRead | None:
        """Finalize and forget a track that has left the scene."""
        consensus = self._pool.pop(track_id, None)
        return consensus.result() if consensus else None

    def explain(self, track_id: int) -> dict:
        consensus = self._pool.get(track_id)
        return consensus.explain() if consensus else {"n_votes": 0, "candidates": []}


class MultiVoterPool:
    """A :class:`VoterPool` per country, for ``--country auto`` streams.

    In a mixed-traffic stream one vehicle's readings can legitimately parse as
    several countries across frames. Votes are kept in separate per-country
    pools and only compared at the end, so a burst of misparses as one country
    cannot drown out the consistent readings of another.
    """

    def __init__(self, specs: dict[str, CountrySpec], *, min_votes: int = 1, max_votes: int = 200):
        self.specs = specs
        self.min_votes = min_votes
        self.max_votes = max_votes
        self._pools: dict[str, VoterPool] = {}

    def _pool(self, country: str) -> VoterPool | None:
        spec = self.specs.get(country.upper())
        if spec is None:
            return None
        pool = self._pools.get(spec.code)
        if pool is None:
            pool = VoterPool(spec, min_votes=self.min_votes, max_votes=self.max_votes)
            self._pools[spec.code] = pool
        return pool

    def __len__(self) -> int:
        return len({tid for pool in self._pools.values() for tid in pool._pool})

    def add(
        self, track_id: int, read: PlateRead | None, *, quality: float = 1.0, frame_index: int = 0
    ) -> None:
        if read is None:
            return
        pool = self._pool(read.country)
        if pool is not None:
            pool.add(track_id, read, quality=quality, frame_index=frame_index)

    def votes(self, track_id: int) -> int:
        return sum(pool.votes(track_id) for pool in self._pools.values())

    def result(self, track_id: int) -> PlateRead | None:
        candidates = [pool.result(track_id) for pool in self._pools.values()]
        found = [c for c in candidates if c is not None]
        return max(found, key=lambda r: r.confidence) if found else None

    def pop(self, track_id: int) -> PlateRead | None:
        found = [r for pool in self._pools.values() if (r := pool.pop(track_id)) is not None]
        return max(found, key=lambda r: r.confidence) if found else None

    def explain(self, track_id: int) -> dict:
        return {
            "by_country": {
                code: pool.explain(track_id)
                for code, pool in self._pools.items()
                if pool.votes(track_id)
            }
        }
