"""Temporal fusion: turn many noisy per-frame readings into one answer."""

from __future__ import annotations

from pelakx.fusion.voting import (
    Candidate,
    MultiVoterPool,
    TrackConsensus,
    Vote,
    VoterPool,
)

__all__ = ["Candidate", "MultiVoterPool", "TrackConsensus", "Vote", "VoterPool"]
