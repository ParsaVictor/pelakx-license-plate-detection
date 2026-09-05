"""Core data types shared by every PelakX stage.

Everything here is a plain dataclass with no heavy dependencies, so the
grammar / fusion / analytics layers can be imported (and unit-tested) without
torch, onnxruntime or OpenCV installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in absolute pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    def as_int(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def clip(self, w: int, h: int) -> BBox:
        """Clamp to a ``w`` x ``h`` frame.

        A box entirely outside the frame clamps to zero area rather than to a
        one-pixel sliver at the edge — callers test ``area``/``width`` to decide
        whether anything is left.
        """
        x1 = min(max(0.0, self.x1), float(w))
        y1 = min(max(0.0, self.y1), float(h))
        x2 = min(max(0.0, self.x2), float(w))
        y2 = min(max(0.0, self.y2), float(h))
        return BBox(x1, y1, max(x1, x2), max(y1, y2))

    def pad(self, ratio: float, w: int | None = None, h: int | None = None) -> BBox:
        """Grow the box by `ratio` on every side, optionally clipped to a frame."""
        dx, dy = self.width * ratio, self.height * ratio
        out = BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)
        return out.clip(w, h) if w is not None and h is not None else out

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_center_of(self, other: BBox) -> bool:
        cx, cy = other.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


# ---------------------------------------------------------------------------
# Detection / tracking
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    """One object emitted by a detector."""

    bbox: BBox
    confidence: float
    class_id: int = 0
    class_name: str = "object"
    track_id: int | None = None


@dataclass(slots=True)
class RawRead:
    """Exactly what an OCR engine returned, before any grammar is applied."""

    text: str
    confidence: float
    engine: str
    char_confidences: list[float] = field(default_factory=list)
    elapsed_ms: float = 0.0
    #: country/region the engine guessed, when it can (fast-plate-ocr does).
    #: Used by `--country auto` as a prior, never as the answer.
    region: str = ""

    def per_char_confidence(self) -> list[float]:
        """Character confidences, falling back to a flat value when unavailable."""
        if len(self.char_confidences) == len(self.text):
            return list(self.char_confidences)
        return [self.confidence] * len(self.text)


@dataclass(slots=True)
class PlateRead:
    """A grammar-validated plate reading."""

    canonical: str  # compact, DB-friendly key, e.g. "12B34511"
    display: str  # human/native rendering, e.g. "12 ب 345 | ایران 11"
    country: str  # ISO-3166 alpha-2
    layout_id: str  # which layout of that country matched
    confidence: float  # 0..1, grammar-adjusted
    ocr_confidence: float = 0.0  # 0..1, raw from the engine
    fields: dict[str, str] = field(default_factory=dict)
    repairs: int = 0  # how many confusion substitutions were applied
    valid: bool = True  # did it satisfy the country's grammar at all?
    engine: str = ""
    raw_text: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.display} ({self.confidence:.0%})"


@dataclass(slots=True)
class PlateObservation:
    """One plate sighting inside one frame, attached to a vehicle track."""

    frame_index: int
    timestamp: float
    bbox: BBox
    det_confidence: float
    quality: float = 1.0  # 0..1 from the quality gate
    raw: RawRead | None = None
    read: PlateRead | None = None
    track_id: int | None = None


@dataclass(slots=True)
class TrackRecord:
    """Everything PelakX knows about one tracked vehicle."""

    track_id: int
    class_name: str = "vehicle"
    first_frame: int = 0
    last_frame: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    boxes: list[BBox] = field(default_factory=list)
    observations: list[PlateObservation] = field(default_factory=list)
    consensus: PlateRead | None = None
    speed_kmh: float | None = None
    direction: str | None = None
    crossings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def n_reads(self) -> int:
        return sum(1 for o in self.observations if o.read is not None)

    def trajectory(self) -> list[tuple[float, float]]:
        return [b.center for b in self.boxes]

    def displacement(self) -> float:
        pts = self.trajectory()
        if len(pts) < 2:
            return 0.0
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        return math.hypot(x2 - x1, y2 - y1)


@dataclass(slots=True)
class VehicleEvent:
    """A finalized, exportable record for one vehicle that left the scene."""

    track_id: int
    plate: PlateRead | None
    class_name: str
    first_seen: float
    last_seen: float
    n_frames: int
    n_reads: int
    speed_kmh: float | None = None
    direction: str | None = None
    crossings: list[str] = field(default_factory=list)
    best_crop_path: str | None = None
    alerts: list[str] = field(default_factory=list)
    source: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flat dict for CSV / SQLite export."""
        p = self.plate
        return {
            "track_id": self.track_id,
            "plate": p.canonical if p else "",
            "plate_display": p.display if p else "",
            "country": p.country if p else "",
            "layout": p.layout_id if p else "",
            "confidence": round(p.confidence, 4) if p else 0.0,
            "ocr_confidence": round(p.ocr_confidence, 4) if p else 0.0,
            "valid": bool(p.valid) if p else False,
            "vehicle_class": self.class_name,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "n_frames": self.n_frames,
            "n_reads": self.n_reads,
            "speed_kmh": round(self.speed_kmh, 1) if self.speed_kmh else "",
            "direction": self.direction or "",
            "crossings": "|".join(self.crossings),
            "alerts": "|".join(self.alerts),
            "crop": self.best_crop_path or "",
            "source": self.source,
        }
