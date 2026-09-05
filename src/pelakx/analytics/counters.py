"""Counting lines and travel direction.

A counting line is two image points. A track crosses it when the signed side
of its centre flips between consecutive frames *and* the crossing point falls
inside the segment. The sign of the flip gives the direction, so one line
counts both ways without extra configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def _side(a: Point, b: Point, p: Point) -> float:
    """Signed area of triangle (a, b, p): >0 left of a->b, <0 right."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _side(q1, q2, p1), _side(q1, q2, p2)
    d3, d4 = _side(p1, p2, q1), _side(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass(slots=True)
class CountingLine:
    """One named line that vehicles are counted across."""

    name: str
    p1: Point
    p2: Point
    forward: int = 0  # crossings left -> right of the p1->p2 vector
    backward: int = 0

    @property
    def total(self) -> int:
        return self.forward + self.backward

    def crossing(self, prev: Point, curr: Point) -> str | None:
        """``"forward"`` / ``"backward"`` when the move crosses this line."""
        if not _segments_intersect(prev, curr, self.p1, self.p2):
            return None
        return "forward" if _side(self.p1, self.p2, curr) > 0 else "backward"


@dataclass(slots=True)
class LineCounter:
    """Tracks crossings for a set of named lines."""

    lines: list[CountingLine] = field(default_factory=list)
    _last: dict[int, Point] = field(default_factory=dict, repr=False)
    #: (track_id, line name) pairs already counted, so one pass counts once
    _seen: set[tuple[int, str]] = field(default_factory=set, repr=False)

    @classmethod
    def from_config(cls, lines: dict[str, list[list[float]]]) -> LineCounter:
        return cls(
            lines=[
                CountingLine(name, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                for name, (a, b) in (lines or {}).items()
            ]
        )

    def update(self, track_id: int, center: Point) -> list[str]:
        """Feed one track position; returns the crossings that just happened.

        Each entry is ``"<line>:<direction>"``, e.g. ``"north_gate:forward"``.
        """
        prev = self._last.get(track_id)
        self._last[track_id] = center
        if prev is None:
            return []
        events: list[str] = []
        for line in self.lines:
            key = (track_id, line.name)
            if key in self._seen:
                continue
            direction = line.crossing(prev, center)
            if direction is None:
                continue
            self._seen.add(key)
            if direction == "forward":
                line.forward += 1
            else:
                line.backward += 1
            events.append(f"{line.name}:{direction}")
        return events

    def forget(self, track_id: int) -> None:
        self._last.pop(track_id, None)

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            line.name: {
                "forward": line.forward,
                "backward": line.backward,
                "total": line.total,
            }
            for line in self.lines
        }


def dominant_direction(points: list[Point], min_displacement: float = 12.0) -> str | None:
    """Compass-ish direction of travel from a trajectory.

    Returns one of ``N``, ``NE``, ``E``, ``SE``, ``S``, ``SW``, ``W``, ``NW``
    (image coordinates: +y is down, so "N" means moving up the frame), or
    ``None`` when the track barely moved.
    """
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    if (dx * dx + dy * dy) ** 0.5 < min_displacement:
        return None
    import math

    # atan2(-dy, dx): flip y so that "up the frame" is north.
    angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
    labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    return labels[int((angle + 22.5) % 360.0 // 45.0)]
