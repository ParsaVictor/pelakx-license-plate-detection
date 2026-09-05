"""Speed estimation via ground-plane homography.

Pixel distance is not distance: a car near the camera moves many more pixels
per second than the same car far away. PelakX therefore asks for four image
points and the real-world metres they correspond to (a rectangle painted on
the road, lane markings, a known-length stretch of kerb) and maps every track
centre onto that plane before differentiating.

Calibrate once per camera::

    analytics:
      speed_image_points: [[420, 640], [880, 640], [1180, 980], [140, 980]]
      speed_world_points: [[0, 0], [7.2, 0], [7.2, 25.0], [0, 25.0]]   # metres

Without calibration, speed is simply not reported — an uncalibrated number
would be worse than no number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class SpeedEstimator:
    """Maps image points to a metric ground plane and differentiates them."""

    def __init__(
        self,
        image_points: list[list[float]] | None,
        world_points: list[list[float]] | None,
        *,
        min_frames: int = 8,
        smoothing: float = 0.35,
        max_kmh: float = 300.0,
    ) -> None:
        self.matrix: np.ndarray | None = None
        self.min_frames = min_frames
        self.smoothing = smoothing
        self.max_kmh = max_kmh
        self._history: dict[int, list[tuple[float, float, float]]] = {}
        self._smoothed: dict[int, float] = {}
        if image_points and world_points and len(image_points) == len(world_points) >= 4:
            import cv2

            src = np.asarray(image_points, dtype=np.float32)
            dst = np.asarray(world_points, dtype=np.float32)
            self.matrix, _ = cv2.findHomography(src, dst)

    @property
    def calibrated(self) -> bool:
        return self.matrix is not None

    def to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        """Project an image point onto the ground plane, in metres."""
        if self.matrix is None:
            return point
        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        out = self.matrix @ vec
        if abs(out[2]) < 1e-9:
            return point
        return float(out[0] / out[2]), float(out[1] / out[2])

    def update(self, track_id: int, center: tuple[float, float], timestamp: float) -> float | None:
        """Feed one observation; returns the current speed estimate in km/h."""
        if self.matrix is None:
            return None
        wx, wy = self.to_world(center)
        history = self._history.setdefault(track_id, [])
        history.append((timestamp, wx, wy))
        if len(history) > 90:
            del history[:-90]
        if len(history) < self.min_frames:
            return None

        # Use a window rather than adjacent frames: detector jitter of a few
        # centimetres over 33 ms is a huge instantaneous speed error.
        t0, x0, y0 = history[0]
        t1, x1, y1 = history[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        metres = float(np.hypot(x1 - x0, y1 - y0))
        kmh = metres / dt * 3.6
        if not np.isfinite(kmh) or kmh > self.max_kmh:
            return None

        previous = self._smoothed.get(track_id)
        value = kmh if previous is None else previous + self.smoothing * (kmh - previous)
        self._smoothed[track_id] = value
        return round(value, 1)

    def get(self, track_id: int) -> float | None:
        value = self._smoothed.get(track_id)
        return round(value, 1) if value is not None else None

    def forget(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._smoothed.pop(track_id, None)


@dataclass(slots=True)
class SpeedStats:
    """Rolling aggregate of every speed PelakX has estimated."""

    values: list[float] = field(default_factory=list)
    limit_kmh: float | None = None

    def add(self, kmh: float | None) -> None:
        if kmh is not None and kmh > 0:
            self.values.append(float(kmh))

    def summary(self) -> dict[str, float | int]:
        if not self.values:
            return {"count": 0}
        arr = np.asarray(self.values)
        out: dict[str, float | int] = {
            "count": int(arr.size),
            "mean_kmh": round(float(arr.mean()), 1),
            "median_kmh": round(float(np.median(arr)), 1),
            "p85_kmh": round(float(np.percentile(arr, 85)), 1),
            "max_kmh": round(float(arr.max()), 1),
        }
        if self.limit_kmh:
            out["over_limit"] = int((arr > self.limit_kmh).sum())
        return out
