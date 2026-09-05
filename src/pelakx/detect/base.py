"""Detector protocol shared by the vehicle and plate stages."""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import numpy as np

from pelakx.types import BBox, Detection


class DetectorUnavailable(RuntimeError):
    """Raised when a detector's dependency or weights file is missing."""


class BaseDetector(abc.ABC):
    """Anything that turns a frame into a list of :class:`Detection`."""

    id: ClassVar[str] = "base"
    label: ClassVar[str] = "Base detector"
    install_hint: ClassVar[str] = "pip install 'pelakx[detect]'"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._model: Any = None
        self._loaded = False

    @classmethod
    def is_available(cls) -> bool:
        try:
            return cls._probe()
        except Exception:
            return False

    @classmethod
    def _probe(cls) -> bool:
        return True

    def load(self) -> None:
        if not self._loaded:
            self._model = self._load()
            self._loaded = True

    def _load(self) -> Any:
        return None

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detections for a single frame, in absolute pixel coordinates."""

    def warmup(self, size: tuple[int, int] = (640, 640)) -> None:
        self.load()
        try:
            self.detect(np.zeros((*size, 3), dtype=np.uint8))
        except Exception:  # pragma: no cover
            pass

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {'loaded' if self._loaded else 'lazy'}>"


def boxes_from_ultralytics(result: Any, names: dict[int, str] | None = None) -> list[Detection]:
    """Convert one Ultralytics ``Results`` object into :class:`Detection` list."""
    out: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    names = names or getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        cls_id = int(classes[i])
        out.append(
            Detection(
                bbox=BBox(x1, y1, x2, y2),
                confidence=float(confs[i]),
                class_id=cls_id,
                class_name=str(names.get(cls_id, cls_id)),
                track_id=int(ids[i]) if ids is not None else None,
            )
        )
    return out
