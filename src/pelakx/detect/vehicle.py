"""Vehicle detection + multi-object tracking (YOLO26 by default).

Detection and tracking are one stage on purpose: Ultralytics' ``model.track``
runs ByteTrack / BoT-SORT inside the same forward pass bookkeeping, so a
separate tracker would mean re-associating boxes we already associated.

Why YOLO26 as the default backbone: it is the current Ultralytics generation
(Jan 2026), it is *natively end-to-end* — predictions come out without an NMS
pass — and on CPU ONNX it is roughly 43% faster than YOLO11n at comparable
accuracy. On a CPU-only machine, which is where most CCTV boxes actually live,
that is the whole ballgame. Any Ultralytics-compatible weights still work:
pass ``weights="yolo11n.pt"`` or your own fine-tune.
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from pelakx.detect.base import BaseDetector, DetectorUnavailable, boxes_from_ultralytics
from pelakx.runtime import resolve_device
from pelakx.types import Detection

#: COCO class ids that count as vehicles
COCO_VEHICLE_IDS: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

#: sensible default: cars, motorcycles, buses, trucks (bicycles have no plates)
DEFAULT_VEHICLE_CLASSES: tuple[int, ...] = (2, 3, 5, 7)


class VehicleDetector(BaseDetector):
    """YOLO-based vehicle detector with optional built-in tracking.

    Args:
        weights: Ultralytics weights. ``yolo26n.pt`` (fastest) through
            ``yolo26x.pt`` (most accurate), or any local ``.pt`` / ``.onnx``.
        conf: confidence threshold.
        iou: NMS IoU threshold (ignored by end-to-end YOLO26 heads).
        imgsz: inference resolution.
        device: ``auto`` (CUDA -> MPS -> CPU), or force ``cpu`` / ``cuda:0`` / ``mps``.
        classes: COCO class ids to keep; ``None`` keeps every vehicle class.
        tracker: Ultralytics tracker config — ``bytetrack.yaml`` (fast) or
            ``botsort.yaml`` (better through occlusions, slower).
        half: FP16 inference (CUDA only).
    """

    id = "yolo_vehicle"
    label = "Ultralytics YOLO (vehicles)"
    install_hint = "pip install 'pelakx[detect]'"

    DEFAULT_WEIGHTS = "yolo26n.pt"

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        conf: float = 0.30,
        iou: float = 0.50,
        imgsz: int = 640,
        device: str = "auto",
        classes: tuple[int, ...] | None = DEFAULT_VEHICLE_CLASSES,
        tracker: str = "bytetrack.yaml",
        half: bool = False,
        verbose: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = resolve_device(device)
        self.classes = list(classes) if classes else None
        self.tracker = tracker
        self.half = half
        self.verbose = verbose

    @classmethod
    def _probe(cls) -> bool:
        return importlib.util.find_spec("ultralytics") is not None

    def _load(self) -> Any:
        if not self.is_available():
            raise DetectorUnavailable(f"ultralytics is not installed. {self.install_hint}")
        from ultralytics import YOLO

        return YOLO(self.weights)

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "classes": self.classes,
            "verbose": self.verbose,
        }
        # `half` is deprecated in recent Ultralytics and warns on every call,
        # so only pass it when it is actually requested.
        if self.half:
            kwargs["half"] = True
        return kwargs

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Stateless detection — no track ids."""
        self.load()
        results = self._model.predict(frame, **self._predict_kwargs())
        return boxes_from_ultralytics(results[0]) if results else []

    def track(self, frame: np.ndarray, *, persist: bool = True) -> list[Detection]:
        """Detection **with** stable track ids across calls.

        Ultralytics keeps tracker state on the model, so frames must be fed in
        order. Call :meth:`reset` between videos.
        """
        self.load()
        results = self._model.track(
            frame, persist=persist, tracker=self.tracker, **self._predict_kwargs()
        )
        return boxes_from_ultralytics(results[0]) if results else []

    def reset(self) -> None:
        """Drop tracker state so the next video starts from track id 1."""
        if not self._loaded:
            return
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        for tracker in trackers or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def warmup(self, size: tuple[int, int] = (640, 640)) -> None:
        """Run one dummy pass through the *tracking* code path, not `.predict`.

        Ultralytics keeps a mode-specific `self._model.predictor` on the
        model instance, built the first time either `.predict()` or
        `.track()` is called. The base class's default warmup always calls
        `.detect()` (`.predict()`), so on `Pipeline.warmup()` -> real frame,
        the very first `.track()` call of the run is *also* the first call
        that ever builds the tracking predictor — a cold-start that can
        return a subtly different box/track split than a second or later
        `.track()` call would (observed switching a vehicle's plate-detector
        input region by tens of pixels on an otherwise identical frame).
        Warming up through `.track()` instead means the real first frame is
        never the tracker's first call; `reset()` afterwards drops the
        dummy detection's track ids so real ids still start at 1.
        """
        self.load()
        with contextlib.suppress(Exception):
            self.track(np.zeros((*size, 3), dtype=np.uint8), persist=True)
        self.reset()
