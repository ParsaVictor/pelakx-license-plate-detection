"""License-plate localisation.

Two interchangeable backends ship with PelakX:

``UltralyticsPlateDetector``
    Any YOLO ``.pt``/``.onnx`` fine-tuned on plates — including one you train
    yourself with ``pelakx train-plate``. Highest ceiling.

``OnnxPlateDetector``
    ``open-image-models``' ``yolo-v9-t-384-license-plate-end2end`` — a tiny
    end-to-end ONNX model that needs no weights file on disk and runs in a few
    milliseconds on CPU. This is what makes ``pelakx run`` work the moment the
    package is installed, with nothing to download by hand.

:func:`build` picks the best backend that is actually available and tells you
exactly what to install when none are.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from pelakx.detect.base import BaseDetector, DetectorUnavailable, boxes_from_ultralytics
from pelakx.types import BBox, Detection


class UltralyticsPlateDetector(BaseDetector):
    """Plate detector backed by Ultralytics weights."""

    id = "yolo_plate"
    label = "Ultralytics YOLO (plates)"
    install_hint = "pip install 'pelakx[detect]'"

    def __init__(
        self,
        weights: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        verbose: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.verbose = verbose

    @classmethod
    def _probe(cls) -> bool:
        return importlib.util.find_spec("ultralytics") is not None

    def _load(self) -> Any:
        if not self.is_available():
            raise DetectorUnavailable(f"ultralytics is not installed. {self.install_hint}")
        if not Path(self.weights).exists() and not self.weights.startswith("yolo"):
            raise DetectorUnavailable(
                f"plate weights not found: {self.weights}\n"
                f"Fetch a ready-made detector with:  pelakx download-models"
            )
        from ultralytics import YOLO

        return YOLO(self.weights)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.load()
        results = self._model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=self.verbose,
        )
        if not results:
            return []
        detections = boxes_from_ultralytics(results[0])
        for d in detections:
            d.class_name = "plate"
        return detections


class OnnxPlateDetector(BaseDetector):
    """Zero-setup ONNX plate detector from ``open-image-models``."""

    id = "onnx_plate"
    label = "open-image-models YOLOv9-t (ONNX, end-to-end)"
    install_hint = "pip install 'pelakx[onnx]'"

    DEFAULT_MODEL = "yolo-v9-t-384-license-plate-end2end"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        conf: float = 0.25,
        device: str = "auto",
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.model_name = model
        self.conf = conf
        self.device = device

    @classmethod
    def _probe(cls) -> bool:
        return importlib.util.find_spec("open_image_models") is not None

    def _load(self) -> Any:
        if not self.is_available():
            raise DetectorUnavailable(f"open-image-models is not installed. {self.install_hint}")
        import open_image_models

        # 0.6 renamed the constructor and deprecated the old class.
        factory = getattr(open_image_models, "create_detector", None)
        if factory is not None:
            try:
                return factory(detection_model=self.model_name, conf_thresh=self.conf)
            except TypeError:
                return factory(self.model_name, conf_thresh=self.conf)
        return open_image_models.LicensePlateDetector(
            detection_model=self.model_name, conf_thresh=self.conf
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.load()
        out: list[Detection] = []
        for item in self._model.predict(frame) or []:
            box = getattr(item, "bounding_box", None)
            if box is None:
                continue
            out.append(
                Detection(
                    bbox=BBox(float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
                    confidence=float(getattr(item, "confidence", 0.0)),
                    class_id=0,
                    class_name="plate",
                )
            )
        return out


def build(
    weights: str | Path | None = None,
    *,
    conf: float = 0.25,
    device: str = "cpu",
    imgsz: int = 640,
    **options: Any,
) -> BaseDetector:
    """Return the best available plate detector.

    Preference order:
      1. explicit `weights` (Ultralytics),
      2. ``models/license_plate.pt`` if it exists,
      3. the zero-setup ONNX detector,
      4. a clear error listing what to install.
    """
    if weights:
        return UltralyticsPlateDetector(weights, conf=conf, device=device, imgsz=imgsz, **options)

    local = Path(__file__).resolve().parents[3] / "models" / "license_plate.pt"
    if local.exists() and UltralyticsPlateDetector.is_available():
        return UltralyticsPlateDetector(local, conf=conf, device=device, imgsz=imgsz, **options)

    if OnnxPlateDetector.is_available():
        return OnnxPlateDetector(conf=conf, device=device, **options)

    raise DetectorUnavailable(
        "no plate detector available. Pick one:\n"
        "  pip install 'pelakx[onnx]'    # zero-setup ONNX detector (recommended)\n"
        "  pelakx download-models        # fetch YOLO plate weights into models/\n"
        "  pelakx run ... --plate-weights /path/to/your.pt"
    )
