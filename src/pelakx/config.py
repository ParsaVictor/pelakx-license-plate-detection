"""Configuration objects for the PelakX pipeline.

Everything is a plain dataclass with working defaults, so ``Pipeline()`` runs
with no configuration at all. A YAML file can override any subset::

    pelakx run traffic.mp4 --config configs/default.yaml --country IR

Precedence: CLI flags > YAML file > dataclass defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class VehicleConfig:
    """Vehicle detector + tracker."""

    weights: str = "yolo26n.pt"
    conf: float = 0.30
    iou: float = 0.50
    imgsz: int = 640
    #: "auto" picks CUDA -> MPS -> CPU; or force "cpu", "cuda:0", "cuda:1", "mps"
    device: str = "auto"
    classes: tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck
    tracker: str = "bytetrack.yaml"
    half: bool = False
    #: run plate detection on the whole frame instead of inside vehicle boxes.
    #: Cheaper when the camera is close and plates are large.
    standalone_plates: bool = False


@dataclass(slots=True)
class PlateConfig:
    """Plate detector."""

    weights: str | None = None  # None -> auto-select (see detect.plate.build)
    #: ONNX detector name when `weights` is None. Bigger input = smaller
    #: plates found, at a proportional CPU cost. See detect.plate.OnnxPlateDetector.
    model: str | None = None
    conf: float = 0.25
    imgsz: int = 640
    device: str = "auto"
    #: pad the vehicle box before searching for a plate inside it
    vehicle_pad: float = 0.02
    #: skip vehicles smaller than this (px, longest side). A 60px car cannot
    #: contain a readable plate, and upscaling it into the detector's 384px
    #: input reliably manufactures false positives.
    min_vehicle_px: int = 96
    #: reject plate candidates outside this width/height range before OCR
    aspect_range: tuple[float, float] = (1.2, 8.0)


@dataclass(slots=True)
class OcrConfig:
    """OCR stage."""

    engine: str | None = None  # None -> the country's preferred chain
    options: dict[str, Any] = field(default_factory=dict)
    #: only OCR a track this many times per second of video (0 = every frame)
    max_reads_per_second: float = 6.0
    #: stop OCR-ing a track once consensus is this confident
    early_stop_confidence: float = 0.97
    #: minimum votes before a track is allowed to early-stop
    early_stop_min_votes: int = 4
    #: how many confusion substitutions the grammar may apply per reading
    repair_budget: int = 2


@dataclass(slots=True)
class QualityConfig:
    """Crop gating and preparation."""

    enabled: bool = True
    min_width: int = 48
    min_height: int = 14
    target_width: int = 160
    min_sharpness: float = 18.0
    good_sharpness: float = 140.0
    aspect_min: float = 1.4
    aspect_max: float = 7.0
    min_score: float = 0.25
    rectify: bool = True
    enhance: bool = True
    ocr_height: int = 64


@dataclass(slots=True)
class AnalyticsConfig:
    """Counting lines, speed estimation, watchlist."""

    #: named counting lines: {"north_gate": [[x1, y1], [x2, y2]]} in pixels
    lines: dict[str, list[list[float]]] = field(default_factory=dict)
    #: 4 image points (px) mapping to `speed_world` metres, for speed estimation
    speed_image_points: list[list[float]] = field(default_factory=list)
    speed_world_points: list[list[float]] = field(default_factory=list)
    #: minimum track length (frames) before a speed estimate is emitted
    speed_min_frames: int = 8
    #: plate strings (or prefixes) to alert on
    watchlist: list[str] = field(default_factory=list)
    #: allow N character edits when matching the watchlist
    watchlist_max_distance: int = 1


@dataclass(slots=True)
class PrivacyConfig:
    """Privacy-preserving output. Off by default; one flag turns it all on."""

    #: blur every detected plate in the rendered video
    blur_plates: bool = False
    #: blur faces too (needs a face detector; skipped when unavailable)
    blur_faces: bool = False
    #: replace plate strings in exports with a salted hash
    hash_plates: bool = False
    #: salt for the hash; set it per-deployment and keep it secret
    hash_salt: str = ""
    #: do not write plate crops to disk
    no_crops: bool = False


@dataclass(slots=True)
class OutputConfig:
    """Where results go."""

    dir: str = "outputs"
    video: bool = True
    video_name: str = "annotated.mp4"
    csv: bool = True
    csv_name: str = "events.csv"
    jsonl: bool = False
    jsonl_name: str = "events.jsonl"
    sqlite: bool = True
    sqlite_name: str = "pelakx.sqlite"
    crops: bool = True
    crops_dir: str = "crops"
    #: draw the annotated frame with native-script plate text
    draw_labels: bool = True
    draw_tracks: bool = True
    font: str | None = None  # path to a Persian-capable TTF


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration."""

    country: str = "IR"
    #: process every Nth frame (2 roughly halves the cost on 30 fps footage)
    frame_stride: int = 1
    #: stop after N processed frames (0 = whole video)
    max_frames: int = 0
    #: seconds a track may go unseen before it is finalized and exported
    track_timeout: float = 2.0
    #: minimum consensus confidence for an event to be exported
    min_export_confidence: float = 0.35
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- (de)serialization --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return _build(cls, data or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    def merged(self, **overrides: Any) -> PipelineConfig:
        """Return a copy with dotted overrides applied.

        >>> cfg.merged(**{"vehicle.device": "cuda:0", "country": "GB"})
        """
        data = self.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            node = data
            *path, leaf = key.split(".")
            for part in path:
                node = node.setdefault(part, {})
            node[leaf] = value
        return type(self).from_dict(data)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively construct nested dataclasses from a plain dict.

    Unknown keys are ignored rather than raising, so a stale config file from
    an older PelakX still loads instead of crashing a running camera.
    """
    if not is_dataclass(cls):
        return data
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for name, value in (data or {}).items():
        if name not in known:
            continue
        nested = _NESTED.get((cls.__name__, name))
        kwargs[name] = _build(nested, value) if nested and isinstance(value, dict) else value
    return cls(**kwargs)


#: explicit map for nested config classes, because ``from __future__ import
#: annotations`` turns field types into strings that ``is_dataclass`` can't see.
_NESTED: dict[tuple[str, str], type] = {
    ("PipelineConfig", "vehicle"): VehicleConfig,
    ("PipelineConfig", "plate"): PlateConfig,
    ("PipelineConfig", "ocr"): OcrConfig,
    ("PipelineConfig", "quality"): QualityConfig,
    ("PipelineConfig", "analytics"): AnalyticsConfig,
    ("PipelineConfig", "privacy"): PrivacyConfig,
    ("PipelineConfig", "output"): OutputConfig,
}
