"""PelakX — Multilingual License Plate Intelligence.

Turn any video stream into searchable, multilingual traffic intelligence:
detect vehicles, track them, read their plates in the right script, validate
the reading against the country's real plate grammar, and export the result.

Quick start
-----------
    from pelakx import Pipeline, PipelineConfig

    pipe = Pipeline(PipelineConfig(country="IR"))
    for event in pipe.run("traffic.mp4"):
        print(event.plate.display, event.plate.confidence)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "BBox",
    "Detection",
    "PlateRead",
    "RawRead",
    "TrackRecord",
    "VehicleEvent",
    "CountrySpec",
    "registry",
]

from pelakx.grammar import registry
from pelakx.grammar.spec import CountrySpec
from pelakx.types import BBox, Detection, PlateRead, RawRead, TrackRecord, VehicleEvent


def __getattr__(name: str):  # pragma: no cover - lazy heavy imports
    """Import the pipeline lazily so `import pelakx` stays dependency-light."""
    if name in {"Pipeline", "PipelineConfig"}:
        from pelakx import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module 'pelakx' has no attribute {name!r}")
