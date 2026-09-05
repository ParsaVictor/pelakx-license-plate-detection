"""OCR engines: a pluggable registry, not a hard-coded library call."""

from __future__ import annotations

from pelakx.ocr.base import (
    BaseOcrEngine,
    EngineUnavailable,
    available,
    clear_instances,
    get,
    register,
    registered,
    resolve,
)

__all__ = [
    "BaseOcrEngine",
    "EngineUnavailable",
    "available",
    "clear_instances",
    "get",
    "register",
    "registered",
    "resolve",
]
