"""Runtime/device resolution.

``device: auto`` (the default) picks the best accelerator that is actually
present, so the same config file runs on a GPU workstation and on a fanless
camera box without editing:

    CUDA GPU present   -> "cuda:0"
    Apple Silicon      -> "mps"
    otherwise          -> "cpu"

Anything else you write in the config is passed through untouched, so
``cuda:1``, ``0,1`` or a forced ``cpu`` all still work.

The probe is cached: it imports torch once and never again, which matters
because it is called from every detector's constructor.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache

#: values that mean "work it out for me"
AUTO = frozenset({"auto", "", "default", None})


@lru_cache(maxsize=1)
def _probe() -> tuple[str, str]:
    """Return (device, human-readable reason). Cached for the process."""
    override = os.environ.get("PELAKX_DEVICE", "").strip()
    if override:
        return override, f"PELAKX_DEVICE={override}"

    if importlib.util.find_spec("torch") is None:
        return "cpu", "torch not installed"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - broken install
        return "cpu", f"torch import failed ({type(exc).__name__})"

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda:0", f"CUDA GPU: {name}"
    except Exception:  # pragma: no cover - driver weirdness
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple Silicon (MPS)"
    except Exception:  # pragma: no cover
        pass
    return "cpu", "no GPU detected"


def resolve_device(device: str | None = "auto") -> str:
    """Turn a configured device string into a concrete one.

    >>> resolve_device("auto")   # on a CPU-only box
    'cpu'
    >>> resolve_device("cuda:1") # explicit values pass through
    'cuda:1'
    """
    if device is None or str(device).strip().lower() in AUTO:
        return _probe()[0]
    return str(device).strip()


def describe_device(device: str | None = "auto") -> str:
    """One line explaining what `device` resolved to and why."""
    resolved = resolve_device(device)
    if device is None or str(device).strip().lower() in AUTO:
        return f"{resolved}  (auto — {_probe()[1]})"
    return f"{resolved}  (explicit)"


def is_gpu(device: str | None = "auto") -> bool:
    resolved = resolve_device(device).lower()
    return resolved.startswith(("cuda", "mps")) or resolved.isdigit()


def onnx_providers(device: str | None = "auto") -> list[str]:
    """ONNX Runtime execution providers matching `device`, best first.

    Only providers that are actually registered in the installed
    onnxruntime build are returned, so this never asks for a provider that
    would fail at session creation.
    """
    try:
        import onnxruntime as ort

        installed = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime optional
        return []
    wanted: list[str] = []
    if is_gpu(device):
        wanted += ["CUDAExecutionProvider", "CoreMLExecutionProvider"]
    wanted += ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return [p for p in wanted if p in installed]


def clear_cache() -> None:
    """Forget the probe result (tests, or after installing a GPU driver)."""
    _probe.cache_clear()
