"""OCR engine protocol and registry.

An *engine* is anything that turns a plate crop into text. PelakX never talks
to a specific OCR library directly — it asks the registry for the first
available engine in the country's preferred chain::

    IR -> [hezar_fa] then [paddle, easyocr, fast_plate]
    GB -> [fast_plate] then [paddle, easyocr]

That indirection is what makes "support my country's script" a ~40-line file
instead of a fork:

    from pelakx.ocr import BaseOcrEngine, register

    class MyScriptOcr(BaseOcrEngine):
        id = "my_script"
        def _read(self, crop, spec=None):
            return RawRead(text=..., confidence=..., engine=self.id)

    register(MyScriptOcr)

See ``docs/ADDING_A_COUNTRY.md`` for the full walkthrough.
"""

from __future__ import annotations

import abc
import contextlib
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from pelakx.types import RawRead

if TYPE_CHECKING:  # pragma: no cover
    from pelakx.grammar.spec import CountrySpec


class EngineUnavailable(RuntimeError):
    """Raised when an engine's optional dependency or weights are missing."""


class BaseOcrEngine(abc.ABC):
    """Base class for every OCR engine.

    Subclasses implement :meth:`_read` (and usually :meth:`_load`). The base
    class handles lazy loading, timing, and turning exceptions into ``None``
    so one flaky crop can never kill a long video run.
    """

    #: registry key, e.g. ``"hezar_fa"``
    id: ClassVar[str] = "base"
    #: human-readable name shown by ``pelakx engines``
    label: ClassVar[str] = "Base engine"
    #: scripts this engine can read; ``("*",)`` means anything
    scripts: ClassVar[tuple[str, ...]] = ("*",)
    #: pip extra that provides it, e.g. ``"pelakx[fa]"``
    install_hint: ClassVar[str] = "pip install pelakx[all]"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._model: Any = None
        self._loaded = False
        self._failures = 0

    # -- lifecycle ----------------------------------------------------------
    @classmethod
    def is_available(cls) -> bool:
        """True when this engine's dependencies can be imported."""
        try:
            return cls._probe()
        except Exception:
            return False

    @classmethod
    def _probe(cls) -> bool:
        """Cheap import check. Override in subclasses."""
        return True

    def load(self) -> None:
        """Load weights. Safe to call repeatedly; the first call does the work."""
        if self._loaded:
            return
        self._model = self._load()
        self._loaded = True

    def _load(self) -> Any:
        """Build and return the underlying model object."""
        return None

    def warmup(self, size: tuple[int, int] = (64, 192)) -> None:
        """Load and run once on a blank image so the first real frame is fast."""
        self.load()
        # Warmup must never raise: it runs before the first real frame.
        with contextlib.suppress(Exception):
            self._read(np.zeros((*size, 3), dtype=np.uint8), None)

    # -- inference ----------------------------------------------------------
    @abc.abstractmethod
    def _read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        """Engine-specific inference. Return None when nothing was read."""

    def read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        """Read `crop`, returning a :class:`RawRead` or ``None``.

        Never raises: a failing engine degrades to "no reading" and increments
        :attr:`failures`, which ``pelakx doctor`` surfaces.
        """
        if crop is None or crop.size == 0:
            return None
        self.load()
        started = time.perf_counter()
        try:
            result = self._read(crop, spec)
        except Exception:
            self._failures += 1
            return None
        if result is None or not result.text.strip():
            return None
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.engine = result.engine or self.id
        return result

    @property
    def failures(self) -> int:
        return self._failures

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "loaded" if self._loaded else "lazy"
        return f"<{type(self).__name__} id={self.id!r} {state}>"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_ENGINES: dict[str, type[BaseOcrEngine]] = {}
_INSTANCES: dict[tuple[str, frozenset], BaseOcrEngine] = {}


def register(engine_cls: type[BaseOcrEngine]) -> type[BaseOcrEngine]:
    """Register an engine class. Usable as a decorator."""
    if not engine_cls.id or engine_cls.id == "base":
        raise ValueError(f"{engine_cls.__name__} must define a unique `id`")
    _ENGINES[engine_cls.id] = engine_cls
    return engine_cls


def registered() -> dict[str, type[BaseOcrEngine]]:
    _ensure_builtins()
    return dict(_ENGINES)


def available() -> list[str]:
    """Ids of every registered engine whose dependencies are importable."""
    return sorted(eid for eid, cls in registered().items() if cls.is_available())


def get(engine_id: str, **options: Any) -> BaseOcrEngine:
    """Fetch (and cache) an engine instance by id."""
    _ensure_builtins()
    cls = _ENGINES.get(engine_id)
    if cls is None:
        raise KeyError(f"unknown OCR engine {engine_id!r}. Registered: {sorted(_ENGINES)}")
    if not cls.is_available():
        raise EngineUnavailable(
            f"OCR engine {engine_id!r} ({cls.label}) is not installed. "
            f"Install it with: {cls.install_hint}"
        )
    key = (engine_id, frozenset(options.items()))
    instance = _INSTANCES.get(key)
    if instance is None:
        instance = cls(**options)
        _INSTANCES[key] = instance
    return instance


def resolve(spec: CountrySpec, *, override: str | None = None, **options: Any) -> BaseOcrEngine:
    """Pick the best available engine for a country.

    Walks ``spec.engine_chain`` and returns the first installed engine. Raises
    :class:`EngineUnavailable` with an actionable message when none are.
    """
    if override:
        return get(override, **options)
    chain = spec.engine_chain or ("fast_plate",)
    tried: list[str] = []
    for engine_id in chain:
        _ensure_builtins()
        cls = _ENGINES.get(engine_id)
        if cls is None:
            tried.append(f"{engine_id} (not registered)")
            continue
        if cls.is_available():
            return get(engine_id, **options)
        tried.append(f"{engine_id} (needs: {cls.install_hint})")
    raise EngineUnavailable(
        f"no OCR engine available for {spec.code} ({spec.name_en}). Tried:\n  " + "\n  ".join(tried)
    )


def clear_instances() -> None:
    """Drop cached engine instances (frees model memory)."""
    _INSTANCES.clear()


_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    """Import the shipped engines once, so registration happens on demand."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from pelakx.ocr import engines  # noqa: F401  (import registers them)
