"""The OCR engines that ship with PelakX.

Every engine is optional. Nothing here is imported at package-import time —
:func:`pelakx.ocr.base._ensure_builtins` pulls this module in lazily, and each
engine only touches its third-party library inside ``_probe`` / ``_load``.

| id           | backend                              | best for                    |
|--------------|--------------------------------------|-----------------------------|
| ``fast_plate``| fast-plate-ocr (ONNX, ~13 ms CPU)    | Latin plates, 65+ countries |
| ``hezar_fa``  | hezarai CRNN-CTC                     | Persian / Iranian plates    |
| ``paddle``    | PaddleOCR PP-OCRv5                   | any script, heavier         |
| ``easyocr``   | EasyOCR                              | 80+ languages, heaviest     |
| ``echo``      | none                                 | tests and dry runs          |
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

import numpy as np

from pelakx.ocr.base import BaseOcrEngine, register
from pelakx.types import RawRead

if TYPE_CHECKING:  # pragma: no cover
    from pelakx.grammar.spec import CountrySpec


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - namespace weirdness
        return False


def _to_bgr(crop: np.ndarray) -> np.ndarray:
    """Ensure a 3-channel uint8 image."""
    import cv2

    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if crop.dtype != np.uint8:
        crop = np.clip(crop, 0, 255).astype(np.uint8)
    return crop


# ---------------------------------------------------------------------------
# fast-plate-ocr — the CPU-fast default for Latin-script plates
# ---------------------------------------------------------------------------
@register
class FastPlateOcr(BaseOcrEngine):
    """ONNX plate recognizer from ``fast-plate-ocr``.

    Options:
        model: hub model name. ``cct-xs-v2-global-model`` covers 65+ countries;
            ``cct-s-v2-global-model`` is larger and more accurate;
            ``european-plates-mobile-vit-v2-model`` is Europe-tuned.
        device: ``auto`` | ``cpu`` | ``cuda``.
    """

    id = "fast_plate"
    label = "fast-plate-ocr (ONNX)"
    scripts = ("latin",)
    install_hint = "pip install 'pelakx[onnx]'"

    DEFAULT_MODEL = "cct-xs-v2-global-model"

    @classmethod
    def _probe(cls) -> bool:
        return _installed("fast_plate_ocr")

    def _load(self) -> Any:
        from fast_plate_ocr import LicensePlateRecognizer

        return LicensePlateRecognizer(
            self.options.get("model", self.DEFAULT_MODEL),
            device=self.options.get("device", "auto"),
        )

    def _read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        crop = _to_bgr(crop)
        try:
            output = self._model.run(crop, return_confidence=True)
        except TypeError:  # older signature without the kwarg
            output = self._model.run(crop)

        text, char_confs, region = _unwrap_fast_plate(output)
        # fast-plate-ocr pads short plates with '_'
        text = text.replace("_", "").strip()
        if not text:
            return None

        confidence = (
            float(np.mean(char_confs))
            if char_confs
            else float(self.options.get("default_confidence", 0.85))
        )
        return RawRead(
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            engine=self.id,
            char_confidences=char_confs[: len(text)],
            region=region,
        )


def _unwrap_fast_plate(output: Any) -> tuple[str, list[float], str]:
    """Normalise every fast-plate-ocr return shape into (text, char_confs, region).

    1.1+ returns ``list[PlatePrediction]`` with ``.plate`` / ``.char_probs`` /
    ``.region``; earlier versions returned ``list[str]`` or a
    ``(list[str], ndarray)`` tuple.
    """
    confidences: Any = None
    if isinstance(output, tuple) and len(output) >= 2:
        output, confidences = output[0], output[1]
    item = output[0] if isinstance(output, (list, tuple)) and output else output
    if item is None:
        return "", [], ""

    region = str(getattr(item, "region", "") or "")
    text = getattr(item, "plate", None)
    if text is None:
        text = item if isinstance(item, str) else str(item)

    probs = getattr(item, "char_probs", None)
    if probs is None:
        probs = confidences
    char_confs: list[float] = []
    if probs is not None:
        arr = np.asarray(probs, dtype=float).reshape(-1)
        char_confs = [float(c) for c in arr]
    return str(text), char_confs, region


# ---------------------------------------------------------------------------
# Hezar CRNN — purpose-built for Iranian plates
# ---------------------------------------------------------------------------
@register
class HezarPersianOcr(BaseOcrEngine):
    """Persian license-plate CRNN-CTC recognizer from the Hezar model hub.

    Options:
        model: hub id, default ``hezarai/crnn-fa-license-plate-recognition-v2``.
    """

    id = "hezar_fa"
    label = "Hezar CRNN (Persian plates)"
    scripts = ("arabic",)
    install_hint = "pip install 'pelakx[fa]'"

    DEFAULT_MODEL = "hezarai/crnn-fa-license-plate-recognition-v2"

    @classmethod
    def _probe(cls) -> bool:
        return _installed("hezar")

    def _load(self) -> Any:
        from hezar.models import Model

        return Model.load(self.options.get("model", self.DEFAULT_MODEL))

    def _read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        crop = _to_bgr(crop)
        outputs = self._model.predict(crop)
        text, score = _unwrap_hezar(outputs)
        if not text:
            return None
        return RawRead(
            text=text,
            confidence=score if score is not None else 0.9,
            engine=self.id,
        )


def _unwrap_hezar(outputs: Any) -> tuple[str, float | None]:
    """Hezar's return shape varies by version; pull out (text, score)."""
    item = outputs
    while isinstance(item, (list, tuple)) and item:
        item = item[0]
    if isinstance(item, dict):
        text = item.get("text") or item.get("label") or ""
        score = item.get("score", item.get("confidence"))
        return str(text), float(score) if score is not None else None
    if hasattr(item, "text"):
        score = getattr(item, "score", None)
        return str(item.text), float(score) if score is not None else None
    return (str(item) if item is not None else ""), None


# ---------------------------------------------------------------------------
# PaddleOCR — general multi-script fallback
# ---------------------------------------------------------------------------
@register
class PaddleOcr(BaseOcrEngine):
    """PaddleOCR (PP-OCRv5 and newer) as a generic any-script fallback.

    Options:
        lang: PaddleOCR language code (``en``, ``arabic``, ``ch``, ``devanagari``…).
    """

    id = "paddle"
    label = "PaddleOCR (PP-OCRv5+)"
    scripts = ("*",)
    install_hint = "pip install 'pelakx[paddle]'"

    #: PelakX script name -> PaddleOCR language code
    SCRIPT_TO_LANG = {
        "arabic": "arabic",
        "latin": "en",
        "cyrillic": "cyrillic",
        "devanagari": "devanagari",
        "han": "ch",
    }

    @classmethod
    def _probe(cls) -> bool:
        return _installed("paddleocr")

    def _load(self) -> Any:
        from paddleocr import PaddleOCR

        lang = self.options.get("lang", "en")
        kwargs: dict[str, Any] = {"lang": lang}
        try:  # PaddleOCR 3.x
            return PaddleOCR(use_textline_orientation=False, **kwargs)
        except TypeError:  # PaddleOCR 2.x
            return PaddleOCR(use_angle_cls=False, show_log=False, **kwargs)

    def _read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        crop = _to_bgr(crop)
        texts, scores = _run_paddle(self._model, crop)
        if not texts:
            return None
        # A plate crop is one line; join in reading order and let the grammar
        # layer strip whatever separators the engine invented.
        text = "".join(texts)
        confidence = float(np.mean(scores)) if scores else 0.8
        return RawRead(text=text, confidence=confidence, engine=self.id)


def _run_paddle(model: Any, image: np.ndarray) -> tuple[list[str], list[float]]:
    """Normalise PaddleOCR 2.x / 3.x output into (texts, scores)."""
    if hasattr(model, "predict"):
        try:
            result = model.predict(image)
        except Exception:
            result = None
        if result:
            first = result[0]
            data = first if isinstance(first, dict) else getattr(first, "json", None) or {}
            if isinstance(data, dict):
                texts = list(data.get("rec_texts") or [])
                scores = [float(s) for s in (data.get("rec_scores") or [])]
                if texts:
                    return texts, scores
    # 2.x: [[ [box, (text, score)], ... ]]
    result = model.ocr(image)
    texts, scores = [], []
    for page in result or []:
        for line in page or []:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                payload = line[1]
                if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                    texts.append(str(payload[0]))
                    scores.append(float(payload[1]))
    return texts, scores


# ---------------------------------------------------------------------------
# EasyOCR — widest language coverage, slowest
# ---------------------------------------------------------------------------
@register
class EasyOcr(BaseOcrEngine):
    """EasyOCR fallback — 80+ languages, useful for bootstrapping a new country.

    Options:
        langs: list of EasyOCR language codes, default ``["en"]``.
        gpu: run on CUDA when available.
    """

    id = "easyocr"
    label = "EasyOCR"
    scripts = ("*",)
    install_hint = "pip install 'pelakx[easyocr]'"

    SCRIPT_TO_LANGS = {
        "arabic": ["fa", "ar"],
        "latin": ["en"],
        "cyrillic": ["ru"],
        "devanagari": ["hi"],
    }

    @classmethod
    def _probe(cls) -> bool:
        return _installed("easyocr")

    def _load(self) -> Any:
        import easyocr

        return easyocr.Reader(
            self.options.get("langs", ["en"]),
            gpu=bool(self.options.get("gpu", False)),
            verbose=False,
        )

    def _read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        crop = _to_bgr(crop)
        results = self._model.readtext(crop, detail=1, paragraph=False)
        if not results:
            return None
        texts = [str(r[1]) for r in results]
        scores = [float(r[2]) for r in results if len(r) > 2]
        return RawRead(
            text="".join(texts),
            confidence=float(np.mean(scores)) if scores else 0.7,
            engine=self.id,
        )


# ---------------------------------------------------------------------------
# echo — no dependencies, for tests and `--dry-run`
# ---------------------------------------------------------------------------
@register
class EchoOcr(BaseOcrEngine):
    """Returns a canned reading. Lets the whole pipeline be tested offline.

    Options:
        text: the string to return (default ``"12ب34511"``).
        confidence: the confidence to report.
    """

    id = "echo"
    label = "Echo (test double)"
    scripts = ("*",)
    install_hint = "built in"

    def _read(self, crop: np.ndarray, spec: CountrySpec | None = None) -> RawRead | None:
        return RawRead(
            text=str(self.options.get("text", "12ب34511")),
            confidence=float(self.options.get("confidence", 0.9)),
            engine=self.id,
        )
