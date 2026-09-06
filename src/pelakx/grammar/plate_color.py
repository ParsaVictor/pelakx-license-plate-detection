"""Plate background-colour classification (Iran-oriented, HSV histogram).

Iranian plates are printed on a small number of standard backgrounds and the
background carries real meaning, same idea as many other countries:

* **white** — ordinary private vehicles (the overwhelming majority of plates).
* **yellow** — public-transport / commercial use: taxis, some vans (the
  ``ت`` taxi letter class in :mod:`pelakx.grammar` overlaps heavily with
  yellow plates, but the colour is decided by the vehicle's registered use,
  not the letter, so we classify colour independently of the letter).
* **red** — government/official vehicles in many countries; in Iran this is
  less standardised than white/yellow and is reported inconsistently across
  sources, so treat a ``red`` classification here as a low-confidence hint,
  not a certified category.
* **green** — police vehicles (verified against a real photographed sample,
  see ``configs/countries/ir.yaml``'s letter_semantics header).
* **blue** — diplomatic/political vehicles (also photo-verified).
* **brown** — historical/vintage plates (پلاک تاریخی), photo-verified; a
  structurally different layout (province name + code, not the usual
  digit-letter-digit slots) that PelakX does not attempt to parse today.
* **black** — included so the classifier degrades to a label instead of
  silently guessing "white"; no verified Iranian category found for it.

This module is deliberately simple and fast (a plain HSV histogram over the
plate crop, no model, no training data) so it can run per-crop on CPU with no
measurable cost. It is a heuristic, not a certified classifier: treat its
output as a hint to display next to the OCR text, not as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: (name, semantic hint) — order matters: checked most-specific-first below.
PlateColorName = str

#: rough real-world meaning, for the notebook / CLI to print alongside the
#: colour. Simplification — see module docstring.
COLOR_SEMANTICS: dict[str, dict[str, str]] = {
    "white": {"en": "Private / personal", "fa": "شخصی"},
    "yellow": {"en": "Taxi / public transport / commercial", "fa": "تاکسی/حمل‌ونقل عمومی"},
    "red": {"en": "Government / protocol", "fa": "دولتی/تشریفات"},
    "green": {"en": "Police", "fa": "پلیس"},
    "blue": {"en": "Diplomatic / political", "fa": "دیپلمات/سیاسی"},
    "brown": {"en": "Historical / vintage", "fa": "تاریخی"},
    "black": {"en": "No verified Iranian category", "fa": "دسته‌ی تأییدشده‌ای ندارد"},
    "unknown": {"en": "Unclassified", "fa": "نامشخص"},
}

# BGR display swatches for the notebook / annotator legend.
COLOR_SWATCH_BGR: dict[str, tuple[int, int, int]] = {
    "white": (230, 230, 230),
    "yellow": (0, 210, 255),
    "red": (40, 40, 220),
    "green": (60, 160, 60),
    "blue": (200, 100, 20),
    "brown": (35, 65, 110),
    "black": (30, 30, 30),
    "unknown": (150, 150, 150),
}


@dataclass(slots=True)
class ColorResult:
    name: PlateColorName
    confidence: float  # fraction of background pixels matching `name`
    en: str
    fa: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ColorResult({self.name}, conf={self.confidence:.2f})"


# HSV thresholds tuned for 8-bit OpenCV HSV (H:0-179, S/V:0-255).
# Order matters: colours are tested most-saturated/most-specific first so a
# strongly yellow plate is never mistaken for a slightly warm white.
_HSV_RANGES: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] = [
    ("red_lo", (0, 70, 60), (9, 255, 255)),
    ("red_hi", (170, 70, 60), (179, 255, 255)),
    # Sits in the narrow gap between red and yellow — measured directly off
    # a real "پلاک تاریخی" (historical) plate photo: median H=15, S=250,
    # V=101. Checked before "yellow" so a genuinely brown plate isn't
    # swallowed by yellow's much wider, brighter range.
    ("brown", (10, 120, 40), (17, 255, 160)),
    ("yellow", (18, 60, 80), (35, 255, 255)),
    ("green", (36, 40, 40), (85, 255, 255)),
    ("blue", (90, 40, 40), (130, 255, 255)),
    ("black", (0, 0, 0), (179, 90, 60)),
    ("white", (0, 0, 140), (179, 60, 255)),
]
_LABEL_MAP = {"red_lo": "red", "red_hi": "red"}


def _border_mask(h: int, w: int, margin: float = 0.14) -> np.ndarray:
    """Exclude the character strokes / plate frame roughly: keep the
    background-dominated border ring where digits rarely fall, plus centre
    strip minus text baseline band. Cheap approximation, not a segmenter."""
    mask = np.zeros((h, w), dtype=np.uint8)
    my = max(1, int(h * margin))
    mx = max(1, int(w * margin * 0.5))
    mask[:my, :] = 1
    mask[-my:, :] = 1
    mask[:, :mx] = 1
    mask[:, -mx:] = 1
    return mask


def classify_plate_color(crop_bgr: np.ndarray, *, use_border_only: bool = True) -> ColorResult:
    """Classify the dominant background colour of a plate crop.

    Args:
        crop_bgr: the plate region (already cropped, ideally rectified).
        use_border_only: sample the border ring (dominated by background,
            since digits sit in the middle) instead of the whole crop, which
            is more robust to dark plate text skewing the histogram toward
            "black". Falls back to the whole crop if it is too small.

    Returns:
        A :class:`ColorResult` with the best-matching colour name, the
        fraction of sampled pixels that matched it (as a rough confidence),
        and the (simplified, unverified) semantic hint in English/Farsi.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return ColorResult("unknown", 0.0, **COLOR_SEMANTICS["unknown"])

    h, w = crop_bgr.shape[:2]
    if use_border_only and h >= 12 and w >= 40:
        mask = _border_mask(h, w)
    else:
        mask = np.ones((h, w), dtype=np.uint8)

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sampled = hsv[mask.astype(bool)]
    if sampled.size == 0:
        sampled = hsv.reshape(-1, 3)
    total = len(sampled)

    scores: dict[str, int] = {}
    for label, lo, hi in _HSV_RANGES:
        lo_a = np.array(lo, dtype=np.uint8)
        hi_a = np.array(hi, dtype=np.uint8)
        within = np.all((sampled >= lo_a) & (sampled <= hi_a), axis=1)
        name = _LABEL_MAP.get(label, label)
        scores[name] = scores.get(name, 0) + int(within.sum())

    if not scores or max(scores.values()) == 0:
        return ColorResult("unknown", 0.0, **COLOR_SEMANTICS["unknown"])

    best_name = max(scores, key=lambda k: scores[k])
    confidence = scores[best_name] / total
    sem = COLOR_SEMANTICS.get(best_name, COLOR_SEMANTICS["unknown"])
    return ColorResult(best_name, float(confidence), **sem)
