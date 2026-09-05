"""Frame annotation with correct right-to-left text.

``cv2.putText`` cannot draw Persian: it has no Arabic glyphs, no contextual
letter shaping, and no bidirectional reordering. Drawing "12 ب 345 ایران 11"
with OpenCV produces boxes, or disconnected mirrored letters.

PelakX renders labels through PIL with a script-capable TrueType font, after
running the string through ``arabic-reshaper`` (join the letters) and
``python-bidi`` (reorder for display). Latin-only labels take the fast
OpenCV path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pelakx.grammar.normalize import has_rtl, shape_rtl
from pelakx.types import BBox

# --- palette (BGR) ---------------------------------------------------------
COLOR_VEHICLE = (255, 176, 59)
COLOR_PLATE = (66, 245, 152)
COLOR_PLATE_WEAK = (80, 190, 255)
COLOR_ALERT = (60, 60, 255)
COLOR_TEXT = (18, 18, 18)
COLOR_TRACK = (200, 200, 200)

#: font files that are known to carry Arabic/Persian glyphs, by platform
_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "win32": (
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ),
    "darwin": (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Geneva.ttf",
    ),
    "linux": (
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ),
}


def find_font(explicit: str | None = None) -> str | None:
    """Locate a TrueType font that can draw the labels.

    Order: explicit path -> ``assets/fonts/*.ttf`` in the repo -> platform
    fonts known to include Arabic glyphs. Returns None when nothing is found,
    in which case rendering falls back to OpenCV (Latin only).
    """
    if explicit and Path(explicit).is_file():
        return explicit
    bundled = Path(__file__).resolve().parents[3] / "assets" / "fonts"
    if bundled.is_dir():
        for pattern in ("*.ttf", "*.otf"):
            for path in sorted(bundled.glob(pattern)):
                return str(path)
    platform = "linux"
    if sys.platform.startswith("win"):
        platform = "win32"
    elif sys.platform == "darwin":
        platform = "darwin"
    for candidate in _FONT_CANDIDATES[platform]:
        if Path(candidate).is_file():
            return candidate
    return None


@lru_cache(maxsize=32)
def _pil_font(path: str, size: int):  # pragma: no cover - needs PIL + a font file
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


@dataclass(slots=True)
class Annotator:
    """Draws PelakX overlays onto frames.

    Args:
        font: path to a TTF; auto-detected when omitted.
        font_size: label text size in pixels.
        thickness: box line thickness.
    """

    font: str | None = None
    font_size: int = 18
    thickness: int = 2
    _font_path: str | None = None
    _pil_ok: bool = True

    def __post_init__(self) -> None:
        self._font_path = find_font(self.font)
        try:
            import PIL  # noqa: F401
        except ImportError:  # pragma: no cover - optional
            self._pil_ok = False

    @property
    def can_draw_rtl(self) -> bool:
        return self._pil_ok and self._font_path is not None

    # -- primitives ---------------------------------------------------------
    def box(
        self,
        frame: np.ndarray,
        bbox: BBox,
        color: tuple[int, int, int] = COLOR_VEHICLE,
        thickness: int | None = None,
    ) -> None:
        x1, y1, x2, y2 = bbox.as_int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness or self.thickness)

    def label(
        self,
        frame: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        *,
        color: tuple[int, int, int] = COLOR_PLATE,
        text_color: tuple[int, int, int] = COLOR_TEXT,
        above: bool = True,
    ) -> None:
        """Draw `text` in a filled pill anchored at `anchor` (a box corner)."""
        if not text:
            return
        if has_rtl(text) and self.can_draw_rtl:
            self._label_pil(frame, text, anchor, color, text_color, above)
        else:
            self._label_cv(frame, text, anchor, color, text_color, above)

    def _label_cv(self, frame, text, anchor, color, text_color, above) -> None:
        scale = self.font_size / 32.0
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = anchor
        pad = 4
        top = y - th - 2 * pad if above else y
        top = max(0, top)
        cv2.rectangle(
            frame, (x, top), (x + tw + 2 * pad, top + th + 2 * pad + baseline // 2), color, -1
        )
        cv2.putText(
            frame,
            text,
            (x + pad, top + th + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def _label_pil(self, frame, text, anchor, color, text_color, above) -> None:  # pragma: no cover
        from PIL import Image, ImageDraw

        shaped = shape_rtl(text)
        font = _pil_font(self._font_path, self.font_size)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        left, top_, right, bottom = draw.textbbox((0, 0), shaped, font=font)
        tw, th = right - left, bottom - top_
        x, y = anchor
        pad = 5
        top = max(0, y - th - 2 * pad) if above else y
        draw.rectangle(
            [x, top, x + tw + 2 * pad, top + th + 2 * pad],
            fill=(color[2], color[1], color[0]),
        )
        draw.text(
            (x + pad - left, top + pad - top_),
            shaped,
            font=font,
            fill=(text_color[2], text_color[1], text_color[0]),
        )
        frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def trail(
        self,
        frame: np.ndarray,
        points: list[tuple[float, float]],
        color: tuple[int, int, int] = COLOR_TRACK,
        max_points: int = 40,
    ) -> None:
        """Draw a fading trajectory tail behind a tracked vehicle."""
        pts = points[-max_points:]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            shade = tuple(int(c * (0.25 + 0.75 * alpha)) for c in color)
            cv2.line(
                frame,
                (int(pts[i - 1][0]), int(pts[i - 1][1])),
                (int(pts[i][0]), int(pts[i][1])),
                shade,
                max(1, int(1 + 2 * alpha)),
                cv2.LINE_AA,
            )

    def line(
        self,
        frame: np.ndarray,
        p1: tuple[float, float],
        p2: tuple[float, float],
        name: str = "",
        counts: str = "",
    ) -> None:
        """Draw a counting line and its tally."""
        a = (int(p1[0]), int(p1[1]))
        b = (int(p2[0]), int(p2[1]))
        cv2.line(frame, a, b, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, a, b, (0, 140, 255), 1, cv2.LINE_AA)
        if name:
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            self.label(
                frame,
                f"{name} {counts}".strip(),
                mid,
                color=(0, 140, 255),
                text_color=(255, 255, 255),
            )

    def hud(self, frame: np.ndarray, lines: list[str]) -> None:
        """Bottom-left status block: fps, counts, engine."""
        h = frame.shape[0]
        y = h - 10 - 18 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y - 8), (330, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (10, y + 14 + 18 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )


def plate_color(confidence: float, alerted: bool = False) -> tuple[int, int, int]:
    """Green for a confident read, amber for a shaky one, red for an alert."""
    if alerted:
        return COLOR_ALERT
    return COLOR_PLATE if confidence >= 0.7 else COLOR_PLATE_WEAK
