"""Quality gating and plate-crop preparation.

Running OCR on every plate box of every frame is the single biggest waste of
CPU in a naive ALPR pipeline — most crops are too small, too blurry or too
skewed to ever produce a correct reading, and they poison the temporal vote
when they do produce one.

The gate scores a crop *before* OCR and lets the pipeline skip it. It also
prepares the crops that pass: deskew, upscale, contrast-equalise.

Scores are all normalised to 0..1 and combined multiplicatively, so a crop is
only as good as its worst property.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pelakx.types import BBox


@dataclass(slots=True)
class QualityReport:
    """Why a crop passed or failed, so operators can tune thresholds."""

    score: float
    sharpness: float
    area_score: float
    aspect_score: float
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.passed


def laplacian_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian — the standard cheap blur metric.

    Returned raw (not normalised); typical in-focus plate crops land well
    above 100, motion-blurred ones below 30.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(slots=True)
class QualityGate:
    """Decide whether a plate crop is worth spending an OCR call on.

    Args:
        min_width: reject crops narrower than this many pixels.
        min_height: reject crops shorter than this many pixels.
        target_width: width at which a crop is considered "full resolution";
            the area score saturates here.
        min_sharpness: Laplacian variance below which a crop counts as blurred.
        good_sharpness: Laplacian variance at which the sharpness score saturates.
        aspect_range: plausible width/height range for a plate of this country.
        min_score: overall score required to pass.
    """

    min_width: int = 48
    min_height: int = 14
    target_width: int = 160
    min_sharpness: float = 18.0
    good_sharpness: float = 140.0
    aspect_range: tuple[float, float] = (1.4, 7.0)
    min_score: float = 0.25

    def score(self, crop: np.ndarray) -> QualityReport:
        if crop is None or crop.size == 0:
            return QualityReport(0.0, 0.0, 0.0, 0.0, False, "empty crop")
        h, w = crop.shape[:2]
        if w < self.min_width or h < self.min_height:
            return QualityReport(0.0, 0.0, 0.0, 0.0, False, f"too small ({w}x{h})")

        # --- resolution -----------------------------------------------------
        area_score = min(1.0, w / float(self.target_width))

        # --- aspect ratio ---------------------------------------------------
        ar = w / float(h)
        lo, hi = self.aspect_range
        if ar < lo:
            aspect_score = max(0.0, ar / lo)
        elif ar > hi:
            aspect_score = max(0.0, hi / ar)
        else:
            aspect_score = 1.0

        # --- focus ----------------------------------------------------------
        raw_sharp = laplacian_sharpness(crop)
        span = max(1e-6, self.good_sharpness - self.min_sharpness)
        sharp_score = float(np.clip((raw_sharp - self.min_sharpness) / span, 0.0, 1.0))

        total = area_score * aspect_score * (0.35 + 0.65 * sharp_score)
        passed = total >= self.min_score
        reason = "" if passed else f"score {total:.2f} < {self.min_score:.2f}"
        return QualityReport(
            score=round(total, 4),
            sharpness=round(raw_sharp, 2),
            area_score=round(area_score, 4),
            aspect_score=round(aspect_score, 4),
            passed=passed,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# crop preparation
# ---------------------------------------------------------------------------
def crop_bbox(frame: np.ndarray, bbox: BBox, pad_ratio: float = 0.04) -> np.ndarray:
    """Cut `bbox` out of `frame`, padded and clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.pad(pad_ratio, w, h).as_int()
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def deskew(crop: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Rotate a plate crop so its text baseline is horizontal.

    Uses the minimum-area rectangle of the crop's strong edges. Rotations
    beyond `max_angle` are treated as a bad estimate and skipped.
    """
    if crop.size == 0:
        return crop
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    coords = cv2.findNonZero(edges)
    if coords is None or len(coords) < 30:
        return crop
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.6 or abs(angle) > max_angle:
        return crop
    h, w = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        crop, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def rectify(crop: np.ndarray, min_area_ratio: float = 0.35) -> np.ndarray:
    """Perspective-unwarp a plate crop onto a frontal rectangle.

    Looks for the largest 4-corner contour covering at least `min_area_ratio`
    of the crop and warps it flat. Falls back to :func:`deskew` when no
    convincing quadrilateral is found — which is the common case for small or
    low-contrast crops, so this must stay cheap and never raise.
    """
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return crop
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    edges = cv2.Canny(gray, 50, 160)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return deskew(crop)

    h, w = crop.shape[:2]
    frame_area = float(h * w)
    best: np.ndarray | None = None
    best_area = min_area_ratio * frame_area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < best_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best, best_area = approx.reshape(4, 2).astype(np.float32), area
    if best is None:
        return deskew(crop)

    src = _order_corners(best)
    width = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
    height = int(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])))
    if width < 16 or height < 8:
        return deskew(crop)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(crop, matrix, (width, height), flags=cv2.INTER_CUBIC)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _unsharp_mask(image: np.ndarray, amount: float = 0.8, radius: int = 3) -> np.ndarray:
    """Cheap classical sharpening: subtract a blurred copy from the original.

    Tiny plate crops lose stroke edges when they are stretched several times
    their source size — CLAHE alone re-balances contrast but does not put
    edge energy back. An unsharp mask is a few microseconds of `GaussianBlur`
    + `addWeighted`, orders of magnitude cheaper than a super-resolution
    network, and it measurably crisps character strokes for the CRNN.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def enhance(crop: np.ndarray, target_height: int = 64, clahe: bool = True) -> np.ndarray:
    """Upscale a small crop and equalise its contrast for OCR.

    Source plate photos vary wildly in how much of the crop's pixels are
    real signal: a live traffic-camera plate crop is close to `target_height`
    already, while a phone-crop photo of a printed plate can arrive at 25-50px
    tall. Cubic interpolation is fine for a mild resize but starts to look
    smeared/blocky once the crop must be stretched more than ~2x — Lanczos4
    keeps edges better at those larger factors, and a light unsharp mask
    afterwards puts back stroke contrast that any resampler softens.
    """
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    upscaled = False
    if h < target_height:
        scale = target_height / float(h)
        interp = cv2.INTER_CUBIC if scale <= 2.0 else cv2.INTER_LANCZOS4
        crop = cv2.resize(crop, (max(1, int(w * scale)), target_height), interpolation=interp)
        upscaled = True
    if upscaled:
        # Sharpen a bit harder the more we had to stretch — a 4x blow-up needs
        # more help than a 1.2x one, but never enough to introduce ringing.
        amount = min(1.2, 0.5 + 0.15 * scale)
        crop = _unsharp_mask(crop, amount=amount)
    if not clahe:
        return crop
    if crop.ndim == 2:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare(
    frame: np.ndarray,
    bbox: BBox,
    *,
    do_rectify: bool = True,
    do_enhance: bool = True,
    target_height: int = 64,
    pad_ratio: float = 0.04,
) -> np.ndarray:
    """Full crop -> OCR-ready image path: cut, unwarp, upscale, equalise."""
    crop = crop_bbox(frame, bbox, pad_ratio)
    if crop.size == 0:
        return crop
    if do_rectify:
        crop = rectify(crop)
    if do_enhance:
        crop = enhance(crop, target_height=target_height)
    return crop
