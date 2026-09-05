"""Privacy-preserving output.

License-plate systems are surveillance systems. Most open-source ALPR projects
ignore that entirely; PelakX makes the privacy-preserving mode a first-class,
one-flag feature:

``--privacy`` turns on plate blurring in the rendered video, salted-hash plate
identifiers in the exports, and disables crop dumping. The pipeline still
counts vehicles, measures speeds and matches a watchlist — it just stops
producing a re-identifiable record of who was where.

The hash is a keyed HMAC, not a bare digest: plate strings have a tiny search
space (a few million per country), so an unsalted SHA-256 of a plate is
trivially reversible by brute force and is *not* anonymisation.
"""

from __future__ import annotations

import hmac
import secrets
from hashlib import sha256

import cv2
import numpy as np

from pelakx.types import BBox


def generate_salt(n_bytes: int = 32) -> str:
    """A fresh random salt. Store it per-deployment; losing it is the point."""
    return secrets.token_hex(n_bytes)


def hash_plate(plate: str, salt: str, *, length: int = 12) -> str:
    """Pseudonymise a plate string with a keyed hash.

    Same plate + same salt -> same id, so vehicles remain linkable across a
    deployment while the plate itself is not recoverable without the salt.
    """
    if not plate:
        return ""
    if not salt:
        raise ValueError(
            "privacy.hash_plates requires a non-empty privacy.hash_salt; "
            "generate one with pelakx.privacy.generate_salt()"
        )
    digest = hmac.new(salt.encode("utf-8"), plate.encode("utf-8"), sha256).hexdigest()
    return f"px_{digest[:length]}"


def blur_region(
    frame: np.ndarray, bbox: BBox, *, strength: int = 25, pixelate: bool = True
) -> np.ndarray:
    """Irreversibly obscure `bbox` in `frame` (in place).

    Pixelation (downsample-then-upsample) is preferred over Gaussian blur:
    a Gaussian is a linear operator and can be partially inverted, while
    resampling actually discards the information.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.clip(w, h).as_int()
    if x2 <= x1 or y2 <= y1:
        return frame
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    if pixelate:
        rh, rw = region.shape[:2]
        blocks_w = max(1, rw // max(2, strength // 3))
        blocks_h = max(1, rh // max(2, strength // 3))
        small = cv2.resize(region, (blocks_w, blocks_h), interpolation=cv2.INTER_LINEAR)
        frame[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        k = strength | 1  # kernel size must be odd
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return frame


class FaceBlurrer:
    """Optional face blurring using OpenCV's bundled Haar cascade.

    Deliberately dependency-free: it uses the cascade that ships inside the
    ``opencv-python`` wheel. It is not state of the art, and it says so — for
    a production deployment plug in a proper face detector.
    """

    def __init__(self, scale: float = 1.15, min_neighbors: int = 5) -> None:
        self.scale = scale
        self.min_neighbors = min_neighbors
        self._cascade: cv2.CascadeClassifier | None = None
        self._unavailable = False

    @property
    def available(self) -> bool:
        return not self._unavailable

    def _load(self) -> cv2.CascadeClassifier | None:
        if self._cascade is not None or self._unavailable:
            return self._cascade
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            self._unavailable = True
            return None
        self._cascade = cascade
        return cascade

    def apply(self, frame: np.ndarray, *, strength: int = 25) -> int:
        """Blur every detected face in place; returns how many were blurred."""
        cascade = self._load()
        if cascade is None:
            return 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, self.scale, self.min_neighbors, minSize=(24, 24))
        for x, y, w, h in faces:
            blur_region(
                frame, BBox(float(x), float(y), float(x + w), float(y + h)), strength=strength
            )
        return len(faces)
