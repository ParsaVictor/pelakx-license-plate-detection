"""Privacy mode: plate blurring, face blurring, salted plate pseudonymisation."""

from __future__ import annotations

from pelakx.privacy.anonymize import FaceBlurrer, blur_region, generate_salt, hash_plate

__all__ = ["FaceBlurrer", "blur_region", "generate_salt", "hash_plate"]
