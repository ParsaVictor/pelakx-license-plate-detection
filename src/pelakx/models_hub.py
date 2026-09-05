"""Optional model downloads.

PelakX deliberately ships **no weights**. Three ways to get a plate detector,
in order of how little work they are:

1. ``pip install 'pelakx[onnx]'`` — the ``open-image-models`` ONNX detector
   downloads itself on first use. Nothing to manage. This is the default.
2. ``pelakx download-models`` — pulls a community YOLO plate detector into
   ``models/``, which the pipeline picks up automatically.
3. Train your own and pass ``--plate-weights path/to/best.pt``.

Everything here is opt-in and prints exactly what it is fetching from where,
because silently downloading binaries onto someone's machine is not okay.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RemoteModel:
    """A downloadable weights file."""

    name: str
    url: str
    filename: str
    license: str
    note: str = ""

    def destination(self, directory: Path) -> Path:
        return Path(directory) / self.filename


#: Community plate detectors that work out of the box with Ultralytics.
PLATE_DETECTORS: dict[str, RemoteModel] = {
    "yolov8-plate": RemoteModel(
        name="yolov8-plate",
        url=(
            "https://github.com/Muhammad-Zeerak-Khan/"
            "Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
        ),
        filename="license_plate.pt",
        license="see upstream repository",
        note="Widely used YOLOv8 single-class plate detector. Good general starting point.",
    ),
}

DEFAULT_PLATE_DETECTOR = "yolov8-plate"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(
    model: RemoteModel,
    dest_dir: str | Path = "models",
    *,
    force: bool = False,
    on_progress=None,
) -> Path:
    """Download `model` into `dest_dir`, skipping the fetch if it is present."""
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = model.destination(directory)
    if target.exists() and not force:
        return target

    request = urllib.request.Request(model.url, headers={"User-Agent": "PelakX/0.1"})
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        while chunk := response.read(1 << 16):
            out.write(chunk)
            seen += len(chunk)
            if on_progress is not None:
                on_progress(seen, total)
    tmp.replace(target)
    return target


def download_default_plate_detector(
    dest_dir: str | Path = "models", *, force: bool = False
) -> Path:
    """Fetch the default community plate detector into ``models/``."""
    model = PLATE_DETECTORS[DEFAULT_PLATE_DETECTOR]
    print(f"downloading {model.name} from {model.url}")
    path = download(model, dest_dir, force=force)
    print(f"  -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"  sha256: {sha256_of(path)}")
    print(f"  license: {model.license}")
    return path
