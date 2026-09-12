# Notebooks

## `pelak.ipynb` — standalone quick pipeline

A single self-contained notebook: **YOLO vehicle detection → YOLOv8 plate detection →
OCR → Iran plate analysis**, with light centroid tracking so the printed plate text
does not flicker frame to frame.

- `COUNTRY_MODE`: `IR` (full Iranian analysis — digits/letter, province code, colour,
  plate category), `GLOBAL` (Latin text only), or `AUTO` (per plate).
- Persian OCR: [Hezar](https://github.com/hezarai/hezar) CRNN. Latin OCR:
  [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr). Both fall back to EasyOCR.
- Vehicle labels: Car / Motorcycle / Bus / Truck / Bicycle / Train.
- Outputs an annotated `.mp4`, a one-row-per-plate `.csv`, and an optional `.gif`.

### Setup

```powershell
# from this folder
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

Creates a `.venv` and registers a Jupyter kernel **"Python (pelak)"** for both
Jupyter Lab and VS Code. Then open `pelak.ipynb`, pick that kernel, and edit the
paths in the **بخش ۲ / Settings** cell (`PROJECT_DIR`, `VIDEO_DIR`).

`requirements.txt` here pins the notebook's own dependencies (CPU torch, ONNX
runtime 1.17.3, hezar, fast-plate-ocr). It is separate from the packaged
`PelakX` library at the repo root.

## `pelak_2.ipynb` — crowded-scene / highway variant

Same pipeline as `pelak.ipynb`, tuned for footage with **many vehicles** (highway
cameras) where the nano model missed about half of them:

- vehicle model **`yolo11s` @ `imgsz=1280`, `conf=0.30`** — roughly 3× the recall on
  dense scenes (also fine for close-up / large objects).
- **ByteTrack** (`model.track`) — each vehicle gets a persistent ID, robust to
  occlusion; needs `lapx` (in `requirements.txt`).
- OCR runs at most `MAX_OCR_TRIES` times per vehicle, then the plate is **locked**
  and skipped — keeps total compute low despite the heavier detector.
- smaller on-frame labels; plate text shown only once a reading locks.

Use `pelak.ipynb` for clear single-vehicle clips, `pelak_2.ipynb` for busy scenes.

## `PelakX_Quickstart.ipynb`

Walkthrough of the packaged `pelakx` library (grammar engine, fusion, analytics).
