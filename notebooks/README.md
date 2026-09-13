# Notebooks

Two notebooks, same pipeline, same logic — just different display language.

## `license_plate_detection.ipynb` — English

**Recommended starting point for most users.** Self-contained pipeline: **YOLO
vehicle detection → YOLOv8 plate detection → OCR → Iran plate parsing**, with
light centroid tracking so the printed plate text does not flicker frame to
frame.

- `COUNTRY_MODE`: `IR` (full Iranian analysis — digits/letter, province code,
  colour, plate category), `GLOBAL` (Latin text only), or `AUTO` (per plate).
- Persian OCR: [Hezar](https://github.com/hezarai/hezar) CRNN. Latin OCR:
  [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr). Both fall
  back to EasyOCR.
- Vehicle labels: Car / Motorcycle / Bus / Truck / Bicycle / Train.
- Outputs an annotated `.mp4`, a one-row-per-plate `.csv`, and an optional `.gif`.
- Vehicle model and input resolution are config-driven — see
  [`../configs/`](../configs) — so a specific camera/scene can be retuned
  without touching a single line of code.

## `pelak.ipynb` — همین نوت‌بوک، به فارسی

دقیقاً همان منطق و همان کد بالا — فقط راهنماها، کامنت‌ها و پیام‌های چاپ‌شده
به فارسی‌اند، برای کاربران ایرانی.

## Setup

```powershell
# from this folder
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

Creates a `.venv` and registers a Jupyter kernel **"Python (pelak)"** for both
Jupyter Lab and VS Code. Then open either notebook, pick that kernel, and edit
the paths in the settings cell (`PROJECT_DIR`, `VIDEO_DIR`).

`requirements.txt` here pins the notebooks' own dependencies (CPU torch, ONNX
runtime 1.17.3, hezar, fast-plate-ocr). It is separate from the packaged
`PelakX` library at the repo root.

## `PelakX_Quickstart.ipynb`

Walkthrough of the packaged `pelakx` library (grammar engine, fusion, analytics) —
a different, more heavily-instrumented codebase than the two notebooks above.
