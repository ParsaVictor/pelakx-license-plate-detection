"""Generate notebooks/PelakX_Quickstart.ipynb.

The notebook is generated rather than hand-edited so it stays diff-friendly:
a .ipynb is JSON with execution counts and outputs baked in, which makes
reviewing a hand-edited one miserable. Edit the cells here and re-run:

    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------
md(
    """
# PelakX — hands-on walkthrough

**Multilingual license plate intelligence.** This notebook takes you from a
plate *string* to a fully processed video, and shows you how to tune it for
your own hardware.

Run the cells in order. Sections 1–4 need **no models and no video** — they
exercise the grammar engine, which is the part that makes PelakX different
from a plain `YOLO → OCR → print` demo.

| # | Section | Needs |
|---|---|---|
| 1 | Setup and environment check | nothing |
| 2 | The grammar engine | nothing |
| 3 | Temporal voting | nothing |
| 4 | Add your own country | nothing |
| 5 | Quality gate | OpenCV |
| 6 | Run the full pipeline on a video | detector + OCR + a video |
| 7 | Where the frame budget goes | same |
| 8 | Tuning: is it real time? | same |
| 9 | Explore the results | a completed run |
| 10 | Persian round-trip test | `pelakx[fa]` |
"""
)

md("## 1. Setup and environment check")

code(
    """
# Run PelakX straight from the repo without installing it.
import sys, os
from pathlib import Path

REPO = Path.cwd()
if not (REPO / "src" / "pelakx").is_dir():          # notebook opened from notebooks/
    REPO = REPO.parent
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

# Ultralytics auto-installs missing packages and that can upgrade numpy
# underneath a working environment. Turn it off before importing anything.
os.environ["YOLO_AUTOINSTALL"] = "false"

import pelakx
print("PelakX", pelakx.__version__, "| repo:", REPO)
"""
)

code(
    """
# What is installed, and what device will `device: auto` pick?
from pelakx import ocr
from pelakx.grammar import registry
from pelakx.runtime import describe_device

print("device (auto) ->", describe_device("auto"))
print("OCR engines   ->", ocr.available() or "none beyond the built-in test double")
print("grammars      ->", len(registry.codes()), ":", ", ".join(registry.codes()))
"""
)

md(
    """
Missing something? Each extra is optional and independent:

```bash
pip install -e ".[detect]"   # YOLO26 vehicle + plate detection (torch)
pip install -e ".[onnx]"     # fast CPU plate detector + Latin plate OCR
pip install -e ".[fa]"       # Persian plate OCR + RTL text rendering
pip install -e ".[dash]"     # dashboard
```

`pelakx doctor` prints the same thing from a terminal, with the exact fix for
anything missing.
"""
)

md(
    """
## 2. The grammar engine

An OCR engine returns a noisy string. The country grammar says what a *legal*
plate looks like. PelakX uses the second to correct the first.
"""
)

code(
    """
from pelakx.grammar import parse, registry

ir = registry.get("IR")

# The word "ایران" is printed on the plate but is not part of the reading, and
# the digits are Persian-Indic. Both are handled by the grammar file.
read = parse("۱۲ ب ۳۴۵ ایران ۱۱", ir, ocr_confidence=0.90)

print("canonical  ", read.canonical)
print("display    ", read.display)
print("layout     ", read.layout_id)
print("fields     ", read.fields)
print("confidence ", f"{read.confidence:.1%}", "| valid:", read.valid)
"""
)

code(
    """
# Confusion repair: slot 6 of an Iranian plate must be a digit. OCR gave "I".
# Instead of discarding the reading, the grammar substitutes 1 — and charges
# for it, so a repaired plate never outranks a clean one.
clean    = parse("12ب34511", ir, ocr_confidence=0.90)
repaired = parse("12ب345I1", ir, ocr_confidence=0.90)

print(f"clean    {clean.canonical}  conf={clean.confidence:.3f}  repairs={clean.repairs}")
print(f"repaired {repaired.canonical}  conf={repaired.confidence:.3f}  repairs={repaired.repairs}")
print("same plate:", clean.canonical == repaired.canonical)
"""
)

code(
    """
# Structural validators. Iran's province code ("ایران XX") must be 10..99.
for plate in ["12ب34555", "12ب34509"]:
    r = parse(plate, ir, ocr_confidence=0.90)
    print(f"{plate}  province={r.fields.get('province'):>3}  "
          f"conf={r.confidence:.3f}  valid={r.valid}")
"""
)

code(
    """
# Auto-detect the issuing country. Every grammar competes; the best *valid*
# parse wins. 12 countries ship in the box.
from pelakx.grammar import identify_country

specs = registry.all_specs()
for text in ["AB12CDE", "1234BCD", "AA-123-BB", "MH12AB1234", "ABC1D23", "12ب34511", "34ABC123"]:
    r = identify_country(text, specs, ocr_confidence=0.92)
    if r:
        print(f"{text:12s} -> {r.country}  {r.display:22s} layout={r.layout_id:12s} {r.confidence:.0%}")
    else:
        print(f"{text:12s} -> no country grammar accepts this")
"""
)

md(
    """
### Single country vs auto

Auto is opt-in. Pointing PelakX at one country is both the default and the
cheaper path — it matches one grammar instead of twelve, and the grammar layer
costs well under a millisecond either way (you will see this in §7).

```bash
pelakx run traffic.mp4 --country IR      # only Iranian grammars are considered
pelakx run traffic.mp4 --country auto    # all 12 compete for every reading
```

Use a fixed country whenever you know it: a tighter alphabet means fewer ways
for OCR to be wrong, which is worth more than the compute you save.
"""
)

md(
    """
## 3. Temporal voting

A per-frame demo prints a different string every frame. PelakX treats every
frame as *evidence about one tracked vehicle*.
"""
)

code(
    """
from pelakx.fusion import TrackConsensus

# Five sightings of one car. Every one has a different single-character error;
# not one of them is fully correct.
sightings = ["12ب34511", "12ب34521", "12ب34511", "12ب94511", "12ب34511"]

consensus = TrackConsensus(ir)
for text in sightings:
    consensus.add(parse(text, ir, ocr_confidence=0.82))

result = consensus.result()
print("frames saw   :", sightings)
print("consensus    :", result.canonical, f"({result.confidence:.1%})")
print()
for c in consensus.explain()["candidates"]:
    print(f"  {c['canonical']:12s} votes={c['votes']}  share={c['share']:.0%}")
"""
)

code(
    """
# Blurry frames count less. Quality is a multiplier on every vote.
c = TrackConsensus(ir)
c.add(parse("12ب34511", ir, ocr_confidence=0.9), quality=1.00)   # sharp crop
c.add(parse("12ب34599", ir, ocr_confidence=0.9), quality=0.05)   # blurry crop
print("winner:", c.result().canonical, "— the sharp frame carries the vote")
"""
)

md(
    """
## 4. Add your own country

A country is a YAML file. No Python, no fork. Here is one built at runtime.
"""
)

code(
    """
import yaml
from pelakx.grammar.spec import CountrySpec
from pelakx.grammar import registry

# Pakistan: "ABC-1234" — 3 letters, 4 digits.
pk_yaml = '''
code: PK
iso3: PAK
name_en: Pakistan
name_native: Pakistan
script: latin
text_direction: ltr
read_order: ltr
digit_style: latin
ocr_engines: { preferred: [fast_plate], fallback: [paddle, easyocr] }
alphabet:
  digits: "0123456789"
  letters: [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z]
layouts:
  - id: standard
    name_en: Standard private plate
    slots: "LLLDDDD"
    groups: { series: [0, 3], serial: [3, 7] }
    display: "{series}-{serial}"
    canonical: "{series}{serial}"
    priority: 100
confusions:
  to_digit:  { "O": "0", "I": "1", "S": "5", "B": "8", "G": "6" }
  to_letter: { "0": "O", "1": "I", "5": "S", "8": "B", "6": "G" }
validators: []
'''

registry.register_spec(CountrySpec.from_dict(yaml.safe_load(pk_yaml)))
pk = registry.get("PK")

print(parse("ABC1234", pk, ocr_confidence=0.9).display)
print(parse("ABC1Z34", pk, ocr_confidence=0.9).display, "  <- Z repaired to 2 in a digit slot")
print("registry now has", len(registry.codes()), "countries:", ", ".join(registry.codes()))
"""
)

md(
    """
To make it permanent, write the same YAML to `configs/countries/pk.yaml`
(or `~/.pelakx/countries/`, or anywhere in `$PELAKX_COUNTRIES`) and PelakX
picks it up on the next run. `pelakx new-country PK` scaffolds it for you.

Full guide: [`docs/ADDING_A_COUNTRY.md`](../docs/ADDING_A_COUNTRY.md)
"""
)

md(
    """
## 5. The quality gate

Running OCR on every plate box of every frame is the biggest waste of CPU in a
naive pipeline. Most crops are too small, too blurry or too skewed to ever be
read correctly — and when they *do* produce a reading, it poisons the vote.
"""
)

code(
    """
import cv2, numpy as np
import matplotlib.pyplot as plt
from pelakx.quality import QualityGate

def fake_plate(width=200, height=50, blur=0):
    img = np.full((height, width, 3), 240, np.uint8)
    for i in range(6):
        x = 12 + i * 30
        cv2.rectangle(img, (x, 10), (x + 14, height - 10), (20, 20, 20), -1)
    return cv2.GaussianBlur(img, (blur | 1, blur | 1), 0) if blur else img

gate = QualityGate()
samples = {
    "sharp, plate-shaped": fake_plate(),
    "half resolution":     fake_plate(100, 25),
    "motion blurred":      fake_plate(blur=15),
    "wrong aspect ratio":  fake_plate(80, 80),
    "too small":           fake_plate(30, 10),
}

fig, axes = plt.subplots(1, len(samples), figsize=(15, 2.6))
for ax, (label, img) in zip(axes, samples.items()):
    r = gate.score(img)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title(f"{label}\\n{'PASS' if r.passed else 'skip'}  score={r.score:.2f}",
                 fontsize=9, color="#1baf7a" if r.passed else "#e34948")
plt.tight_layout(); plt.show()
"""
)

md(
    """
## 6. Run the full pipeline on a video

Point `SOURCE` at a video file, an RTSP URL, or a webcam index (`"0"`).
"""
)

code(
    """
SOURCE  = "data/videos/traffic.mp4"   # <-- your video here
COUNTRY = "auto"                      # or "IR", "GB", "DE", ...
FRAMES  = 120                         # 0 = the whole video

from pathlib import Path
if not Path(SOURCE).exists():
    print(f"!! {SOURCE} not found — drop a video there, or point SOURCE at one.")
else:
    print(f"ready: {SOURCE}")
"""
)

code(
    """
from pelakx.config import PipelineConfig
from pelakx.pipeline import Pipeline

cfg = PipelineConfig().merged(**{
    "country": COUNTRY,
    "max_frames": FRAMES,
    "output.dir": "outputs/notebook",
    # One plate-detector pass per frame instead of one per vehicle.
    # A big win once several vehicles are in view — measured in §7.
    "vehicle.standalone_plates": True,
})

pipeline = Pipeline(cfg)
print("vehicle :", cfg.vehicle.weights, "on", pipeline.vehicles.device)
print("plate   :", type(pipeline.plates).__name__)
print("ocr     :", pipeline.ocr.id)

summary = pipeline.run(SOURCE)
print()
for key, value in summary.as_dict().items():
    if key not in ("events", "stage_ms_per_frame"):
        print(f"  {key:20s} {value}")
"""
)

code(
    """
# Every vehicle PelakX finished with.
import pandas as pd

rows = [e.to_row() for e in summary.events]
df = pd.DataFrame(rows)
df[df["plate"] != ""].sort_values("confidence", ascending=False).head(20)
"""
)

code(
    """
# The sharpest crop kept for each vehicle, with what PelakX made of it.
import matplotlib.pyplot as plt, cv2
from pathlib import Path

read_events = [e for e in summary.events if e.plate and e.best_crop_path]
read_events = [e for e in read_events if Path(e.best_crop_path).exists()][:6]

if read_events:
    fig, axes = plt.subplots(len(read_events), 1, figsize=(7, 1.5 * len(read_events)))
    for ax, e in zip(np.atleast_1d(axes), read_events):
        ax.imshow(cv2.cvtColor(cv2.imread(e.best_crop_path), cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{e.plate.display}   {e.plate.confidence:.0%}   "
                     f"[{e.plate.country}/{e.plate.layout_id}]", fontsize=10)
    plt.tight_layout(); plt.show()
else:
    print("no plate crops were kept — try more frames, or a video with readable plates")
"""
)

code(
    """
# A frame from the annotated video.
import cv2, matplotlib.pyplot as plt

video = summary.outputs.get("video")
if video:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(60, FRAMES - 1 if FRAMES else 60))
    ok, frame = cap.read(); cap.release()
    if ok:
        plt.figure(figsize=(11, 7))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()
"""
)

md(
    """
## 7. Where the frame budget goes

Every stage is timed. This is the only honest way to tune a CPU deployment:
optimise the stage that is actually eating the frame, not the one you assume is.
"""
)

code(
    """
import matplotlib.pyplot as plt

stages = summary.stage_ms_per_frame()
names  = list(stages)[::-1]
values = [stages[n] for n in names]

fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.4))
ax.barh(names, values, color="#2a78d6", height=0.62)
for name, value in zip(names, values):
    ax.text(value + max(values) * 0.015, name, f"{value:.1f} ms",
            va="center", fontsize=9, color="#52514e")
ax.set_xlabel("milliseconds per processed frame")
ax.set_title("Where the frame budget goes", loc="left", fontsize=12)
ax.set_xlim(0, max(values) * 1.18)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)
ax.xaxis.grid(True, color="#e8e7e3"); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()

print(f"{summary.crops_gated} of {summary.crops_gated + summary.ocr_calls} candidate crops "
      f"({summary.ocr_savings:.0%}) never reached the OCR engine.")
"""
)

md(
    """
## 8. Is it real time?

Short answer for CPU: **it depends on the frame size and how many vehicles are
in view**, and the vehicle detector is almost always the bottleneck.

Measured on this project's 1280×720 footage, 8-core CPU. 30 frames × 3 passes
with the variants **interleaved frame by frame**, so background load hits all
of them equally — running them back to back moved the numbers by 40% between
runs. Median reported.

| vehicle backbone | median ms | fps | vs baseline | avg detections |
|---|---|---|---|---|
| PyTorch, imgsz 640 | 111.9 | 8.9 | 1.00× | 7.7 |
| **ONNX, imgsz 640** | **75.7** | **13.2** | **1.48×** | 7.6 |
| PyTorch, imgsz 416 | 85.1 | 11.7 | 1.31× | 4.9 |
| **ONNX, imgsz 416** | **35.5** | **28.2** | **3.15×** | 5.0 |

Two things to read off that table:

* **ONNX export is accuracy-neutral speed** — 1.48× at imgsz 640 with the same
  detections (7.6 vs 7.7 is noise). Do it.
* **Dropping `imgsz` costs recall.** 640→416 lost a third of the detections on
  this far-field footage. That is a real trade, not a free win — check it
  against *your* camera.

The frequently-quoted "43% faster on CPU" for YOLO26 is YOLO26n **versus
YOLO11n**, not PyTorch versus ONNX. They are separate wins and they stack.

The cell below reproduces this on *your* footage. Do not be surprised if a
small clip with one or two vehicles shows a much smaller ONNX gap — it did
here too.
"""
)

code(
    """
# Export the detector to ONNX (one-off; writes into models/).
# Equivalent CLI:  pelakx export --imgsz 640
from pathlib import Path
import os

os.environ["YOLO_AUTOINSTALL"] = "false"   # do not let this upgrade numpy
target = Path("models/yolo26n_640.onnx")

if not target.exists():
    from ultralytics import YOLO
    produced = Path(YOLO("yolo26n.pt").export(
        format="onnx", imgsz=640, dynamic=False, simplify=False, opset=17))
    target.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(target)
print("ONNX detector:", target, f"({target.stat().st_size/1e6:.1f} MB)" if target.exists() else "")
"""
)

code(
    """
# Time the detector alone — no video decoding, no OCR, same frames for each.
# The variants are interleaved frame by frame so background CPU load hits all
# of them equally; running them back to back gives numbers that swing wildly.
import time, statistics, cv2
from pathlib import Path
from pelakx.detect import VehicleDetector

cap = cv2.VideoCapture(SOURCE)
frames = []
for _ in range(25):
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()

variants = [("PyTorch 640", "yolo26n.pt", 640), ("PyTorch 320", "yolo26n.pt", 320)]
if Path("models/yolo26n_640.onnx").exists():
    variants.insert(1, ("ONNX    640", "models/yolo26n_640.onnx", 640))

dets = {}
for label, weights, imgsz in variants:
    d = VehicleDetector(weights, imgsz=imgsz, device="cpu", verbose=False)
    d.warmup((imgsz, imgsz)); d.detect(frames[0])            # settle
    dets[label] = d

times  = {label: [] for label, _, _ in variants}
counts = {label: [] for label, _, _ in variants}
PASSES = 2
for _ in range(PASSES):
    for f in frames:
        for label, _, _ in variants:
            t0 = time.perf_counter(); r = dets[label].detect(f)
            times[label].append((time.perf_counter() - t0) * 1000)
            counts[label].append(len(r))

base = statistics.median(times[variants[0][0]])
print(f"{len(frames)} frames x {PASSES} passes, round-robin, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}\\n")
print(f"{'backbone':13s} {'median ms':>10} {'fps':>7} {'vs first':>9} {'avg dets':>9}")
for label, _, _ in variants:
    ms = statistics.median(times[label])
    print(f"{label:13s} {ms:10.1f} {1000/ms:7.1f} {base/ms:8.2f}x "
          f"{statistics.mean(counts[label]):9.1f}")
"""
)

md(
    """
### Tuning cheat-sheet (CPU)

Apply in this order — the first three are worth more than any model swap:

| lever | effect | cost |
|---|---|---|
| `frame_stride: 2–4` | linear speedup | none in practice: a vehicle is in frame for 2–4 s, you do not need 30 samples |
| `vehicle.standalone_plates: true` | 2.25× on a crowded scene (measured: 2.3 → 5.2 fps) | slightly lower recall on small plates |
| ONNX export of the detector | 1.48× at imgsz 640 | none — identical detections |
| `vehicle.imgsz: 416` | ~1.3× on top | **real recall loss** on distant vehicles |
| `ocr.max_reads_per_second` ↓ | fewer OCR calls | slower convergence of the vote |
| `quality.min_score` ↑ | fewer OCR calls | marginal crops never read |

Rule of thumb from the measurements above: on an 8-core CPU at 1280×720 with
~20 vehicles in view, expect **5–7 fps** out of the box and **12–15 fps** with
ONNX + stride 2. A 25 fps camera is comfortably handled at stride 2–3, which is
what a real deployment would use anyway.

A GPU changes the picture entirely — set `device: auto` (already the default)
and it will be used automatically.
"""
)

md(
    """
## 9. Explore the results

The run wrote a SQLite database with a full-text index over the plate strings.
"""
)

code(
    """
from pelakx.store import EventStore

with EventStore("outputs/notebook/pelakx.sqlite") as store:
    print("stats:", store.stats())
    print()
    for row in store.search(min_confidence=0.5, limit=10):
        print(f"  track {row['track_id']:>4}  {row['plate_display']:<18} "
              f"{row['confidence']:.0%}  {row['vehicle_class']}")
"""
)

code(
    """
# Wildcard and speed queries — the same thing `pelakx search` does.
with EventStore("outputs/notebook/pelakx.sqlite") as store:
    print("plates starting with KL:", len(store.search("KL*")))
    print("faster than 60 km/h    :", len(store.search(min_speed=60)))
    print("grammar-valid only     :", len(store.search(valid_only=True)))
"""
)

md(
    """
Or explore it visually:

```bash
pelakx dashboard --db outputs/notebook/pelakx.sqlite
```
"""
)

md(
    """
## 10. Persian round-trip test

Needs `pip install -e ".[fa]"`. This renders synthetic Iranian plates, reads
them back with the Persian CRNN, and pushes the result through the grammar.

It is a **plumbing test, not an accuracy benchmark** — synthetic plates use a
system font, not the real Iranian plate typeface, so the letter slot is the
weak point here. For real numbers you need real Iranian footage.
"""
)

code(
    """
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from pelakx import ocr
from pelakx.grammar import parse, registry
from pelakx.grammar.normalize import shape_rtl
from pelakx.render import find_font

ir = registry.get("IR")
font_path = find_font(None)
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def make_plate(left, letter, right, province, w=440, h=100):
    img = Image.new("RGB", (w, h), (250, 250, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, w - 3, h - 3], outline=(20, 20, 20), width=3)
    d.rectangle([2, 2, 52, h - 3], fill=(20, 60, 160))          # the blue IR band
    font = ImageFont.truetype(font_path, 58)
    # Draw each group at a fixed x. An Iranian plate reads left-to-right even
    # though it is written in an RTL script, so the groups must NOT be run
    # through the bidi algorithm as one string.
    for x, part in ((72, left), (150, letter), (215, right), (345, province)):
        d.text((x, 16), shape_rtl(part.translate(FA)), font=font, fill=(15, 15, 15))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

try:
    engine = ocr.get("hezar_fa")
except Exception as exc:
    engine = None
    print("Persian OCR unavailable:", exc)

if engine:
    cases = [("12","ب","345","11"), ("47","س","628","22"),
             ("83","ط","914","68"), ("29","د","507","41")]
    hits = 0
    for left, letter, right, province in cases:
        img = make_plate(left, letter, right, province)
        truth = f"{left}{letter}{right}{province}"
        raw = engine.read(img, ir)
        got = parse(raw.text, ir, ocr_confidence=raw.confidence, engine=raw.engine) if raw else None
        ok = got is not None and got.canonical == truth
        hits += ok
        print(f"truth={truth:12s} raw={(raw.text if raw else None)!r:14s} "
              f"-> {got.canonical if got else '-':12s} "
              f"layout={got.layout_id if got else '-':9s} {'MATCH' if ok else 'differ'}")
    print(f"\\n{hits}/{len(cases)} exact")
"""
)

code(
    """
# What the annotator draws, and why the bidi fix matters.
from pelakx.grammar.normalize import shape_rtl

label = "12 ب 345 | ایران 11"
shaped = shape_rtl(label)          # plate mode: LTR group order preserved
print("logical :", label)
print("rendered:", shaped)
print("groups still in reading order:",
      shaped.index("12") < shaped.index("345") < shaped.index("11"))
"""
)

md(
    """
## Where to go next

* [`docs/ADDING_A_COUNTRY.md`](../docs/ADDING_A_COUNTRY.md) — your country's grammar, and OCR for a new script
* [`docs/MODELS.md`](../docs/MODELS.md) — model choices, benchmarks, deployment profiles
* [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — how each stage works and why
* [`configs/default.yaml`](../configs/default.yaml) — every setting, documented

```bash
pelakx doctor                 # environment check
pelakx bench <video>          # where the frame budget goes
pelakx run <video> --privacy  # blurred plates, pseudonymised exports
```
"""
)


# ---------------------------------------------------------------------------
def build() -> dict:
    cells = []
    for kind, source in CELLS:
        lines = source.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "PelakX_Quickstart.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"wrote {out}  ({len(CELLS)} cells, {n_code} code)")
