# Model selection guide

Every model in PelakX is swappable. This page is the reasoning behind the
defaults, so you can override them on purpose rather than by accident.

---

## 1. Vehicle detection — is YOLO26 the right choice?

**Yes, and specifically because most ALPR runs on CPU.**

YOLO26 (Ultralytics, released January 2026) is the current generation. Three
changes matter for this workload:

| Change | Why it matters here |
|---|---|
| **Native end-to-end inference** | Predictions come out without an NMS pass. NMS is a variable-latency, CPU-bound post-process — exactly the wrong thing on a traffic box. |
| **DFL removed** from the head | Simpler export; far better behaved on embedded runtimes and NPUs. |
| **Retrained recipe** (ProgLoss + STAL, MuSGD optimizer) | Better small-object recall, which is what a distant plate *is*. |

Reported COCO numbers:

| Model | mAP<sup>50-95</sup> | T4 TensorRT latency |
|---|---|---|
| YOLO26n | 40.9 | 1.7 ms |
| YOLO26s | 48.6 | 2.5 ms |
| YOLO26m | 53.1 | 4.7 ms |
| YOLO26l | 55.0 | 6.2 ms |
| YOLO26x | 57.5 | 11.8 ms |

The number that decides the default: **YOLO26n is ~43% faster than YOLO11n on
CPU ONNX** (56.1 ms → 38.9 ms on an Intel Xeon). On a GPU the gap is small
enough that either generation is fine; on CPU it is the difference between
real-time and not.

> That 43% is YOLO26n **versus YOLO11n**, both exported to ONNX. It is *not*
> a PyTorch-versus-ONNX figure — that is a separate win, measured below, and
> the two stack.

### Measured on this repo's own footage

Vehicle detector only, 1280×720, ~20 vehicles in view, 8-core CPU, no GPU.

**Method matters here.** A first attempt ran each variant back to back and
produced numbers that moved by 40% between runs — the machine simply had
different background load at different moments. These figures come from 30
frames × 3 passes with the variants **interleaved frame by frame**, so drift
hits every variant equally. Median reported.

| vehicle backbone | median ms | fps | vs baseline | avg detections |
|---|---|---|---|---|
| PyTorch, imgsz 640 | 111.9 | 8.9 | 1.00× | 7.7 |
| **ONNX, imgsz 640** | **75.7** | **13.2** | **1.48×** | 7.6 |
| PyTorch, imgsz 416 | 85.1 | 11.7 | 1.31× | 4.9 |
| **ONNX, imgsz 416** | **35.5** | **28.2** | **3.15×** | 5.0 |

Two things to read off that:

* **ONNX export is accuracy-neutral speed.** 1.48× at imgsz 640 with the same
  detections (7.6 vs 7.7 is noise). There is no reason not to:

  ```bash
  pelakx export --imgsz 640
  pelakx run traffic.mp4 --vehicle-weights models/yolo26n_640.onnx
  ```

* **Dropping `imgsz` costs recall.** 640 → 416 lost a third of the detections
  on this far-field footage. That is a real trade, not a free win — measure it
  against *your* camera before shipping it.

On a 576×720 clip with one or two vehicles the PyTorch/ONNX gap narrowed to
within noise, so treat 1.48× as "what this footage showed", not a constant.
Run it on yours.

### Which size to pick

| Situation | Weights |
|---|---|
| CPU-only box, single stream | `models/yolo26n_640.onnx` (`pelakx export`) |
| CPU-only, many streams | same + `frame_stride: 2` |
| Any GPU | `yolo26s.pt` |
| Far-field / small vehicles, GPU | `yolo26m.pt` + `imgsz: 960` |
| Maximum accuracy, offline batch | `yolo26x.pt` |

```bash
pelakx run traffic.mp4 --vehicle-weights yolo26s.pt --device cuda:0
```

### Device selection

`device: auto` is the default everywhere and resolves at startup:

```
CUDA GPU present  ->  cuda:0
Apple Silicon     ->  mps
otherwise         ->  cpu
```

So the same config file runs on a workstation and on a fanless camera box with
no edit. Force it with `--device cpu` / `--device cuda:1`, or globally with
`PELAKX_DEVICE`. `pelakx doctor` prints what it resolved to and why.

### When *not* to use YOLO26

* You have an existing YOLO11/YOLOv8 fine-tune you trust — keep it, PelakX
  takes any Ultralytics checkpoint.
* Your deployment target has a frozen ONNX opset or a vendor SDK that has not
  caught up with the new head.

---

## 2. Plate detection

| Backend | Setup | Speed (CPU) | Notes |
|---|---|---|---|
| **`open-image-models` YOLOv9-t 384 end2end** *(default)* | `pip install 'pelakx[onnx]'` | ~5–10 ms | Downloads itself, no weights to manage, end-to-end (no NMS) |
| Community YOLOv8 plate detector | `pelakx download-models` | ~20–35 ms | Lands in `models/license_plate.pt`, auto-detected |
| Your own fine-tune | `--plate-weights best.pt` | depends | Best ceiling; a few hundred frames from your own cameras goes a long way |

PelakX searches for plates **inside vehicle boxes**, not across the whole
frame. That cuts the search area by roughly 95% and gives every plate an owner
— which is what makes the temporal vote possible in the first place.

---

## 3. OCR — the part that actually decides your accuracy

### Persian / Iranian plates 🇮🇷

**Default: `hezar_fa`** — `hezarai/crnn-fa-license-plate-recognition-v2`, a
CRNN-CTC recognizer fine-tuned specifically on Iranian plates.

```bash
pip install 'pelakx[fa]'
pelakx run traffic.mp4 --country IR
```

Why a purpose-built CRNN rather than a general OCR model:

* General OCR is trained on *documents*. Plates are short, high-contrast,
  fixed-alphabet strings under motion blur — a different distribution.
* The Persian plate alphabet is ~28 letters plus 10 digits. A CTC head over a
  closed alphabet cannot emit a character that is not on a plate; a general
  model can and will.
* It is small enough to run comfortably on CPU.

Alternatives if you need them: `paddle` with `lang: arabic`, or `easyocr` with
`langs: [fa, ar]`. Both are slower and less accurate on plates specifically —
useful mainly for bootstrapping a dataset.

Two things the grammar layer fixes for free on top of any of these:
Persian-Indic digits are folded to ASCII, and the word **ایران** printed on the
plate is stripped before matching.

### Latin-script plates 🇬🇧🇺🇸🇪🇺

**Default: `fast_plate`** — `fast-plate-ocr`'s `cct-xs-v2-global-model`.

* ONNX, ~13 ms per plate on CPU
* Trained on ~220k plates from 65+ countries
* Reported ~94% on an Argentinian plate set, ~90% on SL-LPR

Model options:

| `ocr.options.model` | When |
|---|---|
| `cct-xs-v2-global-model` *(default)* | Balanced, global |
| `cct-s-v2-global-model` | More accuracy, still fast |
| `european-plates-mobile-vit-v2-model` | Europe-only deployment (MobileViT-V2, 40+ countries) |

```bash
pelakx run traffic.mp4 --country GB --engine fast_plate
```

```yaml
ocr:
  engine: fast_plate
  options: { model: cct-s-v2-global-model }
```

### Any other script

**`paddle`** (PP-OCRv5 / PP-OCRv6) is the generalist: Latin, Arabic, Cyrillic,
Devanagari, Han, and more from one package. PP-OCRv6 (June 2026) reports
+5.1% recognition over PP-OCRv5-server at 34.5M parameters, with a ~5×
OpenVINO speedup on Intel CPUs.

**`easyocr`** covers the widest language list and is the easiest way to
bootstrap: use it to label a few thousand crops, then train a small CRNN on
your own alphabet and register it as an engine.

See [ADDING_A_COUNTRY.md](ADDING_A_COUNTRY.md) for the ~40-line engine plugin.

---

## 4. Tracking

`bytetrack.yaml` is the default: fast, and good enough when vehicles are mostly
unoccluded. Switch to `botsort.yaml` for busy intersections where cars pass
behind each other — it adds appearance features and recovers identity better,
at a real speed cost.

```yaml
vehicle:
  tracker: botsort.yaml
```

Tracking quality matters more than it looks: every identity switch splits one
vehicle's votes across two tracks and throws away the consensus.

---

## 5. Recommended profiles

### CPU-only (laptop, NUC, typical camera server)

```yaml
vehicle: { weights: yolo26n.pt, imgsz: 640, device: cpu }
plate:   { weights: null }            # ONNX end-to-end detector
ocr:     { max_reads_per_second: 4 }
frame_stride: 2
quality: { enabled: true, min_score: 0.30 }
```

### GPU workstation

```yaml
vehicle: { weights: yolo26s.pt, imgsz: 960, device: "cuda:0", half: true }
plate:   { imgsz: 640, device: "cuda:0" }
ocr:     { max_reads_per_second: 10 }
frame_stride: 1
```

### Maximum accuracy, offline

```yaml
vehicle: { weights: yolo26x.pt, imgsz: 1280, device: "cuda:0", tracker: botsort.yaml }
ocr:     { max_reads_per_second: 0, early_stop_confidence: 1.01 }  # never stop early
quality: { min_score: 0.15 }                                       # read even marginal crops
```

---

## 6. Where the speed actually goes

Do not guess — measure:

```bash
pelakx bench traffic.mp4 --country IR -n 100
```

It prints a per-stage millisecond breakdown and tells you whether this machine
keeps up with this stream. Levers in the order they are worth applying:

| lever | effect | cost |
|---|---|---|
| `frame_stride: 2–4` | linear | none in practice — a vehicle is in frame for 2–4 s, you do not need 30 samples |
| `vehicle.standalone_plates: true` | one plate pass per frame instead of one per vehicle | slightly lower recall on small plates |
| ONNX export (`pelakx export`) | ~25% at imgsz 640 | none — identical detections |
| `vehicle.imgsz: 416` | ~2× | **real recall loss** on distant vehicles |
| `quality.min_score` ↑ | fewer OCR calls | marginal crops never read |
| `ocr.early_stop_confidence` ↓ | fewer OCR calls | slower convergence of the vote |

### standalone_plates, measured

Same 1280×720 clip, ~20 vehicles, 60 frames, CPU:

| | fps | plate detection | calls |
|---|---|---|---|
| per vehicle (default) | 2.3 | 268 ms/frame | 327 |
| **whole frame** | **5.2** | **56 ms/frame** | **60** |

On a crowded scene the whole-frame pass was 2.25× faster *and* produced more
readings (8 vs 5) — the per-vehicle path was spending its budget re-running the
detector on 20 small crops. On a sparse scene with distant plates the default
wins instead. This is exactly the kind of thing to check on your own footage
rather than take on faith.

### What to expect on CPU

8-core CPU, 1280×720, ~20 vehicles in view: **5–7 fps** out of the box,
**12–15 fps** with ONNX + `frame_stride: 2`. A 25 fps camera is comfortably
handled at stride 2–3, which is what a real deployment would use anyway —
sampling every frame of a vehicle that is in view for three seconds buys
nothing that the temporal vote does not already have.
