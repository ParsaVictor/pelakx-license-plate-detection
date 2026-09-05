# Architecture

PelakX is a pipeline of small, independently testable stages. Every stage has a
narrow contract, so any one of them can be swapped without touching the others.

```
video / RTSP / webcam
        │
        ▼
  ① vehicle detect + track      YOLO26 + ByteTrack/BoT-SORT
        │
        ▼
  ② plate detect                inside the vehicle box only
        │
        ▼
  ③ quality gate                scored on the RAW crop, before OCR
        │
        ▼
  ④ OCR engine                  chosen per country from the grammar file
        │
        ▼
  ⑤ country grammar             normalize → tokenize → repair → match → validate
        │
        ▼
  ⑥ temporal fusion             per-track string vote + per-character vote
        │
        ▼
  ⑦ analytics · render · export
```

---

## ① Vehicle detection and tracking — `pelakx/detect/vehicle.py`

Detection and tracking are one stage because Ultralytics' `model.track()` runs
the association inside the same call. Splitting them would mean re-associating
boxes that were already associated.

Tracking quality matters more than it looks: **every identity switch splits one
vehicle's votes across two tracks** and throws away the consensus that stage ⑥
depends on. That is the reason to pay for `botsort.yaml` at busy intersections.

**Contract:** `frame → list[Detection]`, each with a stable `track_id`.

## ② Plate detection — `pelakx/detect/plate.py`

Plates are searched for **inside vehicle boxes**, never across the whole frame.
Two reasons:

* the search area shrinks by roughly 95%;
* every plate gets an owner, which is the precondition for temporal fusion.

Two guards keep this from backfiring. Vehicles below `plate.min_vehicle_px` are
skipped entirely — a 60px car cannot contain a readable plate, and upscaling it
into the detector's 384px input reliably manufactures false positives. And
candidates outside `plate.aspect_range` are dropped before anything else looks
at them.

**Contract:** `vehicle crop → list[Detection]` in crop coordinates; the pipeline
maps them back to frame coordinates.

## ③ Quality gate — `pelakx/quality/gate.py`

```
score = area_score × aspect_score × (0.35 + 0.65 × sharpness_score)
```

Multiplicative on purpose: a crop is only as good as its worst property.

**The gate scores the raw crop, before rectification and upscaling.** Scoring
afterwards would make every crop look full-resolution and would flatten the
Laplacian variance — the gate would be measuring the resampler instead of the
camera. This was a real bug, caught by looking at the numbers.

Crops that pass are then rectified (perspective unwarp, falling back to
edge-based deskew), upscaled, and CLAHE-equalised.

Typical effect: **40–70% of candidate crops never reach OCR**, and the votes
that do arrive are worth more. `pelakx run` prints the exact figure.

**Contract:** `crop → QualityReport(score, sharpness, area, aspect, passed, reason)`.

## ④ OCR engines — `pelakx/ocr/`

PelakX never calls an OCR library directly. It asks a registry for the first
*available* engine in the country's chain:

```yaml
# configs/countries/ir.yaml
ocr_engines:
  preferred: [hezar_fa]
  fallback:  [paddle, easyocr, fast_plate]
```

`BaseOcrEngine` provides lazy loading, timing, warmup, and exception
swallowing — a single malformed crop can never kill a running camera; it
increments a failure counter that `pelakx doctor` surfaces.

Adding a script is a subclass with a `_probe`, a `_load` and a `_read`. About
40 lines. → [ADDING_A_COUNTRY.md](ADDING_A_COUNTRY.md)

**Contract:** `crop → RawRead(text, confidence, char_confidences, region) | None`.

## ⑤ Country grammar — `pelakx/grammar/`

The layer that distinguishes PelakX from a demo. Four steps:

**Normalize** (`normalize.py`) — NFKC, strip bidi controls / tatweel /
diacritics, fold Persian-Arabic-Devanagari-Bengali-Thai digits to ASCII, fold
Arabic letter variants (`ي→ی`, `ك→ک`), drop separators, uppercase Latin, strip
per-country noise words (`ایران`).

**Tokenize** — longest-match against the country's letter list, so `الف`,
`معلولین` and `تشریفات` are single tokens rather than 3, 7 and 7 characters.

**Repair** (`match.py`) — for fixed-length layouts, each token must fit its slot
(`D`igit / `L`etter / `A`lphanumeric). When it does not, the country's confusion
map is consulted (`I→1`, `ك→ک`) under a bounded budget. Each substitution costs
`REPAIR_PENALTY` (6%) of confidence, so repairs are *priced*, never free, and
the grammar cannot hallucinate a plate out of noise.

**Validate** — structural checks (`int_range`, `charset_excludes`, `regex`,
`in_set`, …) run against the matched fields. The score is deliberately
interpretable:

```
base       = ocr_confidence × (1 − 0.06 × repairs)
confidence = base + bonus × (1 − base)        # passing validators close the gap to 1
           = base + bonus                     # failing ones subtract outright
```

A reading that matches nothing is still returned with `valid=False` rather than
dropped — an operator needs to see "we saw a plate we could not parse".

**Contract:** `(text, CountrySpec) → PlateRead | None`.

`identify_country()` runs the same thing across every grammar and returns the
best valid parse — that is `--country auto`. An engine's own region guess
(fast-plate-ocr predicts one) is used only as a **tie-break prior**; a hinted
country still has to satisfy its own grammar.

## ⑥ Temporal fusion — `pelakx/fusion/voting.py`

Two votes run in parallel over one tracklet:

**String vote** — each reading votes for its canonical string, weighted by
`ocr_confidence × crop_quality`, times `0.35` if the grammar rejected it.

**Character vote** — among readings of the modal token length, each position
votes independently. This recovers plates that no single frame read correctly.
The stitched winner is then **re-parsed through the grammar** and only replaces
the string-vote winner if it is legal *and* the per-frame vote was genuinely
split.

```
confidence = mean_conf × (0.55 + 0.45 × agreement)
confidence += (1 − confidence) × 0.5 × (1 − 0.6ⁿ)     # n = agreeing sightings
```

In `--country auto`, votes are kept in **separate per-country pools**
(`MultiVoterPool`) and only compared at the end, so a burst of misparses as one
country cannot drown out consistent readings of another.

**Contract:** `many PlateReads → one PlateRead`, with `explain()` returning the
full candidate breakdown for the dashboard.

## ⑦ Analytics, rendering, export

* **Counting lines** (`analytics/counters.py`) — segment-intersection with a
  signed side test, so one line counts both directions and each track is
  counted once.
* **Speed** (`analytics/speed.py`) — a ground-plane homography from four image
  points to four world points, differentiated over a window rather than
  adjacent frames (a few centimetres of detector jitter over 33 ms is a huge
  instantaneous error). **Without calibration, speed is not reported at all** —
  an uncalibrated number is worse than none.
* **Watchlist** (`analytics/watchlist.py`) — bounded edit distance where
  OCR-confusable substitutions cost 0.25 instead of 1.0, using the same
  confusion map as the grammar layer. Exact, prefix (`12ب*`) and regex
  (`re:^12ب`) entries.
* **Privacy** (`privacy/anonymize.py`) — pixelation rather than Gaussian blur
  (resampling discards information; a Gaussian is linear and partially
  invertible), and salted HMAC pseudonyms rather than bare digests (plate
  strings have a tiny search space).
* **Rendering** (`render/annotate.py`) — PIL + a script-capable TTF, after
  `arabic-reshaper` and `python-bidi`. `cv2.putText` has no Arabic glyphs, no
  contextual shaping and no bidi reordering.
* **Storage** (`store/sqlite_store.py`) — SQLite with an FTS5 index over plate
  strings, so hours of footage become `pelakx search "12ب*"`.

---

## Track lifecycle

```
first detection ─► TrackRecord created
        │
        ├── every frame: box appended, line crossings, speed sample
        ├── rate-limited: at most `ocr.max_reads_per_second` OCR calls
        ├── early stop:  consensus ≥ 0.97 with ≥ 4 votes ⇒ stop paying for OCR
        │
unseen for `track_timeout` seconds
        │
        ▼
  _finalize(): consensus → watchlist → optional pseudonymisation → crop saved
        │
        ▼
  VehicleEvent → outbox → CSV / JSONL / SQLite
```

Events are emitted **as tracks finish**, not at the end of the video, so a live
RTSP stream produces results continuously.

---

## Testing strategy

The grammar, fusion, quality, analytics, privacy, config and store layers have
**no ML dependencies**. That is a deliberate architectural constraint, enforced
by CI installing only `pip install -e ".[dev]"` — no torch, no ultralytics, no
weights. 78 tests run in about a second, and an accidental heavy import in a
core module fails the build loudly.

The detection and OCR stages are validated by running the real pipeline on real
footage, not by mocking them.
