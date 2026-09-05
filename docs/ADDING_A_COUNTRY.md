# Adding your country to PelakX

> **The promise:** supporting a new country should be a config file, not a fork.
>
> Persian and English are covered end-to-end out of the box. Everything else —
> Arabic, Cyrillic, Devanagari, Han, Thai, Hebrew, Greek — plugs in through the
> two extension points described here.

There are two, and you usually only need the first:

| You need | Because | Effort |
|---|---|---|
| **1. A country grammar** (`configs/countries/xx.yaml`) | Your plates have a different *shape* | ~30 min, no code |
| **2. An OCR engine** (`pelakx.ocr.BaseOcrEngine` subclass) | Your plates use a script no shipped engine reads | ~40 lines of Python |

If your country uses the Latin alphabet, you almost certainly only need step 1 —
the `fast_plate` engine already reads plates from 65+ countries.

---

## Step 1 — the country grammar

### 1.1 Scaffold it

```bash
pelakx new-country PK --name Pakistan
```

That copies `configs/countries/_template.yaml` to `configs/countries/pk.yaml`.
PelakX discovers it immediately — no rebuild, no registration:

```bash
pelakx countries        # PK is already listed
```

Grammar files are also picked up from `~/.pelakx/countries/` and from every
path in `$PELAKX_COUNTRIES`, so you can keep private or in-house grammars
outside the repo.

### 1.2 Describe the script

```yaml
code: PK
iso3: PAK
name_en: Pakistan
name_native: پاکستان
script: latin            # latin | arabic | cyrillic | devanagari | han | ...
text_direction: ltr      # direction of the language
read_order: ltr          # direction the plate is *read* — often not the same!
digit_style: latin       # which digits are printed on the plate
```

> **`read_order` vs `text_direction` matters.** Iranian plates are written in an
> RTL script but are read left-to-right. Getting this wrong reverses every plate
> you produce.

### 1.3 Declare the alphabet

```yaml
alphabet:
  digits: "0123456789"
  letters: [A, B, C, D, E, ...]
```

Two rules that save you hours:

* **List only letters that legally appear on a plate.** Spain excludes vowels;
  the UK excludes I, Q and Z from the area code. Every letter you leave out is
  a whole class of OCR errors the engine can no longer make.
* **Multi-character letters are fine.** Iran's `الف`, `معلولین` and `تشریفات`
  are single *tokens*; the tokenizer matches them longest-first.

If the plate has printed words that are not part of the reading — a country
name, a state name, a slogan — list them so they get stripped:

```yaml
noise_words: [ایران, IRAN]     # single characters are ignored on purpose
```

### 1.4 Define the layouts

A layout is one legal plate shape. Two styles:

**Fixed length — use `slots`.** Preferred, because it enables confusion repair.

```yaml
layouts:
  - id: standard
    name_en: Standard plate
    slots: "LLLDDDD"          # D = digit, L = letter, A = either
    groups: { head: [0, 3], serial: [3, 7] }   # [start, end) over the slots
    display: "{head}-{serial}"
    canonical: "{head}{serial}"
    priority: 100
```

**Variable length — use `regex` with named groups.**

```yaml
  - id: legacy
    regex: '^(?P<area>[A-Z]{1,3})(?P<serial>\d{1,4})$'
    display: "{area} {serial}"
    canonical: "{area}{serial}"
    priority: 50
```

* `canonical` is the compact database key — no spaces, no punctuation.
* `display` is what humans see; use the native script freely.
* `priority` breaks ties when several layouts match. Give the modern format the
  higher number.

> **Ambiguity tip.** When a regex block can absorb characters from its
> neighbour (Germany's `city` + `letters`), make the *first* one lazy
> (`{1,3}?`). `BMW1234` then splits as `B | MW | 1234`, which is right far more
> often than `BM | W | 1234`.

### 1.5 Teach it your OCR confusions

This is where accuracy comes from. When a character lands in a slot it cannot
legally occupy, PelakX tries these substitutions instead of throwing the
reading away:

```yaml
confusions:
  to_digit:  { "O": "0", "I": "1", "S": "5", "B": "8", "G": "6" }
  to_letter: { "0": "O", "1": "I", "5": "S", "8": "B", "6": "G" }
```

Each applied substitution costs ~6% confidence, and the budget is capped
(`ocr.repair_budget`, default 2) so the grammar can never hallucinate a plate
out of noise.

Populate this from *your own* engine's mistakes, not from theory — run
`pelakx run` on an hour of footage, look at the low-confidence readings, and
write down what it actually confuses.

### 1.6 Add structural validators

Anything that is checkable, check:

```yaml
validators:
  - { id: province_range, applies_to: [standard], field: province,
      type: int_range, min: 1, max: 81, weight: 0.2 }
```

| type | meaning |
|---|---|
| `int_range` | numeric field within `min`..`max` |
| `equals` / `not_equal` | exact value comparison |
| `charset_excludes` / `charset_includes` | forbidden / required characters |
| `in_set` | value must appear in `value:` list |
| `regex` | field must fully match `value:` |

A passing validator closes part of the gap to certainty; a failing one both
subtracts its weight and marks the reading `valid: false`.

### 1.7 Test it

```bash
pelakx parse "ABC1234" --country PK
pelakx parse "ABC1Z34" --country PK      # does the repair do what you expect?
pelakx parse "ABC1234" --country auto    # is it distinguishable from other countries?
```

Then add a case to `tests/test_grammar.py`. `test_every_shipped_country_parses_a_sample_of_its_own_layouts`
already round-trips every fixed layout automatically.

---

## Step 2 — an OCR engine for a new script

Only needed when no shipped engine reads your script. Check first:

```bash
pelakx engines
```

`paddle` (PP-OCRv5+) covers Latin, Arabic, Cyrillic, Devanagari and Han;
`easyocr` covers 80+ languages. Point your grammar at one of them before
writing any code:

```yaml
ocr_engines:
  preferred: [paddle]
  fallback:  [easyocr]
```

### 2.1 Write the engine

```python
# my_plugin.py
import numpy as np
from pelakx.ocr import BaseOcrEngine, register
from pelakx.types import RawRead


@register
class MyScriptOcr(BaseOcrEngine):
    id = "my_script"                       # what grammars reference
    label = "My script recognizer"
    scripts = ("mine",)
    install_hint = "pip install my-ocr-lib"

    @classmethod
    def _probe(cls) -> bool:
        """Cheap check: is the dependency importable?"""
        import importlib.util
        return importlib.util.find_spec("my_ocr_lib") is not None

    def _load(self):
        """Build the model once; called lazily on first use."""
        import my_ocr_lib
        return my_ocr_lib.Recognizer(self.options.get("model", "default"))

    def _read(self, crop: np.ndarray, spec=None) -> RawRead | None:
        text, score = self._model(crop)          # BGR uint8, already deskewed
        if not text:
            return None
        return RawRead(text=text, confidence=float(score), engine=self.id)
```

The base class gives you lazy loading, timing, warmup, and exception
swallowing — a single bad crop can never kill a running camera.

### 2.2 Point your grammar at it

```yaml
ocr_engines:
  preferred: [my_script]
  fallback:  [paddle, easyocr]
```

PelakX walks that chain and uses the first engine that is actually installed,
so your users get a graceful fallback instead of an ImportError.

### 2.3 Load it

Import your module before building the pipeline:

```python
import my_plugin          # registers the engine
from pelakx import Pipeline, PipelineConfig

Pipeline(PipelineConfig(country="PK")).run("traffic.mp4")
```

---

## Training a plate detector for your country

Plate *detection* usually transfers across countries better than plate
*reading* does — try the shipped detector first. If your plates are an unusual
shape (square Gulf plates, motorcycle stacks), fine-tune:

```bash
yolo detect train model=yolo26n.pt data=your_plates.yaml imgsz=640 epochs=100
pelakx run traffic.mp4 --country PK --plate-weights runs/detect/train/weights/best.pt
```

A few hundred annotated frames from your own cameras beats ten thousand from
someone else's.

---

## Checklist before opening a PR

- [ ] `pelakx countries XX` renders your layouts correctly
- [ ] `pelakx parse` handles a clean plate, a plate with one OCR error, and
      an out-of-range one
- [ ] `letters` contains only characters that legally appear on plates
- [ ] `confusions` reflect mistakes you actually observed
- [ ] Tests added to `tests/test_grammar.py`
- [ ] Any table you cite (province codes, state codes) has a source in a comment

Grammars are shipped as data, so a country PR touches **no Python at all**.
That is the whole point.
