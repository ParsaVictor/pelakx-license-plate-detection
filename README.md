<div align="center">

# 🚗 PelakX

**Multilingual License Plate Intelligence — Read. Track. Understand.**

Real-time vehicle detection, multi-object tracking, and multilingual license plate recognition
(**Persian 🇮🇷 + English 🇬🇧 first**, extensible to any country), with traffic analytics,
an interactive dashboard, and structured exports.

*پلتفرم هوشمند تشخیص و تحلیل پلاک خودرو — چندزبانه، بلادرنگ، ماژولار*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![YOLO](https://img.shields.io/badge/Detection-YOLO-orange)]()
[![OCR](https://img.shields.io/badge/OCR-Fa%20%7C%20En-success)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

</div>

---

## 🎯 Why PelakX?

Thousands of traffic cameras record endless streams of vehicles every day — and most of that
footage is never analyzed. Existing open-source plate-recognition projects are usually
**single-country, English-only, frame-by-frame demos**. PelakX turns raw CCTV/traffic video into
**searchable, multilingual traffic intelligence**:

| Capability | Typical OSS LPR | **PelakX** |
|---|---|---|
| Vehicle detection | ✅ | ✅ best-in-class detector |
| Plate detection | ✅ | ✅ |
| OCR | English only | ✅ **Persian + English** (script-aware) |
| Multi-country plate formats | ❌ | ✅ pluggable **plate registry** per country |
| Tracking & re-identification | ❌ | ✅ tracklet-level plate aggregation |
| Temporal voting (no flicker) | ❌ | ✅ per-track OCR consensus |
| Traffic analytics (counts, speeds) | ❌ | ✅ |
| Dashboard | ❌ | ✅ live web dashboard |
| Structured export | CSV | ✅ CSV / JSON / SQLite |
| Extending to a new language | fork & rewrite | ✅ **add a config + reader plugin** |

> 🔒 Privacy-first: PelakX processes frames locally. No video, image, or plate data ever leaves your machine.

## 🏗️ Architecture (draft)

```
Video/RTSP ─► Detector (vehicle) ─► Tracker ─► Plate Detector ─► OCR Engine (Fa/En)
                                                                        │
                                     Analytics ◄── Plate Registry ◄─────┘
                                          │
                              Dashboard / CSV / JSON / SQLite
```

## 🗺️ Roadmap

- [ ] **v0.1 — Core pipeline**: vehicle + plate detection, OCR (En), video output
- [ ] **v0.2 — Persian support**: Persian OCR engine + plate-format validation
- [ ] **v0.3 — Tracking & voting**: multi-object tracking, temporal consensus, confidence scores
- [ ] **v0.4 — Analytics & export**: counts, speed estimation, CSV/JSON/SQLite
- [ ] **v0.5 — Dashboard**: live interactive web dashboard
- [ ] **v1.0 — Extensibility**: plate registry for other countries + *Language Plugin Guide*

## 📦 Installation

```bash
git clone https://github.com/ParsaVictor/PelakX.git
cd PelakX
pip install -r requirements.txt
```

*(full quickstart lands with v0.1)*

## 🤝 Contributing

The **Language Plugin Guide** (shipping in v1.0) will let you add your country's plates with just
a config file (character set + regex + ordering rules) and an OCR reader — no core changes.

## 📄 License

MIT — see [LICENSE](LICENSE).
