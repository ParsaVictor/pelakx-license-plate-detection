"""The PelakX pipeline — video in, traffic intelligence out.

    frame ─► vehicle detector + tracker ─► per-track state
                                            │
                          plate detector ◄──┘ (inside the vehicle box)
                                │
                        quality gate ──► rejected crops never reach OCR
                                │
                            OCR engine ──► country grammar ──► temporal vote
                                                                    │
                            annotated video ◄── analytics ◄─────────┘
                            CSV / JSONL / searchable SQLite

Three design decisions carry most of the accuracy and speed:

**Plates are found inside vehicles, not across the frame.** It shrinks the
search area by ~95%, and it gives every plate an owner, which is what makes
the temporal vote possible.

**Crops are gated before OCR.** Blurry, tiny or skewed crops are skipped, so
CPU goes to frames that can actually be read and the vote is not poisoned.

**Reads are rate-limited and early-stopped per track.** Once a vehicle's plate
is known with high confidence, PelakX stops paying for OCR on it.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pelakx.analytics import LineCounter, SpeedEstimator, SpeedStats, Watchlist, dominant_direction
from pelakx.config import PipelineConfig
from pelakx.detect import VehicleDetector, build_plate_detector
from pelakx.fusion import MultiVoterPool, VoterPool
from pelakx.grammar import identify_country, parse, region_to_code, registry
from pelakx.grammar.plate_color import classify_plate_color
from pelakx.grammar.spec import CountrySpec
from pelakx.ocr import resolve as resolve_engine
from pelakx.privacy import FaceBlurrer, blur_region, hash_plate
from pelakx.quality import QualityGate, crop_bbox, enhance, rectify
from pelakx.render import Annotator, plate_color, vehicle_box_color
from pelakx.store import CsvWriter, EventStore, JsonlWriter
from pelakx.types import (
    BBox,
    Detection,
    PlateObservation,
    RawRead,
    TrackRecord,
    VehicleEvent,
)


@dataclass(slots=True)
class RunSummary:
    """What one ``pelakx run`` produced."""

    source: str = ""
    country: str = ""
    frames_read: int = 0
    frames_processed: int = 0
    elapsed: float = 0.0
    vehicles: int = 0
    plates_read: int = 0
    ocr_calls: int = 0
    crops_gated: int = 0
    alerts: int = 0
    events: list[VehicleEvent] = field(default_factory=list)
    line_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    speed: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    #: cumulative seconds per stage, and how many times each ran
    stage_seconds: dict[str, float] = field(default_factory=dict)
    stage_calls: dict[str, int] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.frames_processed / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def ocr_savings(self) -> float:
        """Fraction of candidate crops that never reached the OCR engine."""
        total = self.ocr_calls + self.crops_gated
        return self.crops_gated / total if total else 0.0

    def stage_ms_per_frame(self) -> dict[str, float]:
        """Milliseconds each stage costs per processed frame, slowest first."""
        n = max(1, self.frames_processed)
        out = {k: v * 1000.0 / n for k, v in self.stage_seconds.items()}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "country": self.country,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "elapsed_s": round(self.elapsed, 2),
            "fps": round(self.fps, 2),
            "vehicles": self.vehicles,
            "plates_read": self.plates_read,
            "ocr_calls": self.ocr_calls,
            "crops_gated": self.crops_gated,
            "ocr_savings": round(self.ocr_savings, 3),
            "alerts": self.alerts,
            "line_counts": self.line_counts,
            "speed": self.speed,
            "outputs": self.outputs,
            "stage_ms_per_frame": {k: round(v, 2) for k, v in self.stage_ms_per_frame().items()},
        }


class Pipeline:
    """End-to-end license plate intelligence over a video source."""

    #: engine used for `--country auto` when none is configured explicitly
    AUTO_DEFAULT_ENGINE = "fast_plate"

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        requested = self.config.country.strip().upper()

        #: In auto mode every grammar competes for each reading. One OCR engine
        #: still reads every crop, so auto works within a script (Latin plates
        #: across Europe), not across scripts — run one pipeline per camera for
        #: mixed-script sites.
        self.auto = requested == "AUTO"
        if self.auto:
            self.specs: list[CountrySpec] = registry.all_specs()
            self.spec: CountrySpec = registry.get("GB")  # representative, for defaults
        else:
            self.spec = registry.get(requested)
            self.specs = [self.spec]

        self.vehicles = VehicleDetector(
            self.config.vehicle.weights,
            conf=self.config.vehicle.conf,
            iou=self.config.vehicle.iou,
            imgsz=self.config.vehicle.imgsz,
            device=self.config.vehicle.device,
            classes=tuple(self.config.vehicle.classes) if self.config.vehicle.classes else None,
            tracker=self.config.vehicle.tracker,
            half=self.config.vehicle.half,
        )
        self.plates = build_plate_detector(
            self.config.plate.weights,
            model=self.config.plate.model,
            conf=self.config.plate.conf,
            device=self.config.plate.device,
            imgsz=self.config.plate.imgsz,
        )
        self.ocr = resolve_engine(
            self.spec,
            override=self.config.ocr.engine or (self.AUTO_DEFAULT_ENGINE if self.auto else None),
            **(self.config.ocr.options or {}),
        )

        q = self.config.quality
        self.gate = QualityGate(
            min_width=q.min_width,
            min_height=q.min_height,
            target_width=q.target_width,
            min_sharpness=q.min_sharpness,
            good_sharpness=q.good_sharpness,
            aspect_range=(q.aspect_min, q.aspect_max),
            min_score=q.min_score,
        )
        self.voters = (
            MultiVoterPool({s.code: s for s in self.specs}) if self.auto else VoterPool(self.spec)
        )
        self.counter = LineCounter.from_config(self.config.analytics.lines)
        self.speed = SpeedEstimator(
            self.config.analytics.speed_image_points,
            self.config.analytics.speed_world_points,
            min_frames=self.config.analytics.speed_min_frames,
        )
        self.speed_stats = SpeedStats()
        self.watchlist = Watchlist.from_entries(
            self.config.analytics.watchlist,
            self.spec,
            max_distance=self.config.analytics.watchlist_max_distance,
        )
        self.annotator = Annotator(
            font=self.config.output.font,
            base_dir="R" if self.spec.read_order == "rtl" else "L",
        )
        self.faces = FaceBlurrer() if self.config.privacy.blur_faces else None

        self.tracks: dict[int, TrackRecord] = {}
        self._last_ocr: dict[int, float] = {}
        self._done: set[int] = set()
        #: outbox of events finalized since the caller last drained it
        self._pending: list[VehicleEvent] = []
        #: where to save the best plate crop per track (set by `run`)
        self._crops_dir: Path | None = None
        #: counter for synthetic track ids handed out by the no-vehicle
        #: plate-scan fallback (negative, so they never collide with the
        #: tracker's own ids)
        self._fallback_track_id = 0
        self._stage_seconds: dict[str, float] = defaultdict(float)
        self._stage_calls: dict[str, int] = defaultdict(int)
        self.summary = RunSummary(country="AUTO" if self.auto else self.spec.code)

    def warmup(self) -> None:
        """Load every model and run one dummy pass through each.

        Called before the throughput timer starts, because loading a CRNN off
        disk is a one-time startup cost, not a per-frame cost — folding it into
        the average makes a short benchmark look several times slower than the
        steady state it is trying to measure.
        """
        self.vehicles.warmup((self.config.vehicle.imgsz, self.config.vehicle.imgsz))
        self.plates.warmup((384, 384))
        self.ocr.warmup()

    @contextmanager
    def _stage(self, name: str):
        """Accumulate wall time for one pipeline stage.

        Two `perf_counter` calls per stage per frame is noise next to a YOLO
        forward pass, and knowing *which* stage costs the frame budget is the
        only way to tune a CPU deployment honestly.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._stage_seconds[name] += time.perf_counter() - started
            self._stage_calls[name] += 1

    # ------------------------------------------------------------------
    # frame processing
    # ------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, frame_index: int, timestamp: float, *, annotate: bool = True
    ) -> np.ndarray:
        """Run every stage on one frame; returns the (optionally) annotated frame."""
        with self._stage("vehicle+track"):
            detections = self.vehicles.track(frame)
        seen_now: set[int] = set()

        # Two ways to find plates, and the right one depends on the scene.
        # Per-vehicle (default) searches a ~95% smaller area per call but costs
        # one detector call *per vehicle*; on a busy road that is 5-10 calls a
        # frame. `standalone_plates` runs the detector once over the whole
        # frame and assigns each plate to the vehicle containing it — a big win
        # once the scene is crowded, at the cost of recall on small plates.
        frame_plates: list[Detection] | None = None
        if self.config.vehicle.standalone_plates:
            with self._stage("plate detect"):
                frame_plates = self._plausible_plates(self.plates.detect(frame))

        for det in detections:
            if det.track_id is None:
                continue
            track = self._touch_track(det, frame_index, timestamp)
            seen_now.add(det.track_id)

            center = det.bbox.center
            crossings = self.counter.update(det.track_id, center)
            if crossings:
                track.crossings.extend(crossings)
            kmh = self.speed.update(det.track_id, center, timestamp)
            if kmh is not None:
                track.speed_kmh = kmh

            self._read_plates_for(frame, det, track, frame_index, timestamp, frame_plates)

        # Fallback: nothing vehicle-shaped was found in this frame at all —
        # common for phone-crop photos that are already a tight shot of a
        # bumper/plate rather than a live traffic scene, where the COCO-trained
        # vehicle detector's confidence lands just under threshold or picks
        # the wrong class entirely. Scan the whole frame for plates directly
        # rather than silently producing no reading.
        if (
            not detections
            and not self.config.vehicle.standalone_plates
            and self.config.vehicle.plate_fallback_when_no_vehicle
        ):
            with self._stage("plate detect"):
                stray = self._plausible_plates(self.plates.detect(frame))
            fh, fw = frame.shape[:2]
            for sp in stray:
                self._fallback_track_id -= 1
                pseudo_bbox = sp.bbox.pad(0.6, fw, fh)
                pseudo = Detection(
                    bbox=pseudo_bbox,
                    confidence=sp.confidence,
                    class_id=-1,
                    class_name="plate (no vehicle match)",
                    track_id=self._fallback_track_id,
                )
                track = self._touch_track(pseudo, frame_index, timestamp)
                seen_now.add(pseudo.track_id)
                self._read_plates_for(
                    frame, pseudo, track, frame_index, timestamp, frame_plates=[sp]
                )
                detections.append(pseudo)

        if annotate:
            with self._stage("annotate"):
                frame = self._annotate(frame, detections, timestamp)

        self._expire_tracks(timestamp, keep=seen_now)
        return frame

    def _touch_track(self, det: Detection, frame_index: int, timestamp: float) -> TrackRecord:
        track = self.tracks.get(det.track_id)
        if track is None:
            track = TrackRecord(
                track_id=det.track_id,
                class_name=det.class_name,
                first_frame=frame_index,
                first_seen=timestamp,
            )
            self.tracks[det.track_id] = track
            self.summary.vehicles += 1
        track.last_frame = frame_index
        track.last_seen = timestamp
        track.class_name = det.class_name or track.class_name
        track.boxes.append(det.bbox)
        if len(track.boxes) > 300:
            del track.boxes[:-300]
        return track

    # ------------------------------------------------------------------
    def _should_read(self, track_id: int, timestamp: float) -> bool:
        """Rate-limit and early-stop OCR per track."""
        cfg = self.config.ocr
        if self.voters.votes(track_id) >= cfg.early_stop_min_votes:
            current = self.voters.result(track_id)
            if current and current.confidence >= cfg.early_stop_confidence:
                return False
        if cfg.max_reads_per_second <= 0:
            return True
        last = self._last_ocr.get(track_id)
        return last is None or (timestamp - last) >= 1.0 / cfg.max_reads_per_second

    def _plausible_plates(self, detections: list[Detection]) -> list[Detection]:
        """Drop candidates whose shape could not be a plate."""
        lo, hi = self.config.plate.aspect_range
        return [d for d in detections if lo <= d.bbox.aspect_ratio <= hi]

    def _read_plates_for(
        self,
        frame: np.ndarray,
        vehicle: Detection,
        track: TrackRecord,
        frame_index: int,
        timestamp: float,
        frame_plates: list[Detection] | None = None,
    ) -> None:
        """Find, gate and read the plate belonging to one tracked vehicle.

        When `frame_plates` is given, plates were already detected once over
        the whole frame and are only assigned here; otherwise the detector runs
        inside this vehicle's box.
        """
        if not self._should_read(vehicle.track_id, timestamp):
            return

        h, w = frame.shape[:2]
        cfg = self.config.plate
        if max(vehicle.bbox.width, vehicle.bbox.height) < cfg.min_vehicle_px:
            return

        if frame_plates is not None:
            owned = [d for d in frame_plates if vehicle.bbox.contains_center_of(d.bbox)]
            if not owned:
                return
            best = max(owned, key=lambda d: d.confidence)
            plate_box = best.bbox.clip(w, h)
        else:
            region = vehicle.bbox.pad(cfg.vehicle_pad, w, h)
            rx1, ry1, rx2, ry2 = region.as_int()
            if rx2 - rx1 < 24 or ry2 - ry1 < 24:
                return
            sub = frame[ry1:ry2, rx1:rx2]
            if sub.size == 0:
                return
            with self._stage("plate detect"):
                found = self.plates.detect(sub)
            candidates = self._plausible_plates(found)
            if not candidates:
                return
            best = max(candidates, key=lambda d: d.confidence)
            plate_box = BBox(
                best.bbox.x1 + rx1, best.bbox.y1 + ry1, best.bbox.x2 + rx1, best.bbox.y2 + ry1
            ).clip(w, h)

        observation = PlateObservation(
            frame_index=frame_index,
            timestamp=timestamp,
            bbox=plate_box,
            det_confidence=best.confidence,
            track_id=vehicle.track_id,
        )
        track.observations.append(observation)
        if len(track.observations) > 120:
            del track.observations[:-120]

        # Gate on the *raw* crop. Scoring after upscaling would make every crop
        # look full-resolution and would flatten the Laplacian, so the gate
        # would measure the resampler instead of the camera.
        raw_crop = crop_bbox(frame, plate_box, self.config.plate.vehicle_pad)
        if raw_crop.size == 0:
            return

        if self.config.quality.enabled:
            with self._stage("quality gate"):
                report = self.gate.score(raw_crop)
            observation.quality = report.score
            if not report.passed:
                self.summary.crops_gated += 1
                return
        else:
            observation.quality = 1.0

        with self._stage("crop prep"):
            crop = raw_crop
            if self.config.quality.rectify:
                crop = rectify(crop)
            if self.config.quality.enhance:
                crop = enhance(crop, target_height=self.config.quality.ocr_height)
        if crop.size == 0:
            return

        # Keep the single sharpest crop of this vehicle for the evidence export.
        if self._crops_dir is not None and observation.quality > track.meta.get(
            "best_quality", 0.0
        ):
            track.meta["best_quality"] = observation.quality
            track.meta["best_crop"] = crop.copy()

        with self._stage("ocr"):
            raw = self.ocr.read(crop, self.spec)
        self.summary.ocr_calls += 1
        self._last_ocr[vehicle.track_id] = timestamp
        if raw is None:
            return
        observation.raw = raw

        with self._stage("grammar"):
            read = self._parse_reading(raw)

        # Iran's two-line "temporary free-zone" plates (پلاک موقت مناطق آزاد)
        # are stacked, not side-by-side, and the CRNN OCR engines PelakX ships
        # are single-line readers — they silently return only the top line's
        # text. Rather than build real spatial multi-line segmentation, retry
        # squat crops (aspect ratio well below any single-line Iranian plate)
        # by OCR-ing the top and bottom halves separately and concatenating
        # the two reads into one string for the grammar to match against
        # `free_zone_temp`. Only fires when the normal single-pass read did
        # not already produce a valid layout, so it costs nothing on the
        # common single-line case.
        if (read is None or not read.valid) and self._looks_two_line(crop):
            with self._stage("ocr"):
                split_raw = self._read_two_line(crop)
            if split_raw is not None:
                self.summary.ocr_calls += 1
                with self._stage("grammar"):
                    split_read = self._parse_reading(split_raw)
                if split_read is not None and split_read.valid:
                    raw, read = split_raw, split_read
                    observation.raw = raw

        # A "second opinion" re-read through a *stronger* upscale was tried
        # and measured here (re-OCR `raw_crop` at a bigger target height
        # whenever the first pass failed grammar validation) and reverted:
        # it changed zero of the then-failing images while costing real
        # throughput, because more upscaling was never the fix — see the
        # root-cause finding below. `QualityConfig.enhance`/`.rectify`
        # default to False for exactly this reason: `hezar_fa` resizes
        # whatever it is given down to its own fixed 128x32 grayscale input
        # regardless (see `model_config.yaml` on the hub), so our own
        # upscale bought nothing, and the unsharp mask + perspective unwarp
        # were actively introducing artifacts the CTC decoder read as
        # extra/duplicate digits (confirmed with identical raw crops,
        # enhance-on vs enhance-off, on real failing files — see notebook
        # §OCR root-cause). Feeding the engine the raw (or lightly padded)
        # crop directly took this project's real 9-image test set from 4/9
        # to 8/9 exact matches; the one remaining miss is a source photo the
        # user themselves flagged as very blurry/dark, not a pipeline issue.
        if read is None:
            return

        observation.read = read
        self.summary.plates_read += 1
        self.voters.add(
            vehicle.track_id, read, quality=observation.quality, frame_index=frame_index
        )

    #: below this width/height ratio a plate crop is squat enough to plausibly
    #: be two stacked lines rather than one wide line (ordinary Iranian
    #: civilian/free-zone plates run roughly 3:1-5:1; a two-line temporary
    #: plate is closer to 1:1-1.6:1)
    TWO_LINE_ASPECT_MAX = 2.2

    def _looks_two_line(self, crop: np.ndarray) -> bool:
        if crop.size == 0:
            return False
        h, w = crop.shape[:2]
        return h > 0 and (w / float(h)) < self.TWO_LINE_ASPECT_MAX

    def _read_two_line(self, crop: np.ndarray) -> RawRead | None:
        """OCR a squat crop as two stacked lines and concatenate the reads.

        The split is asymmetric (58%/42%, overlapping in the middle) rather
        than an even 50/50 cut: on real free-zone-temp photos the top line
        (plus the country flag icon) sits slightly taller than the bottom
        "موقت - NN" line, and a clean half-and-half cut tends to slice through
        descenders on one line or the other. Tuned against the one real
        sample available (see notebooks/PelakX_Quickstart.ipynb §"free zone
        temp"), not a general two-line solution.
        """
        h, w = crop.shape[:2]
        top = crop[: max(1, int(h * 0.58)), :]
        bottom = crop[min(h - 1, int(h * 0.42)) :, :]
        top_raw = self.ocr.read(top, self.spec) if top.size else None
        bottom_raw = self.ocr.read(bottom, self.spec) if bottom.size else None
        if top_raw is None and bottom_raw is None:
            return None
        text = (top_raw.text if top_raw else "") + (bottom_raw.text if bottom_raw else "")
        confs = [r.confidence for r in (top_raw, bottom_raw) if r is not None]
        return RawRead(
            text=text,
            confidence=min(confs) if confs else 0.0,
            engine=(top_raw or bottom_raw).engine,
        )

    def _parse_reading(self, raw: RawRead):
        """Turn one OCR reading into a validated :class:`PlateRead`.

        Single country: match that grammar, keeping unparseable text visible
        (``valid=False``). Auto: let every grammar compete, using the engine's
        own region guess only as a tie-break.
        """
        if self.auto:
            return identify_country(
                raw.text,
                self.specs,
                ocr_confidence=raw.confidence,
                engine=raw.engine,
                hint=region_to_code(raw.region),
            )
        return parse(
            raw.text,
            self.spec,
            ocr_confidence=raw.confidence,
            engine=raw.engine,
            repair_budget=self.config.ocr.repair_budget,
        )

    def _plate_category(self, consensus: Any) -> str | None:
        """Map a validated reading's letter slot to a display category.

        Only "disabled" is surfaced today (the user-facing headline
        feature); the grammar's ``letter_semantics`` table already carries
        Government/Taxi/Police/IRGC/Diplomatic/Political/Protocol too (see
        ``configs/countries/ir.yaml``) and ``CountrySpec.describe_letter``
        exposes them the same way, should a future box/label style want them.
        """
        if consensus is None:
            return None
        letter = consensus.fields.get("letter")
        if not letter:
            return None
        spec = self.spec if not self.auto else registry.get(consensus.country)
        if spec.describe_letter(letter, "en") == "Disabled/Veteran":
            return "disabled"
        return None

    # ------------------------------------------------------------------
    def _annotate(
        self, frame: np.ndarray, detections: list[Detection], timestamp: float
    ) -> np.ndarray:
        cfg = self.config.output
        privacy = self.config.privacy

        for line in self.counter.lines:
            self.annotator.line(
                frame, line.p1, line.p2, line.name, f"{line.forward}/{line.backward}"
            )

        for det in detections:
            if det.track_id is None:
                continue
            track = self.tracks.get(det.track_id)
            if track is None:
                continue

            if cfg.draw_tracks and len(track.boxes) > 2:
                self.annotator.trail(frame, track.trajectory())

            consensus = self.voters.result(det.track_id)
            alerted = bool(consensus and self.watchlist.match(consensus.canonical))
            layout_id = consensus.layout_id if consensus else None

            # Category flag: does the letter slot carry the معلولین/جانباز
            # (disabled/veteran) semantic? Checked via the country grammar's
            # own letter_semantics table (configs/countries/ir.yaml), never
            # guessed — a plate that doesn't match a country with that entry
            # (or hasn't matched a layout with a letter slot yet) is simply
            # not flagged.
            category = self._plate_category(consensus)

            last_plate = next(
                (o for o in reversed(track.observations) if o.frame_index >= track.last_frame - 3),
                None,
            )
            # Background-colour hint (white/yellow/red/...), classified from
            # the *raw* plate crop straight off the frame — not the
            # OCR-enhanced one, which may have been contrast-stretched or
            # sharpened in ways that skew the HSV histogram. Cheap (a
            # histogram, no model), so doing it once per displayed frame per
            # track is free next to detection/OCR.
            bg_name: str | None = None
            if last_plate is not None:
                bg_crop = crop_bbox(frame, last_plate.bbox, 0.0)
                if bg_crop.size:
                    bg_result = classify_plate_color(bg_crop)
                    if bg_result.confidence >= 0.3:
                        bg_name = bg_result.name

            vehicle_color = vehicle_box_color(layout_id, category, bg_name)
            self.annotator.box(frame, det.bbox, vehicle_color)

            if last_plate is not None:
                if privacy.blur_plates:
                    blur_region(frame, last_plate.bbox)
                else:
                    self.annotator.box(
                        frame,
                        last_plate.bbox,
                        plate_color(
                            consensus.confidence if consensus else 0.0,
                            alerted,
                            layout_id,
                            category,
                            bg_name,
                        ),
                        1,
                    )

            if cfg.draw_labels:
                x1, y1, _, _ = det.bbox.as_int()
                if consensus:
                    text = f"{consensus.display}  {consensus.confidence:.0%}"
                    if category == "disabled":
                        text += "  ♿"
                    if track.speed_kmh:
                        text += f"  {track.speed_kmh:.0f} km/h"
                else:
                    text = f"#{det.track_id} {track.class_name}"
                self.annotator.label(
                    frame,
                    text,
                    (x1, y1),
                    color=plate_color(
                        consensus.confidence if consensus else 0.0,
                        alerted,
                        layout_id,
                        category,
                        bg_name,
                    ),
                )

        if privacy.blur_faces and self.faces is not None:
            self.faces.apply(frame)

        if cfg.draw_labels:
            self.annotator.hud(
                frame,
                [
                    f"PelakX  {self.summary.country}   engine={self.ocr.id}",
                    f"vehicles={self.summary.vehicles}  reads={self.summary.plates_read}"
                    f"  gated={self.summary.crops_gated}",
                    f"t={timestamp:6.2f}s  active tracks={len(self.tracks)}",
                ],
            )
        return frame

    # ------------------------------------------------------------------
    # track lifecycle
    # ------------------------------------------------------------------
    def _expire_tracks(self, now: float, keep: set[int]) -> None:
        """Finalize tracks unseen for ``track_timeout`` into the outbox."""
        for track_id in list(self.tracks):
            if track_id in keep:
                continue
            if now - self.tracks[track_id].last_seen < self.config.track_timeout:
                continue
            event = self._finalize(track_id)
            if event is not None:
                self._pending.append(event)

    def _finalize(self, track_id: int) -> VehicleEvent | None:
        track = self.tracks.pop(track_id, None)
        if track is None or track_id in self._done:
            return None
        self._done.add(track_id)

        read = self.voters.pop(track_id)
        self.counter.forget(track_id)
        self.speed.forget(track_id)
        self._last_ocr.pop(track_id, None)

        if read is not None and read.confidence < self.config.min_export_confidence:
            read = None

        alerts: list[str] = []
        if read is not None:
            hit = self.watchlist.match(read.canonical)
            if hit is not None:
                alerts.append(hit.label())
                self.summary.alerts += 1
            if self.config.privacy.hash_plates:
                pseudonym = hash_plate(read.canonical, self.config.privacy.hash_salt)
                read.canonical = pseudonym
                read.display = pseudonym
                read.raw_text = ""
                read.fields = {}

        self.speed_stats.add(track.speed_kmh)
        return VehicleEvent(
            track_id=track_id,
            best_crop_path=self._save_crop(track, read),
            plate=read,
            class_name=track.class_name,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            n_frames=max(1, track.last_frame - track.first_frame + 1),
            n_reads=track.n_reads,
            speed_kmh=track.speed_kmh,
            direction=dominant_direction(track.trajectory()),
            crossings=list(dict.fromkeys(track.crossings)),
            alerts=alerts,
            source=self.summary.source,
        )

    def _save_crop(self, track: TrackRecord, read: Any) -> str | None:
        """Write the sharpest plate crop of a finished track to disk."""
        crop = track.meta.pop("best_crop", None)
        if self._crops_dir is None or crop is None or self.config.privacy.no_crops:
            return None
        # ASCII-safe filename: plate strings can be Persian, paths should not be.
        tag = "".join(
            ch for ch in (read.canonical if read else "") if ch.isascii() and ch.isalnum()
        )
        name = f"track{track.track_id:05d}{'_' + tag if tag else ''}.jpg"
        path = self._crops_dir / name
        try:
            ok, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                return None
            path.write_bytes(buffer.tobytes())
        except OSError:
            return None
        return str(path)

    def flush(self) -> list[VehicleEvent]:
        """Finalize every remaining track (call at end of stream)."""
        return [e for tid in list(self.tracks) if (e := self._finalize(tid)) is not None]

    # ------------------------------------------------------------------
    # streaming API
    # ------------------------------------------------------------------
    def stream(
        self, source: str | int | Path, *, annotate: bool = True
    ) -> Iterator[tuple[np.ndarray, list[VehicleEvent]]]:
        """Yield ``(annotated_frame, events_finished_on_this_frame)``.

        This is the low-level API — :meth:`run` wraps it with writers and a
        summary. Use it to embed PelakX in your own loop or web service.
        """
        capture = _open_capture(source)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(fps) or fps <= 1:
            fps = 25.0
        stride = max(1, self.config.frame_stride)
        self.summary.source = str(source)

        # Models load here, outside the timer, so `fps` measures the pipeline
        # rather than the first frame's cold start.
        self.warmup()

        index = 0
        processed = 0
        started = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.summary.frames_read += 1
                if index % stride:
                    index += 1
                    continue
                timestamp = index / fps
                annotated = self.process_frame(frame, index, timestamp, annotate=annotate)
                processed += 1
                self.summary.frames_processed = processed
                yield annotated, self._drain()
                index += 1
                if self.config.max_frames and processed >= self.config.max_frames:
                    break
        finally:
            capture.release()
            self.summary.elapsed = time.perf_counter() - started
            self.summary.stage_seconds = dict(self._stage_seconds)
            self.summary.stage_calls = dict(self._stage_calls)

    def _drain(self) -> list[VehicleEvent]:
        """Take everything finalized since the last call."""
        out, self._pending = self._pending, []
        return out

    # ------------------------------------------------------------------
    def run(
        self,
        source: str | int | Path,
        *,
        progress: Any = None,
        write_outputs: bool = True,
    ) -> RunSummary:
        """Process a whole source and write every configured output."""
        out_cfg = self.config.output
        out_dir = Path(out_cfg.dir)
        writer = None
        csv_writer = jsonl_writer = store = None
        crops_dir: Path | None = None

        if write_outputs:
            out_dir.mkdir(parents=True, exist_ok=True)
            if out_cfg.csv:
                csv_writer = CsvWriter(out_dir / out_cfg.csv_name)
                self.summary.outputs["csv"] = str(out_dir / out_cfg.csv_name)
            if out_cfg.jsonl:
                jsonl_writer = JsonlWriter(out_dir / out_cfg.jsonl_name)
                self.summary.outputs["jsonl"] = str(out_dir / out_cfg.jsonl_name)
            if out_cfg.sqlite:
                store = EventStore(out_dir / out_cfg.sqlite_name)
                store.start_run(str(source), self.summary.country, self.config.to_dict())
                self.summary.outputs["sqlite"] = str(out_dir / out_cfg.sqlite_name)
            if out_cfg.crops and not self.config.privacy.no_crops:
                crops_dir = out_dir / out_cfg.crops_dir
                crops_dir.mkdir(parents=True, exist_ok=True)
                self._crops_dir = crops_dir
                self.summary.outputs["crops"] = str(crops_dir)

        try:
            for frame, events in self.stream(source, annotate=out_cfg.video or out_cfg.draw_labels):
                if write_outputs and out_cfg.video:
                    if writer is None:
                        path = out_dir / out_cfg.video_name
                        writer = _open_writer(path, frame, self._source_fps(source))
                        self.summary.outputs["video"] = str(path)
                    writer.write(frame)
                for event in events:
                    self._emit(event, csv_writer, jsonl_writer, store)
                if progress is not None:
                    progress(self.summary)
            for event in self.flush():
                self._emit(event, csv_writer, jsonl_writer, store)
        finally:
            if writer is not None:
                writer.release()
            if csv_writer is not None:
                csv_writer.close()
            if jsonl_writer is not None:
                jsonl_writer.close()
            if store is not None:
                store.finish_run(self.summary.frames_processed, self.summary.fps)
                store.close()

        self.summary.line_counts = self.counter.summary()
        self.summary.speed = self.speed_stats.summary()
        return self.summary

    def _emit(self, event: VehicleEvent, csv_writer, jsonl_writer, store) -> None:
        self.summary.events.append(event)
        if csv_writer is not None:
            csv_writer.add(event)
        if jsonl_writer is not None:
            jsonl_writer.add(event)
        if store is not None:
            store.add(event)
            store.commit()

    def _source_fps(self, source: str | int | Path) -> float:
        capture = _open_capture(source)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        capture.release()
        return fps if np.isfinite(fps) and fps > 1 else 25.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _open_capture(source: str | int | Path) -> cv2.VideoCapture:
    if isinstance(source, Path):
        source = str(source)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video source: {source!r}")
    return capture


def _open_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, fps), (w, h))
    if not writer.isOpened():  # pragma: no cover - codec-dependent
        raise RuntimeError(f"could not open video writer for {path}")
    return writer
