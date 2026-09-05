"""Unit tests for the stages around the grammar layer.

These need numpy/OpenCV but no models, no weights and no video, so they run in
CI in a couple of seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from pelakx.analytics import (
    CountingLine,
    LineCounter,
    SpeedEstimator,
    SpeedStats,
    Watchlist,
    confusion_distance,
    dominant_direction,
)
from pelakx.config import PipelineConfig
from pelakx.fusion import MultiVoterPool, TrackConsensus, VoterPool
from pelakx.grammar import parse, region_to_code, registry
from pelakx.privacy import blur_region, generate_salt, hash_plate
from pelakx.quality import QualityGate, crop_bbox, enhance, laplacian_sharpness
from pelakx.store import CsvWriter, EventStore
from pelakx.types import BBox, PlateRead, VehicleEvent


@pytest.fixture(scope="module")
def ir():
    return registry.get("IR")


def _read(text: str, spec, conf: float = 0.9) -> PlateRead:
    read = parse(text, spec, ocr_confidence=conf, engine="test")
    assert read is not None
    return read


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_bbox_geometry():
    box = BBox(10, 20, 110, 70)
    assert box.width == 100 and box.height == 50
    assert box.area == 5000
    assert box.center == (60, 45)
    assert box.aspect_ratio == 2.0


def test_bbox_pad_clips_to_frame():
    padded = BBox(0, 0, 50, 50).pad(0.5, 100, 100)
    assert padded.x1 == 0 and padded.y1 == 0  # cannot go negative
    assert padded.x2 <= 100 and padded.y2 <= 100


def test_bbox_iou():
    a, b = BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)
    assert a.iou(b) == pytest.approx(1 / 3)
    assert a.iou(BBox(100, 100, 110, 110)) == 0.0


# ---------------------------------------------------------------------------
# quality gate
# ---------------------------------------------------------------------------
def _plate_image(width: int = 200, height: int = 50, blur: int = 0) -> np.ndarray:
    import cv2

    img = np.full((height, width, 3), 240, np.uint8)
    for i in range(6):  # high-contrast bars stand in for characters
        x = 12 + i * 30
        cv2.rectangle(img, (x, 10), (x + 14, height - 10), (20, 20, 20), -1)
    return cv2.GaussianBlur(img, (blur | 1, blur | 1), 0) if blur else img


def test_gate_accepts_a_sharp_plate_sized_crop():
    report = QualityGate().score(_plate_image())
    assert report.passed and report.aspect_score == 1.0


def test_gate_rejects_tiny_crops():
    report = QualityGate().score(_plate_image(30, 10))
    assert not report.passed and "too small" in report.reason


def test_gate_rejects_wrong_aspect_ratio():
    report = QualityGate().score(_plate_image(80, 80))
    assert report.aspect_score < 1.0


def test_gate_penalises_blur():
    sharp = QualityGate().score(_plate_image())
    blurred = QualityGate().score(_plate_image(blur=15))
    assert blurred.sharpness < sharp.sharpness
    assert blurred.score < sharp.score


def test_gate_rejects_empty_crop():
    assert not QualityGate().score(np.empty((0, 0, 3), np.uint8)).passed


def test_laplacian_sharpness_orders_correctly():
    assert laplacian_sharpness(_plate_image()) > laplacian_sharpness(_plate_image(blur=21))


def test_crop_bbox_clips_out_of_frame_boxes():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert crop_bbox(frame, BBox(-50, -50, 20, 20)).size > 0
    assert crop_bbox(frame, BBox(500, 500, 600, 600)).size == 0


def test_enhance_upscales_short_crops():
    out = enhance(_plate_image(120, 20), target_height=64)
    assert out.shape[0] == 64


# ---------------------------------------------------------------------------
# temporal fusion
# ---------------------------------------------------------------------------
def test_consensus_returns_none_without_evidence(ir):
    assert TrackConsensus(ir).result() is None


def test_consensus_picks_the_majority_string(ir):
    consensus = TrackConsensus(ir)
    for text in ["12ب34511", "12ب34511", "12ب34599"]:
        consensus.add(_read(text, ir))
    result = consensus.result()
    assert result.canonical == "12ب34511"


def test_consensus_confidence_grows_with_agreement(ir):
    one, many = TrackConsensus(ir), TrackConsensus(ir)
    one.add(_read("12ب34511", ir))
    for _ in range(6):
        many.add(_read("12ب34511", ir))
    assert many.result().confidence > one.result().confidence


def test_character_vote_recovers_a_plate_no_frame_read_correctly(ir):
    """Each reading has a different single-character error; the vote fixes it."""
    consensus = TrackConsensus(ir)
    for text in ["12ب34511", "12ب34521", "12ب34511", "12ب94511", "12ب34511"]:
        consensus.add(_read(text, ir))
    assert consensus.result().canonical == "12ب34511"


def test_invalid_readings_are_down_weighted(ir):
    consensus = TrackConsensus(ir)
    consensus.add(_read("12ب34511", ir, conf=0.8))
    for _ in range(2):
        consensus.add(_read("ZZZZ", ir, conf=0.8))  # parses, but grammar-invalid
    assert consensus.result().canonical == "12ب34511"


def test_low_quality_frames_count_less(ir):
    consensus = TrackConsensus(ir)
    consensus.add(_read("12ب34511", ir), quality=1.0)
    consensus.add(_read("12ب34599", ir), quality=0.05)
    assert consensus.result().canonical == "12ب34511"


def test_explain_lists_competing_candidates(ir):
    consensus = TrackConsensus(ir)
    consensus.add(_read("12ب34511", ir))
    consensus.add(_read("12ب34599", ir))
    explained = consensus.explain()
    assert explained["n_votes"] == 2
    assert {c["canonical"] for c in explained["candidates"]} == {"12ب34511", "12ب34599"}


def test_voter_pool_isolates_tracks(ir):
    pool = VoterPool(ir)
    pool.add(1, _read("12ب34511", ir))
    pool.add(2, _read("34د56722", ir))
    assert pool.result(1).canonical == "12ب34511"
    assert pool.result(2).canonical == "34د56722"
    assert pool.pop(1).canonical == "12ب34511"
    assert pool.result(1) is None


def test_multi_voter_pool_picks_the_best_country():
    specs = {s.code: s for s in registry.all_specs()}
    pool = MultiVoterPool(specs)
    gb = registry.get("GB")
    for _ in range(4):
        pool.add(7, _read("AB12CDE", gb))
    pool.add(7, _read("1234BCD", registry.get("ES")))
    result = pool.pop(7)
    assert result.country == "GB" and result.canonical == "AB12CDE"


# ---------------------------------------------------------------------------
# counting lines and direction
# ---------------------------------------------------------------------------
def test_counting_line_detects_both_directions():
    line = CountingLine("gate", (0, 50), (100, 50))
    assert line.crossing((50, 10), (50, 90)) is not None
    assert line.crossing((50, 90), (50, 10)) is not None
    assert line.crossing((10, 10), (20, 20)) is None


def test_line_counter_counts_each_track_once():
    counter = LineCounter.from_config({"gate": [[0, 50], [100, 50]]})
    counter.update(1, (50, 10))
    assert counter.update(1, (50, 90)) == ["gate:forward"]
    assert counter.update(1, (50, 10)) == []  # already counted
    assert counter.summary()["gate"]["total"] == 1


def test_dominant_direction():
    assert dominant_direction([(0, 100), (0, 0)]) == "N"  # up the frame
    assert dominant_direction([(0, 0), (100, 0)]) == "E"
    assert dominant_direction([(0, 0), (2, 2)]) is None  # barely moved


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------
def test_speed_is_not_reported_without_calibration():
    estimator = SpeedEstimator(None, None)
    assert not estimator.calibrated
    assert estimator.update(1, (10, 10), 0.0) is None


def test_speed_estimate_on_a_calibrated_plane():
    # A 10m x 10m square of road maps to a 100x100 px image region.
    estimator = SpeedEstimator(
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        min_frames=4,
    )
    assert estimator.calibrated
    speed = None
    for i in range(10):
        # 10 px/frame at 10 fps = 1 m per 0.1 s = 36 km/h
        speed = estimator.update(1, (float(i * 10), 0.0), i * 0.1)
    assert speed is not None and 30 <= speed <= 40


def test_speed_stats_summary():
    stats = SpeedStats(limit_kmh=50)
    for value in (30, 40, 60, 70):
        stats.add(value)
    summary = stats.summary()
    assert summary["count"] == 4 and summary["max_kmh"] == 70 and summary["over_limit"] == 2


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------
def test_watchlist_exact_prefix_and_regex(ir):
    watchlist = Watchlist.from_entries(["12ب34511", "34د*", "re:^99"], ir)
    assert watchlist.match("12ب34511").kind == "exact"
    assert watchlist.match("34د56722").kind == "prefix"
    assert watchlist.match("99ب11122").kind == "regex"
    assert watchlist.match("55ل99988") is None


def test_watchlist_tolerates_one_ocr_error():
    gb = registry.get("GB")
    watchlist = Watchlist.from_entries(["AB12CDE"], gb, max_distance=1)
    hit = watchlist.match("AB12CDF")
    assert hit is not None and hit.kind == "fuzzy"


def test_confusable_swaps_are_cheaper_than_real_edits():
    confusables = {frozenset(("0", "O"))}
    assert confusion_distance("AB0", "ABO", confusables) < confusion_distance(
        "AB0", "AB7", confusables
    )


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------
def test_hash_is_stable_and_salt_dependent():
    salt_a, salt_b = generate_salt(), generate_salt()
    digest = hash_plate("12ب34511", salt_a)
    # same plate + same salt -> same pseudonym, so vehicles stay linkable
    assert digest == hash_plate("12ب34511", salt_a)
    # different salt -> different pseudonym, so deployments cannot be joined
    assert digest != hash_plate("12ب34511", salt_b)
    # different plate -> different pseudonym
    assert digest != hash_plate("34د56722", salt_a)
    # the plate itself is gone: the output is `px_` + hex only
    assert digest.startswith("px_")
    assert all(c in "0123456789abcdef" for c in digest[3:])
    assert "ب" not in digest


def test_hash_requires_a_salt():
    with pytest.raises(ValueError):
        hash_plate("12ب34511", "")


def test_blur_region_changes_only_the_region():
    frame = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    blur_region(frame, BBox(10, 10, 50, 50))
    assert not np.array_equal(frame[10:50, 10:50], before[10:50, 10:50])
    assert np.array_equal(frame[60:, 60:], before[60:, 60:])


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_config_roundtrips_through_dict():
    cfg = PipelineConfig()
    restored = PipelineConfig.from_dict(cfg.to_dict())
    assert restored.country == cfg.country
    assert restored.vehicle.weights == cfg.vehicle.weights


def test_config_dotted_overrides():
    cfg = PipelineConfig().merged(**{"country": "GB", "vehicle.device": "cuda:0"})
    assert cfg.country == "GB" and cfg.vehicle.device == "cuda:0"
    assert cfg.vehicle.weights == PipelineConfig().vehicle.weights  # untouched


def test_config_ignores_unknown_keys():
    cfg = PipelineConfig.from_dict({"country": "GB", "not_a_real_key": 1})
    assert cfg.country == "GB"


def test_shipped_default_config_loads():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.country and cfg.vehicle.weights


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def _event(track_id: int, read: PlateRead | None, **kwargs) -> VehicleEvent:
    return VehicleEvent(
        track_id=track_id,
        plate=read,
        class_name=kwargs.pop("class_name", "car"),
        first_seen=kwargs.pop("first_seen", 1.0),
        last_seen=kwargs.pop("last_seen", 3.0),
        n_frames=kwargs.pop("n_frames", 30),
        n_reads=kwargs.pop("n_reads", 5),
        **kwargs,
    )


def test_store_roundtrip_and_search(tmp_path, ir):
    db = tmp_path / "events.sqlite"
    with EventStore(db) as store:
        store.start_run("test.mp4", "IR")
        store.add(_event(1, _read("12ب34511", ir)))
        store.add(_event(2, _read("34د56722", ir), speed_kmh=95.0))
        store.commit()
        assert len(store.search("12ب*")) == 1
        assert len(store.search(min_speed=90)) == 1
        assert len(store.search()) == 2
        assert store.stats()["events"] == 2
        store.finish_run(frames=100, fps=12.5)


def test_store_survives_events_without_a_plate(tmp_path):
    db = tmp_path / "events.sqlite"
    with EventStore(db) as store:
        store.start_run("test.mp4", "IR")
        store.add(_event(3, None))
        store.commit()
        assert store.search()[0]["plate"] == ""


def test_csv_writer_writes_a_header_and_rows(tmp_path, ir):
    path = tmp_path / "events.csv"
    with CsvWriter(path) as writer:
        writer.add(_event(1, _read("12ب34511", ir)))
    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("track_id,plate")
    assert "12ب34511" in lines[1]


# ---------------------------------------------------------------------------
# region hints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("region", "expected"),
    [("Germany", "DE"), ("india", "IN"), ("Unknown", None), ("", None), ("gb", "GB")],
)
def test_region_to_code(region, expected):
    assert region_to_code(region) == expected


# ---------------------------------------------------------------------------
# device resolution
# ---------------------------------------------------------------------------
def test_resolve_device_passes_explicit_values_through():
    from pelakx.runtime import resolve_device

    for explicit in ("cpu", "cuda:0", "cuda:1", "mps", "0,1"):
        assert resolve_device(explicit) == explicit


def test_resolve_device_auto_returns_something_usable():
    from pelakx.runtime import resolve_device

    for value in ("auto", "", None, "AUTO", "  auto  "):
        resolved = resolve_device(value)
        assert resolved.startswith(("cpu", "cuda", "mps")) or resolved.isdigit()


def test_env_override_wins(monkeypatch):
    from pelakx import runtime

    monkeypatch.setenv("PELAKX_DEVICE", "cuda:3")
    runtime.clear_cache()
    try:
        assert runtime.resolve_device("auto") == "cuda:3"
        assert runtime.is_gpu("auto")
        assert "cuda:3" in runtime.describe_device("auto")
    finally:
        monkeypatch.delenv("PELAKX_DEVICE", raising=False)
        runtime.clear_cache()


def test_is_gpu_classifies_correctly():
    from pelakx.runtime import is_gpu

    assert is_gpu("cuda:0") and is_gpu("mps") and is_gpu("0")
    assert not is_gpu("cpu")


def test_describe_device_labels_explicit_vs_auto():
    from pelakx.runtime import describe_device

    assert "explicit" in describe_device("cpu")
    assert "auto" in describe_device("auto")


def test_config_defaults_to_auto_device():
    cfg = PipelineConfig()
    assert cfg.vehicle.device == "auto"
    assert cfg.plate.device == "auto"


# ---------------------------------------------------------------------------
# run summary timing
# ---------------------------------------------------------------------------
def test_stage_ms_per_frame_is_sorted_and_normalised():
    from pelakx.pipeline import RunSummary

    summary = RunSummary(frames_processed=10)
    summary.stage_seconds = {"ocr": 2.0, "vehicle+track": 5.0, "grammar": 0.01}
    per_frame = summary.stage_ms_per_frame()
    assert list(per_frame) == ["vehicle+track", "ocr", "grammar"]  # slowest first
    assert per_frame["vehicle+track"] == pytest.approx(500.0)  # 5 s / 10 frames


def test_ocr_savings_reports_gate_effectiveness():
    from pelakx.pipeline import RunSummary

    assert RunSummary(ocr_calls=30, crops_gated=70).ocr_savings == pytest.approx(0.7)
    assert RunSummary().ocr_savings == 0.0
