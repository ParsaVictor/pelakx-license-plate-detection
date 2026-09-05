"""Durable, *searchable* results.

CSV is where traffic data goes to die. PelakX writes a SQLite database with an
FTS5 index over the plate strings, so hours of footage become a query::

    pelakx search "12ب*"           # every vehicle whose plate starts with 12ب
    pelakx search --min-speed 90   # everything that went over 90 km/h

The CSV and JSONL writers are still here — they are just not the only option.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pelakx.types import VehicleEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,
    country       TEXT    NOT NULL,
    started_at    TEXT    NOT NULL,
    finished_at   TEXT,
    frames        INTEGER DEFAULT 0,
    fps           REAL    DEFAULT 0,
    config        TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES runs(id) ON DELETE CASCADE,
    track_id       INTEGER NOT NULL,
    plate          TEXT,
    plate_display  TEXT,
    country        TEXT,
    layout         TEXT,
    confidence     REAL,
    ocr_confidence REAL,
    valid          INTEGER,
    vehicle_class  TEXT,
    first_seen     REAL,
    last_seen      REAL,
    n_frames       INTEGER,
    n_reads        INTEGER,
    speed_kmh      REAL,
    direction      TEXT,
    crossings      TEXT,
    alerts         TEXT,
    crop           TEXT,
    source         TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate);
CREATE INDEX IF NOT EXISTS idx_events_run   ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_speed ON events(speed_kmh);

CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
USING fts5(plate, plate_display, content='events', content_rowid='id', tokenize='unicode61');

CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, plate, plate_display)
    VALUES (new.id, new.plate, new.plate_display);
END;
CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, plate, plate_display)
    VALUES ('delete', old.id, old.plate, old.plate_display);
END;
"""


@dataclass(slots=True)
class EventStore:
    """SQLite-backed store for pipeline events."""

    path: str | Path
    _conn: sqlite3.Connection | None = None
    run_id: int | None = None

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:  # pragma: no cover - defensive
            raise RuntimeError("EventStore is closed")
        return self._conn

    # -- runs ---------------------------------------------------------------
    def start_run(self, source: str, country: str, config: dict[str, Any] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (source, country, started_at, config) VALUES (?, ?, ?, ?)",
            (
                str(source),
                country,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(config or {}, ensure_ascii=False, default=str),
            ),
        )
        self.conn.commit()
        self.run_id = int(cur.lastrowid)
        return self.run_id

    def finish_run(self, frames: int = 0, fps: float = 0.0) -> None:
        if self.run_id is None:
            return
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, frames = ?, fps = ? WHERE id = ?",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                frames,
                round(fps, 2),
                self.run_id,
            ),
        )
        self.conn.commit()

    # -- events -------------------------------------------------------------
    def add(self, event: VehicleEvent) -> None:
        row = event.to_row()
        self.conn.execute(
            """INSERT INTO events (
                   run_id, track_id, plate, plate_display, country, layout,
                   confidence, ocr_confidence, valid, vehicle_class,
                   first_seen, last_seen, n_frames, n_reads, speed_kmh,
                   direction, crossings, alerts, crop, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.run_id,
                row["track_id"],
                row["plate"],
                row["plate_display"],
                row["country"],
                row["layout"],
                row["confidence"],
                row["ocr_confidence"],
                int(bool(row["valid"])),
                row["vehicle_class"],
                row["first_seen"],
                row["last_seen"],
                row["n_frames"],
                row["n_reads"],
                row["speed_kmh"] or None,
                row["direction"],
                row["crossings"],
                row["alerts"],
                row["crop"],
                row["source"],
            ),
        )

    def add_many(self, events: Iterable[VehicleEvent]) -> int:
        n = 0
        for event in events:
            self.add(event)
            n += 1
        self.commit()
        return n

    def commit(self) -> None:
        self.conn.commit()

    # -- queries ------------------------------------------------------------
    def search(
        self,
        query: str = "",
        *,
        min_confidence: float = 0.0,
        min_speed: float | None = None,
        country: str | None = None,
        valid_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Search events. `query` supports ``*`` prefixes and FTS syntax."""
        sql = ["SELECT e.* FROM events e"]
        params: list[Any] = []
        where: list[str] = []
        if query.strip():
            term = query.strip()
            if any(ch in term for ch in "*"):
                where.append("(e.plate LIKE ? OR e.plate_display LIKE ?)")
                like = term.replace("*", "%")
                params += [like, like]
            else:
                sql.append("JOIN events_fts f ON f.rowid = e.id")
                where.append("events_fts MATCH ?")
                params.append(term)
        if min_confidence > 0:
            where.append("e.confidence >= ?")
            params.append(min_confidence)
        if min_speed is not None:
            where.append("e.speed_kmh >= ?")
            params.append(min_speed)
        if country:
            where.append("e.country = ?")
            params.append(country.upper())
        if valid_only:
            where.append("e.valid = 1")
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY e.confidence DESC, e.id DESC LIMIT ?")
        params.append(limit)
        rows = self.conn.execute(" ".join(sql), params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT COUNT(*)                        AS events,
                      COUNT(DISTINCT plate)           AS unique_plates,
                      AVG(confidence)                 AS mean_confidence,
                      SUM(valid)                      AS valid_reads,
                      AVG(speed_kmh)                  AS mean_speed,
                      MAX(speed_kmh)                  AS max_speed
               FROM events WHERE plate <> ''"""
        ).fetchone()
        # sqlite3.Row iterates its *values*, so .keys() is required here.
        out = {k: row[k] for k in row.keys()}  # noqa: SIM118
        out["by_class"] = {
            r["vehicle_class"]: r["n"]
            for r in self.conn.execute(
                "SELECT vehicle_class, COUNT(*) AS n FROM events GROUP BY vehicle_class"
            )
        }
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# flat-file writers
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "track_id",
    "plate",
    "plate_display",
    "country",
    "layout",
    "confidence",
    "ocr_confidence",
    "valid",
    "vehicle_class",
    "first_seen",
    "last_seen",
    "n_frames",
    "n_reads",
    "speed_kmh",
    "direction",
    "crossings",
    "alerts",
    "crop",
    "source",
]


class CsvWriter:
    """Streaming CSV writer — rows land on disk as tracks finish."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

    def add(self, event: VehicleEvent) -> None:
        self._writer.writerow(event.to_row())

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> CsvWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class JsonlWriter:
    """Streaming newline-delimited JSON writer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def add(self, event: VehicleEvent) -> None:
        self._fh.write(json.dumps(event.to_row(), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read events back from a SQLite database written by PelakX."""
    with closing(sqlite3.connect(str(path))) as conn:
        conn.row_factory = sqlite3.Row
        yield from (dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id"))
