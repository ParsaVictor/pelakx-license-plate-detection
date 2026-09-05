"""PelakX results dashboard.

    pelakx dashboard --db outputs/pelakx.sqlite

Reads a run's SQLite database and renders it: headline numbers, activity over
time, fleet mix, confidence and speed distributions, and a searchable table of
every vehicle PelakX saw.

Charts follow one rule set throughout: one y-axis per chart, a single series
carries no legend (the title names it), categorical hues are assigned in a
fixed order rather than cycled, grid and axes stay recessive, and every chart
has a table view underneath it. Light and dark are two selected palettes, not
an automatic inversion.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st

# --- palettes: both modes are selected, not flipped ------------------------
PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "grid": "#e8e7e3",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "series": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#4a3aa7",
            "#e34948",
        ],
        "good": "#008300",
        "critical": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "grid": "#383835",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "series": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],
        "good": "#008300",
        "critical": "#e66767",
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/pelakx.sqlite")
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


@st.cache_data(show_spinner=False)
def load_events(db_path: str, mtime: float) -> pd.DataFrame:
    """Load every event. `mtime` busts the cache when the file changes."""
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query("SELECT * FROM events ORDER BY first_seen", conn)
    if not frame.empty:
        frame["confidence"] = frame["confidence"].fillna(0.0)
        frame["plate_display"] = frame["plate_display"].fillna("")
        frame["alerts"] = frame["alerts"].fillna("")
    return frame


def _rgba(hex_color: str, alpha: float) -> str:
    """#rrggbb -> rgba(r, g, b, a). Plotly rejects 8-digit hex."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _bar_marker(color: str) -> dict:
    """Bar marker with 4px rounded data-ends where the plotly build supports it.

    `cornerradius` landed in plotly 5.19; older builds raise on it, so degrade
    to square ends rather than refusing to draw the chart.
    """
    marker = {"color": color}
    try:
        major, minor, *_ = (int(part) for part in plotly.__version__.split(".")[:2])
        if (major, minor) >= (5, 19):
            marker["cornerradius"] = 4
    except (ValueError, AttributeError):
        pass
    return marker


def style(fig: go.Figure, palette: dict, *, height: int = 260) -> go.Figure:
    """Recessive axes, transparent surface, text in ink tokens."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text_secondary"], size=12),
        title=dict(font=dict(color=palette["text_primary"], size=14), x=0, xanchor="left"),
        showlegend=False,
        hoverlabel=dict(bgcolor=palette["surface"], font_color=palette["text_primary"]),
        bargap=0.35,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=palette["grid"],
        tickcolor=palette["grid"],
    )
    fig.update_yaxes(
        gridcolor=palette["grid"],
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickcolor="rgba(0,0,0,0)",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="PelakX", page_icon="🚗", layout="wide")
    options = _args()

    st.sidebar.title("PelakX")
    db_path = st.sidebar.text_input("database", options.db)
    mode = st.sidebar.radio("appearance", ["light", "dark"], horizontal=True)
    palette = PALETTES[mode]

    path = Path(db_path)
    if not path.exists():
        st.warning(f"No database at `{db_path}`. Run `pelakx run <video>` first.")
        st.stop()

    events = load_events(str(path), path.stat().st_mtime)
    if events.empty:
        st.info("The database has no events yet.")
        st.stop()

    # --- filters, in one row above the charts -----------------------------
    st.sidebar.subheader("filters")
    query = st.sidebar.text_input("plate contains", "")
    min_conf = st.sidebar.slider("min confidence", 0.0, 1.0, 0.0, 0.05)
    valid_only = st.sidebar.checkbox("grammar-valid readings only", False)
    classes = sorted(c for c in events["vehicle_class"].dropna().unique())
    chosen = st.sidebar.multiselect("vehicle class", classes, default=classes)

    view = events[events["confidence"] >= min_conf]
    if valid_only:
        view = view[view["valid"] == 1]
    if chosen:
        view = view[view["vehicle_class"].isin(chosen)]
    if query.strip():
        needle = query.strip()
        view = view[
            view["plate"].fillna("").str.contains(needle, case=False, regex=False)
            | view["plate_display"].str.contains(needle, case=False, regex=False)
        ]

    identified = view[view["plate"].fillna("") != ""]
    alerted = view[view["alerts"] != ""]

    # --- headline numbers --------------------------------------------------
    st.markdown("### License plate intelligence")
    cols = st.columns(5)
    cols[0].metric("vehicles", f"{len(view):,}")
    cols[1].metric("plates read", f"{len(identified):,}")
    cols[2].metric("unique plates", f"{identified['plate'].nunique():,}")
    cols[3].metric(
        "mean confidence",
        f"{identified['confidence'].mean():.0%}" if len(identified) else "—",
    )
    cols[4].metric("watchlist alerts", f"{len(alerted):,}")

    if len(alerted):
        st.error(
            f"{len(alerted)} watchlist match(es): "
            + ", ".join(sorted(alerted["plate_display"].unique())[:12])
        )

    # --- activity over time -----------------------------------------------
    left, right = st.columns([3, 2])
    with left:
        if len(view) > 1:
            span = max(1.0, float(view["first_seen"].max()))
            bucket = max(1.0, span / 40.0)
            binned = (
                view.assign(t=(view["first_seen"] // bucket) * bucket)
                .groupby("t")
                .size()
                .reset_index(name="vehicles")
            )
            fig = go.Figure(
                go.Scatter(
                    x=binned["t"],
                    y=binned["vehicles"],
                    mode="lines",
                    line=dict(color=palette["series"][0], width=2, shape="spline"),
                    fill="tozeroy",
                    fillcolor=_rgba(palette["series"][0], 0.13),
                    hovertemplate="t=%{x:.0f}s<br>%{y} vehicles<extra></extra>",
                )
            )
            fig.update_layout(title="Vehicles entering the scene, over time")
            fig.update_xaxes(title_text="seconds into the video")
            st.plotly_chart(style(fig, palette, height=280), use_container_width=True)

    with right:
        mix = view["vehicle_class"].value_counts().reset_index()
        mix.columns = ["vehicle_class", "n"]
        fig = go.Figure(
            go.Bar(
                x=mix["n"],
                y=mix["vehicle_class"],
                orientation="h",
                marker=_bar_marker(palette["series"][0]),
                text=mix["n"],
                textposition="outside",
                textfont=dict(color=palette["text_secondary"]),
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )
        fig.update_layout(title="Fleet mix")
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(style(fig, palette, height=280), use_container_width=True)

    # --- distributions -----------------------------------------------------
    left, right = st.columns(2)
    with left:
        if len(identified):
            fig = go.Figure(
                go.Histogram(
                    x=identified["confidence"],
                    nbinsx=20,
                    marker=_bar_marker(palette["series"][2]),
                    hovertemplate="confidence %{x}<br>%{y} vehicles<extra></extra>",
                )
            )
            fig.update_layout(title="Reading confidence")
            fig.update_xaxes(tickformat=".0%", title_text="")
            st.plotly_chart(style(fig, palette), use_container_width=True)

    with right:
        speeds = view["speed_kmh"].dropna()
        if len(speeds):
            fig = go.Figure(
                go.Histogram(
                    x=speeds,
                    nbinsx=20,
                    marker=_bar_marker(palette["series"][1]),
                    hovertemplate="%{x} km/h<br>%{y} vehicles<extra></extra>",
                )
            )
            fig.update_layout(title="Speed distribution")
            fig.update_xaxes(title_text="km/h")
            st.plotly_chart(style(fig, palette), use_container_width=True)
        else:
            st.caption(
                "**Speed** — not reported. Calibrate `analytics.speed_image_points` "
                "and `speed_world_points` to enable it; an uncalibrated number "
                "would be worse than none."
            )

    # --- the table view ----------------------------------------------------
    st.markdown("#### Vehicles")
    columns = [
        "track_id",
        "plate_display",
        "confidence",
        "valid",
        "vehicle_class",
        "speed_kmh",
        "direction",
        "first_seen",
        "last_seen",
        "n_reads",
        "crossings",
        "alerts",
    ]
    table = view[[c for c in columns if c in view.columns]].sort_values(
        "confidence", ascending=False
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", format="%.0f%%", min_value=0.0, max_value=1.0
            ),
            "plate_display": st.column_config.TextColumn("plate"),
        },
    )
    st.download_button(
        "download filtered CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        file_name="pelakx_events.csv",
        mime="text/csv",
    )

    # --- crops -------------------------------------------------------------
    crops = [c for c in view["crop"].dropna().tolist() if c and Path(c).exists()]
    if crops:
        st.markdown("#### Plate crops")
        for row in range(0, min(len(crops), 24), 8):
            for column, crop in zip(st.columns(8), crops[row : row + 8], strict=False):
                column.image(crop)


if __name__ == "__main__":
    main()
