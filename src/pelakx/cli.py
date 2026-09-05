"""``pelakx`` command line interface."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pelakx import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="PelakX — multilingual license plate intelligence.",
)
console = Console()


def _ok(flag: bool) -> str:
    return "[green]yes[/green]" if flag else "[red]no[/red]"


# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the PelakX version."""
    console.print(f"PelakX {__version__}")


# ---------------------------------------------------------------------------
@app.command()
def countries(
    code: Optional[str] = typer.Argument(None, help="Show one country in detail."),
) -> None:
    """List the plate grammars PelakX can read."""
    from pelakx.grammar import registry

    if code:
        spec = registry.get(code)
        table = Table(title=f"{spec.code} — {spec.name_en} ({spec.name_native})", box=None)
        table.add_column("layout", style="cyan")
        table.add_column("shape")
        table.add_column("example display")
        for layout in spec.sorted_layouts():
            shape = layout.slots or layout.regex
            table.add_row(layout.id, shape, layout.display)
        console.print(table)
        console.print(
            f"\nscript={spec.script}  digits={spec.digit_style}  "
            f"letters={len(spec.letters)}  engines={' → '.join(spec.engine_chain)}"
        )
        if spec.letter_semantics:
            console.print("\n[bold]letter meanings[/bold]")
            for letter, meaning in spec.letter_semantics.items():
                console.print(f"  {letter}\t{meaning.get('en', '')} / {meaning.get('fa', '')}")
        console.print(f"\n[dim]{spec.source_path}[/dim]")
        return

    table = Table(title="Available plate grammars", box=None)
    table.add_column("code", style="bold cyan")
    table.add_column("country")
    table.add_column("script")
    table.add_column("layouts", justify="right")
    table.add_column("engine chain", style="dim")
    for spec in registry.all_specs():
        table.add_row(
            spec.code,
            spec.name_en,
            spec.script,
            str(len(spec.layouts)),
            " → ".join(spec.engine_chain),
        )
    console.print(table)
    errors = registry.load_errors()
    if errors:
        console.print("\n[yellow]files that failed to load:[/yellow]")
        for err in errors:
            console.print(f"  {err}")
    console.print(
        "\n[dim]Add your own: cp configs/countries/_template.yaml configs/countries/xx.yaml"
        "  (see docs/ADDING_A_COUNTRY.md)[/dim]"
    )


# ---------------------------------------------------------------------------
@app.command()
def engines() -> None:
    """List OCR engines and whether they are installed."""
    from pelakx import ocr

    table = Table(title="OCR engines", box=None)
    table.add_column("id", style="bold cyan")
    table.add_column("backend")
    table.add_column("scripts")
    table.add_column("installed")
    table.add_column("install with", style="dim")
    for engine_id, cls in sorted(ocr.registered().items()):
        table.add_row(
            engine_id,
            cls.label,
            ", ".join(cls.scripts),
            _ok(cls.is_available()),
            cls.install_hint,
        )
    console.print(table)


# ---------------------------------------------------------------------------
@app.command()
def parse(
    text: str = typer.Argument(..., help="Raw plate text, as an OCR engine would return it."),
    country: str = typer.Option("IR", "--country", "-c", help="Country code, or 'auto'."),
    confidence: float = typer.Option(0.9, "--confidence", help="Pretend OCR confidence."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Parse a plate string through the grammar engine — no video, no models.

    The fastest way to see what PelakX actually does:

        pelakx parse "۱۲ ب ۳۴۵ ایران ۱۱"
        pelakx parse "A81ZCDE" --country GB
        pelakx parse "1234BCD" --country auto
    """
    from pelakx.grammar import identify_country, registry
    from pelakx.grammar import parse as parse_plate

    if country.lower() == "auto":
        read = identify_country(text, registry.all_specs(), ocr_confidence=confidence)
    else:
        read = parse_plate(text, registry.get(country), ocr_confidence=confidence)

    if read is None:
        console.print("[red]no reading[/red] — the text folded to nothing or matched no layout")
        raise typer.Exit(code=1)

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "canonical": read.canonical,
                    "display": read.display,
                    "country": read.country,
                    "layout": read.layout_id,
                    "confidence": read.confidence,
                    "ocr_confidence": read.ocr_confidence,
                    "repairs": read.repairs,
                    "valid": read.valid,
                    "fields": read.fields,
                },
                ensure_ascii=False,
            )
        )
        return

    status = "[green]valid[/green]" if read.valid else "[yellow]unvalidated[/yellow]"
    body = [
        f"[bold]{read.display}[/bold]",
        "",
        f"canonical   {read.canonical}",
        f"country     {read.country}   layout: {read.layout_id or '—'}",
        f"confidence  {read.confidence:.1%}   (ocr {read.ocr_confidence:.0%}, "
        f"{read.repairs} repair{'s' if read.repairs != 1 else ''})",
        f"status      {status}",
    ]
    if read.fields:
        body.append("")
        body.append("fields")
        for key, value in read.fields.items():
            body.append(f"  {key:<10} {value}")
    console.print(Panel("\n".join(body), title=f"input: {text}", border_style="cyan"))


# ---------------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Check the environment and say exactly what is missing."""
    import importlib.util

    from pelakx import ocr
    from pelakx.grammar import registry

    console.print(Panel(f"PelakX {__version__}  ·  python {sys.version.split()[0]}", border_style="cyan"))

    table = Table(title="dependencies", box=None)
    table.add_column("package")
    table.add_column("installed")
    table.add_column("needed for", style="dim")
    for module, purpose in [
        ("cv2", "video I/O and image ops (required)"),
        ("numpy", "arrays (required)"),
        ("yaml", "country grammars (required)"),
        ("ultralytics", "vehicle + plate detection (YOLO26)"),
        ("torch", "ultralytics backend"),
        ("onnxruntime", "fast CPU inference"),
        ("fast_plate_ocr", "Latin-plate OCR"),
        ("open_image_models", "zero-setup plate detector"),
        ("hezar", "Persian plate OCR"),
        ("arabic_reshaper", "Persian text rendering"),
        ("bidi", "right-to-left text rendering"),
        ("PIL", "label rendering"),
        ("paddleocr", "generic multi-script OCR"),
        ("streamlit", "dashboard"),
        ("pandas", "dashboard tables"),
    ]:
        table.add_row(module, _ok(importlib.util.find_spec(module) is not None), purpose)
    console.print(table)

    # torch / GPU
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU only"
        console.print(f"\ntorch {torch.__version__} · CUDA: {_ok(cuda)} · device: {device}")
        if not cuda:
            console.print(
                "[dim]  CPU-only is fine — use yolo26n and the ONNX engines; "
                "see docs/MODELS.md for the CPU profile.[/dim]"
            )
    except ImportError:
        console.print("\n[yellow]torch not installed — detection is unavailable[/yellow]")

    console.print(f"\ncountry grammars: [bold]{len(registry.codes())}[/bold] "
                  f"({', '.join(registry.codes())})")
    installed = ocr.available()
    console.print(f"OCR engines ready: [bold]{', '.join(installed) or 'none'}[/bold]")

    font_path = None
    try:
        from pelakx.render import find_font

        font_path = find_font(None)
    except Exception:
        pass
    console.print(f"RTL-capable font: {font_path or '[yellow]not found[/yellow]'}")

    if len(installed) <= 1:
        console.print(
            "\n[yellow]Only the built-in test engine is available.[/yellow]\n"
            "  pip install 'pelakx[onnx]'   # Latin plates, fast on CPU\n"
            "  pip install 'pelakx[fa]'     # Persian plates"
        )


# ---------------------------------------------------------------------------
@app.command()
def run(
    source: str = typer.Argument(..., help="Video file, RTSP URL, or webcam index."),
    country: str = typer.Option("IR", "--country", "-c"),
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config file."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory."),
    engine: Optional[str] = typer.Option(None, "--engine", help="Force an OCR engine."),
    vehicle_weights: Optional[str] = typer.Option(None, "--vehicle-weights"),
    plate_weights: Optional[str] = typer.Option(None, "--plate-weights"),
    device: Optional[str] = typer.Option(None, "--device", help="cpu, cuda:0, mps…"),
    stride: Optional[int] = typer.Option(None, "--stride", help="Process every Nth frame."),
    max_frames: Optional[int] = typer.Option(None, "--max-frames"),
    no_video: bool = typer.Option(False, "--no-video", help="Skip writing annotated video."),
    privacy: bool = typer.Option(
        False, "--privacy", help="Blur plates, pseudonymise exports, keep no crops."
    ),
    watch: Optional[str] = typer.Option(
        None, "--watch", help="Comma-separated watchlist entries (supports 12ب* and re:…)."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Process a video and produce annotated output, exports and analytics."""
    from pelakx.config import PipelineConfig
    from pelakx.pipeline import Pipeline

    cfg = PipelineConfig.from_yaml(config) if config else PipelineConfig()
    overrides: dict[str, object] = {"country": country.upper()}
    if out is not None:
        overrides["output.dir"] = str(out)
    if engine:
        overrides["ocr.engine"] = engine
    if vehicle_weights:
        overrides["vehicle.weights"] = vehicle_weights
    if plate_weights:
        overrides["plate.weights"] = plate_weights
    if device:
        overrides["vehicle.device"] = device
        overrides["plate.device"] = device
    if stride:
        overrides["frame_stride"] = stride
    if max_frames:
        overrides["max_frames"] = max_frames
    if no_video:
        overrides["output.video"] = False
    if watch:
        overrides["analytics.watchlist"] = [w.strip() for w in watch.split(",") if w.strip()]
    if privacy:
        overrides["privacy.blur_plates"] = True
        overrides["privacy.hash_plates"] = True
        overrides["privacy.no_crops"] = True
        from pelakx.privacy import generate_salt

        if not cfg.privacy.hash_salt:
            overrides["privacy.hash_salt"] = generate_salt()
    cfg = cfg.merged(**overrides)

    pipeline = Pipeline(cfg)
    if not quiet:
        console.print(
            Panel(
                f"[bold]{cfg.country}[/bold] · vehicles={cfg.vehicle.weights} · "
                f"ocr={pipeline.ocr.id} · device={cfg.vehicle.device}\n"
                f"source: {source}",
                title="PelakX run",
                border_style="cyan",
            )
        )
        if privacy:
            console.print("[yellow]privacy mode:[/yellow] plates blurred, exports pseudonymised")

    if quiet:
        summary = pipeline.run(source)
    else:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), console=console
        ) as bar:
            task = bar.add_task("processing", total=None)

            def tick(s) -> None:
                bar.update(
                    task,
                    description=(
                        f"frame {s.frames_processed}  ·  {s.fps:.1f} fps  ·  "
                        f"{s.vehicles} vehicles  ·  {s.plates_read} reads"
                    ),
                )

            summary = pipeline.run(source, progress=tick)

    if not quiet:
        _print_summary(summary)


def _print_summary(summary) -> None:
    table = Table(title="run summary", box=None)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("frames processed", str(summary.frames_processed))
    table.add_row("throughput", f"{summary.fps:.1f} fps")
    table.add_row("vehicles tracked", str(summary.vehicles))
    table.add_row("plate readings", str(summary.plates_read))
    table.add_row("OCR calls", str(summary.ocr_calls))
    table.add_row("crops skipped by gate", f"{summary.crops_gated} ({summary.ocr_savings:.0%})")
    table.add_row("events exported", str(len(summary.events)))
    if summary.alerts:
        table.add_row("[red]watchlist alerts[/red]", str(summary.alerts))
    console.print(table)

    identified = [e for e in summary.events if e.plate]
    if identified:
        plates = Table(title="plates", box=None)
        plates.add_column("track", justify="right")
        plates.add_column("plate")
        plates.add_column("conf", justify="right")
        plates.add_column("class")
        plates.add_column("speed", justify="right")
        plates.add_column("alerts", style="red")
        for event in sorted(identified, key=lambda e: -(e.plate.confidence))[:20]:
            plates.add_row(
                str(event.track_id),
                event.plate.display,
                f"{event.plate.confidence:.0%}",
                event.class_name,
                f"{event.speed_kmh:.0f}" if event.speed_kmh else "—",
                ", ".join(event.alerts),
            )
        console.print(plates)

    if summary.line_counts:
        console.print("\n[bold]line crossings[/bold]")
        for name, counts in summary.line_counts.items():
            console.print(f"  {name}: → {counts['forward']}  ← {counts['backward']}")
    if summary.speed.get("count"):
        console.print(f"\n[bold]speed[/bold]  {summary.speed}")
    if summary.outputs:
        console.print("\n[bold]outputs[/bold]")
        for kind, path in summary.outputs.items():
            console.print(f"  {kind:<7} {path}")


# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str = typer.Argument("", help="Plate text; '*' works as a wildcard."),
    db: Path = typer.Option(Path("outputs/pelakx.sqlite"), "--db"),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
    min_speed: Optional[float] = typer.Option(None, "--min-speed"),
    country: Optional[str] = typer.Option(None, "--country"),
    valid_only: bool = typer.Option(False, "--valid-only"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Search a previous run's results.

        pelakx search "12ب*"
        pelakx search --min-speed 90 --valid-only
    """
    from pelakx.store import EventStore

    if not db.exists():
        console.print(f"[red]no database at {db}[/red] — run `pelakx run` first")
        raise typer.Exit(code=1)
    with EventStore(db) as store:
        rows = store.search(
            query,
            min_confidence=min_confidence,
            min_speed=min_speed,
            country=country,
            valid_only=valid_only,
            limit=limit,
        )
    if not rows:
        console.print("[yellow]no matches[/yellow]")
        return
    table = Table(title=f"{len(rows)} match(es)", box=None)
    for column in ("track_id", "plate_display", "confidence", "vehicle_class", "speed_kmh", "first_seen", "alerts"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row["track_id"]),
            str(row["plate_display"] or ""),
            f"{(row['confidence'] or 0):.0%}",
            str(row["vehicle_class"] or ""),
            f"{row['speed_kmh']:.0f}" if row["speed_kmh"] else "—",
            f"{(row['first_seen'] or 0):.1f}s",
            str(row["alerts"] or ""),
        )
    console.print(table)


# ---------------------------------------------------------------------------
@app.command("new-country")
def new_country(
    code: str = typer.Argument(..., help="ISO-3166 alpha-2 code, e.g. PK"),
    name: str = typer.Option("", "--name", help="English country name."),
    directory: Optional[Path] = typer.Option(None, "--dir", help="Where to write the YAML."),
) -> None:
    """Scaffold a new country grammar from the template."""
    target_dir = directory or (Path(__file__).resolve().parents[2] / "configs" / "countries")
    target_dir.mkdir(parents=True, exist_ok=True)
    template = target_dir / "_template.yaml"
    if not template.exists():
        template = Path(__file__).resolve().parents[2] / "configs" / "countries" / "_template.yaml"
    if not template.exists():
        console.print("[red]_template.yaml not found[/red]")
        raise typer.Exit(code=1)

    target = target_dir / f"{code.lower()}.yaml"
    if target.exists():
        console.print(f"[red]{target} already exists[/red]")
        raise typer.Exit(code=1)
    shutil.copyfile(template, target)
    content = target.read_text(encoding="utf-8")
    content = content.replace("code: XX", f"code: {code.upper()}")
    if name:
        content = content.replace("name_en: Country name", f"name_en: {name}")
    target.write_text(content, encoding="utf-8")

    console.print(
        Panel(
            f"created [bold]{target}[/bold]\n\n"
            "next:\n"
            "  1. fill in `alphabet` and `layouts` for your country\n"
            "  2. pelakx parse \"<a real plate>\" --country " + code.upper() + "\n"
            "  3. add a test to tests/test_grammar.py and open a PR\n\n"
            "guide: docs/ADDING_A_COUNTRY.md",
            title="new country grammar",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
@app.command("download-models")
def download_models(
    dest: Path = typer.Option(Path("models"), "--dest"),
) -> None:
    """Fetch ready-made plate-detection weights into ``models/``."""
    from pelakx.models_hub import download_default_plate_detector

    path = download_default_plate_detector(dest)
    console.print(f"[green]ready[/green] {path}")


# ---------------------------------------------------------------------------
@app.command()
def dashboard(
    db: Path = typer.Option(Path("outputs/pelakx.sqlite"), "--db"),
    port: int = typer.Option(8501, "--port"),
) -> None:
    """Launch the interactive results dashboard (Streamlit)."""
    import subprocess

    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port), "--", "--db", str(db),
    ]
    console.print(f"[cyan]launching dashboard on http://localhost:{port}[/cyan]")
    raise typer.Exit(code=subprocess.call(cmd))


if __name__ == "__main__":  # pragma: no cover
    app()
