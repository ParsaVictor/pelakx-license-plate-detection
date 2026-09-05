"""The country registry — discovers and caches ``configs/countries/*.yaml``.

Adding a country to PelakX means dropping one YAML file into a search path.
No core code changes, no rebuild, no fork::

    export PELAKX_COUNTRIES=/etc/pelakx/countries
    pelakx countries          # your file is now listed

Search order (later paths win on a code collision):

1. ``<package>/../../configs/countries``  (repo checkout)
2. ``~/.pelakx/countries``                (per-user)
3. every path in ``$PELAKX_COUNTRIES``    (os.pathsep-separated)
4. anything registered at runtime via :func:`register_dir` / :func:`register_spec`
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from pelakx.grammar.spec import CountrySpec, GrammarError

_ENV_VAR = "PELAKX_COUNTRIES"
_TEMPLATE_STEM = "_template"

_extra_dirs: list[Path] = []
_runtime_specs: dict[str, CountrySpec] = {}


def _repo_config_dir() -> Path:
    # src/pelakx/grammar/registry.py -> <repo>/configs/countries
    return Path(__file__).resolve().parents[3] / "configs" / "countries"


def search_paths() -> list[Path]:
    """Every directory that is scanned for country YAML files, in order."""
    paths: list[Path] = [_repo_config_dir(), Path.home() / ".pelakx" / "countries"]
    env = os.environ.get(_ENV_VAR, "")
    paths.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p.strip())
    paths.extend(_extra_dirs)
    seen: dict[Path, None] = {}
    for p in paths:
        seen.setdefault(p, None)
    return list(seen)


def register_dir(path: str | os.PathLike[str]) -> None:
    """Add a directory of country YAMLs at runtime and invalidate the cache."""
    _extra_dirs.append(Path(path).expanduser())
    clear_cache()


def register_spec(spec: CountrySpec) -> None:
    """Register an in-memory spec (tests, notebooks, dynamic grammars)."""
    _runtime_specs[spec.code.upper()] = spec
    clear_cache()


def clear_cache() -> None:
    load_all.cache_clear()


@lru_cache(maxsize=1)
def load_all() -> dict[str, CountrySpec]:
    """Load every discoverable country spec, keyed by uppercase ISO-2 code."""
    specs: dict[str, CountrySpec] = {}
    errors: list[str] = []
    for directory in search_paths():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.y*ml")):
            if path.stem.startswith(_TEMPLATE_STEM):
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                spec = CountrySpec.from_dict(data, source_path=str(path))
            except (GrammarError, yaml.YAMLError, OSError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            specs[spec.code] = spec
    specs.update(_runtime_specs)
    if errors and not specs:  # pragma: no cover - misconfiguration
        raise GrammarError("no country grammars could be loaded:\n  " + "\n  ".join(errors))
    load_all.errors = errors  # type: ignore[attr-defined]
    return specs


def load_errors() -> list[str]:
    """Filenames that failed to parse during the last :func:`load_all`."""
    load_all()
    return list(getattr(load_all, "errors", []))


def get(code: str) -> CountrySpec:
    """Fetch one country spec by ISO-2 code (case-insensitive)."""
    specs = load_all()
    key = code.strip().upper()
    if key not in specs:
        available = ", ".join(sorted(specs)) or "<none>"
        raise KeyError(f"unknown country {code!r}. Available: {available}")
    return specs[key]


def has(code: str) -> bool:
    return code.strip().upper() in load_all()


def codes() -> list[str]:
    """All available country codes, sorted."""
    return sorted(load_all())


def all_specs() -> list[CountrySpec]:
    return [load_all()[c] for c in codes()]


def engines_for(code: str) -> tuple[str, ...]:
    """The OCR engine chain a country prefers."""
    return get(code).engine_chain
