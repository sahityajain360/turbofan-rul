"""Canonical project paths. Import these instead of hard-coding strings."""
from __future__ import annotations

from pathlib import Path

# src/pdm/paths.py -> parents[2] is the project root
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
CMAPSS_RAW = RAW / "CMAPSSData"
NCMAPSS_RAW = RAW / "ncmapss"  # N-CMAPSS HDF5 files (DS01..DS08)
NCMAPSS_DS02 = NCMAPSS_RAW / "N-CMAPSS_DS02-006.h5"  # the benchmark starter subset
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CONFIGS = ROOT / "configs"


def ensure_dirs() -> None:
    """Create the writable output dirs if missing (idempotent)."""
    for d in (INTERIM, PROCESSED, MODELS, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
