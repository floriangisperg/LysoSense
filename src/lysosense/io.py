"""Utilities for parsing instrument .dat files into structured measurements."""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


_DATA_SPLIT = re.compile(r"[\s,]+")

_METADATA_LABELS: Dict[int, str] = {
    0: "instrument_profile",
    1: "laser_wavelength",
    2: "grating_setting",
    3: "optics_configuration",
    4: "integration_time",
    5: "laser_power",
    6: "aperture_setting",
    7: "acquisition_program",
    8: "accumulations",
    9: "objective_magnification",
    10: "polarization_state",
    11: "dark_correction",
    12: "calibration_id",
    13: "operator_id",
    14: "sample_name",
    15: "measurement_date",
    16: "measurement_time",
    17: "declared_points",
    18: "scan_mode",
    19: "scan_direction",
    20: "start_shift_hint",
}


@dataclass
class Measurement:
    """Container for a parsed measurement."""

    name: str
    metadata: Dict[str, Any]
    data: pd.DataFrame
    source: str
    notes: List[str] = field(default_factory=list)


def list_dat_files(data_dir: Path) -> List[Path]:
    """Return all .dat files inside *data_dir* sorted by name."""

    if not data_dir.exists():
        return []
    return sorted(p for p in data_dir.glob("*.dat") if p.is_file())


def load_dat_file(path: Path | str) -> Measurement:
    """Load a .dat file from disk."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="ignore")
    return _parse_dat_text(text, source_name=path.stem)


def parse_dat_bytes(payload: bytes, source_name: str) -> Measurement:
    """Parse a .dat payload that was provided via upload."""

    text = payload.decode("utf-8", errors="ignore")
    return _parse_dat_text(text, source_name=source_name)


def _parse_dat_text(text: str, source_name: str | None = None) -> Measurement:
    header_lines: List[str] = []
    data_points: List[tuple[float, float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_data_line(line):
            parsed = _parse_data_line(line)
            if parsed is not None:
                data_points.append(parsed)
        else:
            header_lines.append(line)

    if not data_points:
        raise ValueError("No numeric samples found in provided .dat payload")

    metadata = _build_metadata(header_lines)
    df = pd.DataFrame(data_points, columns=["wavenumber", "intensity"])
    df = df.dropna()
    df = df.astype({"wavenumber": "float64", "intensity": "float64"})
    df = df.sort_values("wavenumber").reset_index(drop=True)

    name = str(metadata.get("sample_name") or source_name or "measurement")
    source = source_name or name
    metadata["total_points"] = len(df)

    return Measurement(name=name, metadata=metadata, data=df, source=source)


def _looks_like_data_line(line: str) -> bool:
    if "," in line:
        return True
    tokens = [token for token in line.split() if token]
    if len(tokens) != 2:
        return False
    return _is_float(tokens[0]) and _is_float(tokens[1])


def _parse_data_line(line: str) -> tuple[float, float] | None:
    tokens = [token for token in _DATA_SPLIT.split(line) if token]
    if len(tokens) < 2:
        return None
    try:
        return float(tokens[0]), float(tokens[1])
    except ValueError:
        return None


def _build_metadata(lines: Sequence[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for idx, value in enumerate(lines):
        label = _METADATA_LABELS.get(idx, f"line_{idx + 1:02d}")
        metadata[label] = _coerce(value)
    return metadata


def _coerce(value: str) -> Any:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
