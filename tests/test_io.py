"""Regression tests for ``lysosense.io`` parsing.

Exercises ``parse_dat_bytes`` and ``load_dat_file`` against synthetic
.dat payloads covering comma- and whitespace-separated data, header
precedence, metadata coercion, sorting, column names, and the empty
payload error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lysosense import load_dat_file, parse_dat_bytes


def test_parse_comma_separated_data_basic():
    """Comma-separated lines parse into the expected DataFrame columns."""

    payload = b"0.48, 1.0\n0.85, 2.0\n1.20, 3.0\n"
    m = parse_dat_bytes(payload, "demo")

    assert m.name == "demo"
    assert m.source == "demo"
    assert list(m.data.columns) == ["particle_size_um", "mass_signal_ug"]
    assert len(m.data) == 3
    assert m.data["particle_size_um"].tolist() == [0.48, 0.85, 1.20]
    assert m.data["mass_signal_ug"].tolist() == [1.0, 2.0, 3.0]


def test_parse_whitespace_separated_data():
    """Whitespace-separated numeric lines are also recognised as data."""

    payload = b"0.48 1.0\n0.85 2.0\n"
    m = parse_dat_bytes(payload, "ws")

    assert m.data["particle_size_um"].tolist() == [0.48, 0.85]
    assert m.data["mass_signal_ug"].tolist() == [1.0, 2.0]


def test_parse_sorts_by_particle_size_ascending():
    """Out-of-order rows are sorted ascending on particle size."""

    payload = b"1.20, 3.0\n0.48, 1.0\n0.85, 2.0\n"
    m = parse_dat_bytes(payload, "unordered")

    assert m.data["particle_size_um"].is_monotonic_increasing
    assert m.data["particle_size_um"].tolist() == [0.48, 0.85, 1.20]
    # The signal values travel with their size values
    assert m.data["mass_signal_ug"].tolist() == [1.0, 2.0, 3.0]


def test_parse_strips_header_lines_before_data():
    """Non-numeric leading lines are captured as metadata, not data."""

    payload = b"Instrument Profile\n0.48, 1.0\n0.85, 2.0\n"
    m = parse_dat_bytes(payload, "with_header")

    assert len(m.data) == 2
    # First header line maps to the known instrument_profile label
    assert m.metadata["instrument_profile"] == "Instrument Profile"
    assert m.metadata["total_points"] == 2


def test_parse_metadata_type_coercion():
    """Metadata values are coerced to int / float / str in that order."""

    # Three header lines -> indices 0/1/2 map to instrument_profile,
    # laser_wavelength, grating_setting per _METADATA_LABELS.
    payload = b"1263\n2.5\nsome text\n0.48, 1.0\n"
    m = parse_dat_bytes(payload, "coerce")

    # int
    assert m.metadata["instrument_profile"] == 1263
    assert isinstance(m.metadata["instrument_profile"], int)
    # float
    assert m.metadata["laser_wavelength"] == 2.5
    assert isinstance(m.metadata["laser_wavelength"], float)
    # str (falls through when neither int nor float parses)
    assert m.metadata["grating_setting"] == "some text"
    assert isinstance(m.metadata["grating_setting"], str)


def test_parse_handles_blank_lines():
    """Blank lines are skipped without affecting the result."""

    payload = b"\n0.48, 1.0\n\n0.85, 2.0\n\n"
    m = parse_dat_bytes(payload, "blanks")

    assert len(m.data) == 2
    assert m.data["particle_size_um"].tolist() == [0.48, 0.85]


def test_parse_empty_payload_raises_value_error():
    """A payload with no numeric data raises ValueError."""

    with pytest.raises(ValueError, match="No numeric samples"):
        parse_dat_bytes(b"just\nsome\nheader\nlines\n", "empty")


def test_parse_header_only_payload_raises_value_error():
    """Header-only payloads (no data lines) also raise."""

    with pytest.raises(ValueError):
        parse_dat_bytes(b"Instrument Profile\nOperator\n", "headers")


def test_parse_uses_sample_name_metadata_as_name():
    """When a sample_name metadata field is present it becomes the name."""

    # 14 non-blank filler header lines so "My Sample" lands at index 14,
    # which _METADATA_LABELS maps to sample_name. Blank lines are stripped,
    # so they must be real tokens here.
    header = b"".join(f"filler{i}\n".encode() for i in range(14))
    payload = header + b"My Sample\n0.48, 1.0\n"
    m = parse_dat_bytes(payload, "src")

    assert m.metadata["sample_name"] == "My Sample"
    assert m.name == "My Sample"


def test_data_is_float64_typed():
    """Parsed data columns are float64 regardless of input formatting."""

    payload = b"0.48, 1\n0.85, 2\n"
    m = parse_dat_bytes(payload, "types")

    assert str(m.data["particle_size_um"].dtype) == "float64"
    assert str(m.data["mass_signal_ug"].dtype) == "float64"


def test_load_dat_file_roundtrip(tmp_path: Path):
    """load_dat_file reads a written .dat and produces the same DataFrame."""

    dat = tmp_path / "sample.dat"
    dat.write_text("0.48, 1.0\n0.85, 2.0\n1.20, 3.0\n", encoding="utf-8")

    m = load_dat_file(dat)

    assert m.source == "sample"
    assert isinstance(m.data, pd.DataFrame)
    assert m.data["particle_size_um"].tolist() == [0.48, 0.85, 1.20]
    assert m.data["mass_signal_ug"].tolist() == [1.0, 2.0, 3.0]


def test_load_dat_file_accepts_string_path(tmp_path: Path):
    """load_dat_file accepts a str path in addition to Path."""

    dat = tmp_path / "strpath.dat"
    dat.write_text("0.48, 1.0\n0.85, 2.0\n", encoding="utf-8")

    m = load_dat_file(str(dat))

    assert len(m.data) == 2
