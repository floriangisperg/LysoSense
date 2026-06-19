"""Tests for lysosense.transforms: pure measurement transforms and R².

These functions were extracted verbatim from app/streamlit_app.py so the
quantification helpers can be tested without importing Streamlit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lysosense.io import Measurement
from lysosense.transforms import (
    NormalizationSkipped,
    calculate_r_squared,
    clip_measurement_range,
    normalize_measurement,
    subtract_baseline,
)


def _measurement(y, name: str = "m") -> Measurement:
    """Build a Measurement whose particle sizes span 0.1-2.0 um."""
    x = np.linspace(0.1, 2.0, len(y))
    df = pd.DataFrame(
        {"particle_size_um": x, "mass_signal_ug": np.asarray(y, dtype=float)}
    )
    return Measurement(name=name, metadata={}, data=df, source=name, notes=[])


# --- calculate_r_squared -----------------------------------------------------


def test_r_squared_perfect_fit_is_one():
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = pd.DataFrame({"mass_signal_ug": y, "fit_signal_ug": y})
    assert calculate_r_squared(SimpleNamespace(observed=df)) == pytest.approx(1.0)


def test_r_squared_zero_variance_returns_zero():
    df = pd.DataFrame(
        {"mass_signal_ug": [3.0, 3.0, 3.0], "fit_signal_ug": [3.0, 3.0, 3.0]}
    )
    assert calculate_r_squared(SimpleNamespace(observed=df)) == 0.0


def test_r_squared_partial_fit_is_between_zero_and_one():
    df = pd.DataFrame(
        {
            "mass_signal_ug": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fit_signal_ug": [1.5, 2.5, 2.5, 3.5, 4.5],
        }
    )
    r2 = calculate_r_squared(SimpleNamespace(observed=df))
    assert 0.0 <= r2 < 1.0


# --- subtract_baseline -------------------------------------------------------


def test_subtract_baseline_minimum_shifts_min_to_zero():
    m = _measurement([5.0, 6.0, 8.0, 10.0])
    out = subtract_baseline(m, "minimum")
    assert out.data["mass_signal_ug"].min() == pytest.approx(0.0)
    assert out.metadata["baseline_method"] == "minimum"
    assert out.name.endswith("_baseline_corrected")
    # original is left untouched
    assert m.data["mass_signal_ug"].min() == pytest.approx(5.0)


def test_subtract_baseline_unknown_method_raises():
    with pytest.raises(ValueError):
        subtract_baseline(_measurement([1.0, 2.0]), "nope")


def test_subtract_baseline_empty_returns_same_object():
    empty = Measurement(
        name="e",
        metadata={},
        data=pd.DataFrame(columns=["particle_size_um", "mass_signal_ug"]),
        source="e",
    )
    assert subtract_baseline(empty) is empty


# --- normalize_measurement ---------------------------------------------------


def test_normalize_scales_max_to_one():
    out = normalize_measurement(_measurement([0.0, 5.0, 10.0]))
    assert out.data["mass_signal_ug"].max() == pytest.approx(1.0)
    assert out.metadata["normalized"] is True
    assert out.metadata["normalization_factor"] == pytest.approx(10.0)


def test_normalize_nonpositive_raises_with_original_message():
    with pytest.raises(NormalizationSkipped) as exc_info:
        normalize_measurement(_measurement([0.0, 0.0, 0.0], name="dead"))
    assert exc_info.value.factor == 0.0
    assert exc_info.value.name == "dead"
    # message must match the old inline st.warning text
    assert "skipping normalization for dead" in str(exc_info.value)


def test_normalize_negative_signal_raises():
    with pytest.raises(NormalizationSkipped):
        normalize_measurement(_measurement([-1.0, -2.0]))


# --- clip_measurement_range --------------------------------------------------


def test_clip_restricts_to_window():
    out = clip_measurement_range(_measurement([10.0, 20.0, 30.0, 40.0]), 0.5, 1.5)
    x = out.data["particle_size_um"]
    assert (x >= 0.5).all() and (x <= 1.5).all()
    assert out.metadata["clip_min_um"] == 0.5
    assert out.metadata["clip_max_um"] == 1.5


def test_clip_window_with_no_overlap_returns_original():
    m = _measurement([10.0, 20.0])  # sizes span 0.1-2.0 um
    assert clip_measurement_range(m, 5.0, 10.0) is m
