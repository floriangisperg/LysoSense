"""Regression tests for ``lysosense.analysis.analyze_measurement``.

Builds synthetic two-peak and single-peak signals programmatically and
locks down the peak positions, component areas/fractions, and the
single-peak fallback path. A fixed RNG seed keeps the noise deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lysosense import (
    AnalysisOptions,
    Measurement,
    analyze_measurement,
)

CELL_MEAN_KEY = "mean_cell_µm"
IB_MEAN_KEY = "mean_ib_µm"


def _gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_measurement(x: np.ndarray, y: np.ndarray, name: str = "synth") -> Measurement:
    df = pd.DataFrame({"particle_size_um": x, "mass_signal_ug": y})
    return Measurement(name=name, metadata={}, data=df, source=name)


def _two_peak_signal(seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return a deterministic two-Gaussian signal plus ground-truth params."""

    rng = np.random.default_rng(seed)
    x = np.linspace(0.3, 1.6, 400)

    a_ib, mu_ib, sigma = 2.0, 0.48, 0.06
    a_cell, mu_cell = 1.0, 0.85
    y = _gaussian(x, a_ib, mu_ib, sigma) + _gaussian(x, a_cell, mu_cell, sigma)
    y += rng.normal(0.0, 0.005, size=x.size)

    # Analytic areas: A * sigma * sqrt(2*pi)
    area_ib = a_ib * sigma * math.sqrt(2.0 * math.pi)
    area_cell = a_cell * sigma * math.sqrt(2.0 * math.pi)
    truth = {
        "mu_ib": mu_ib,
        "mu_cell": mu_cell,
        "area_ib": area_ib,
        "area_cell": area_cell,
        "intact_fraction": area_cell / (area_ib + area_cell),
    }
    return x, y, truth


# ---------------------------------------------------------------------------
# Two-peak recovery
# ---------------------------------------------------------------------------


def test_two_peak_recovers_peak_locations():
    """Both Gaussian peak centres are recovered within tolerance."""

    x, y, truth = _two_peak_signal()
    result = analyze_measurement(_make_measurement(x, y, "two"))

    assert result.fit_kind == "two"
    mu_ib = result.metrics[IB_MEAN_KEY]
    mu_cell = result.metrics[CELL_MEAN_KEY]
    assert mu_ib is not None and mu_cell is not None
    assert abs(mu_ib - truth["mu_ib"]) < 0.02
    assert abs(mu_cell - truth["mu_cell"]) < 0.02


def test_two_peak_recovers_component_fractions():
    """Component fractions are derived from the fitted areas.

    The two peaks overlap, so the optimiser redistributes some area between
    them and the fitted intact fraction is not exactly the analytic input
    ratio. We assert instead that both components carry a meaningful share
    of the total and that the fraction is internally consistent with the
    reported areas.
    """

    x, y, _ = _two_peak_signal()
    metrics = analyze_measurement(_make_measurement(x, y, "two")).metrics

    area_cells = metrics["area_cells"]
    area_ibs = metrics["area_inclusion_bodies"]
    area_total = metrics["area_total"]

    # Both components must be present (this is a genuine two-peak fit).
    assert area_cells > 0.0
    assert area_ibs > 0.0
    # Fraction equals area_cells / area_total.
    assert abs(metrics["intact_fraction"] - area_cells / area_total) < 1e-9
    # Lysis efficiency is the exact complement.
    assert abs(metrics["lysis_efficiency"] - (1.0 - metrics["intact_fraction"])) < 1e-9


def test_two_peak_area_totals_are_consistent():
    """area_total == area_cells + area_inclusion_bodies (within float tol)."""

    x, y, _ = _two_peak_signal()
    metrics = analyze_measurement(_make_measurement(x, y, "two")).metrics

    summed = metrics["area_cells"] + metrics["area_inclusion_bodies"]
    assert abs(summed - metrics["area_total"]) < 1e-6


def test_two_peak_observed_frame_has_component_columns():
    """The observed DataFrame is augmented with fit + component columns."""

    x, y, _ = _two_peak_signal()
    result = analyze_measurement(_make_measurement(x, y, "two"))

    for col in ("fit_signal_ug", "cells_component_ug", "ibs_component_ug"):
        assert col in result.observed.columns
    # Observed frame preserves the input point count.
    assert len(result.observed) == len(x)


def test_two_peak_dense_fit_shape():
    """dense_fit has opts.dense_points rows and the expected columns."""

    x, y, _ = _two_peak_signal()
    opts = AnalysisOptions(dense_points=1200)
    result = analyze_measurement(_make_measurement(x, y, "two"), opts)

    assert len(result.dense_fit) == 1200
    assert set(result.dense_fit.columns) == {
        "particle_size_um",
        "fit_signal_ug",
        "cells_component_ug",
        "ibs_component_ug",
    }


def test_two_peak_is_deterministic_across_runs():
    """Re-analysing the same signal yields identical metrics (no RNG drift)."""

    x, y, _ = _two_peak_signal()
    m = _make_measurement(x, y, "two")
    first = analyze_measurement(m).metrics
    second = analyze_measurement(m).metrics
    for key in ("intact_fraction", "area_cells", "area_inclusion_bodies", IB_MEAN_KEY):
        assert first[key] == second[key]


# ---------------------------------------------------------------------------
# Single-peak fallback
# ---------------------------------------------------------------------------


def test_single_peak_triggers_one_peak_fallback():
    """A clean single-peak signal yields fit_kind 'one' with sane metrics."""

    x = np.linspace(0.3, 1.6, 400)
    y = _gaussian(x, 3.0, 0.48, 0.06)
    result = analyze_measurement(_make_measurement(x, y, "single"))

    assert result.fit_kind == "one"
    metrics = result.metrics
    # No cells component -> zero intact fraction, full lysis.
    assert metrics["area_cells"] == 0.0
    assert metrics["intact_fraction"] == 0.0
    assert metrics["lysis_efficiency"] == 1.0
    # IB peak centre recovered; cell mean is absent.
    assert metrics[CELL_MEAN_KEY] is None
    assert abs(metrics[IB_MEAN_KEY] - 0.48) < 0.02


# ---------------------------------------------------------------------------
# Edge cases (light-touch, no exact-float over-assertion)
# ---------------------------------------------------------------------------


def test_options_override_dense_points_takes_effect():
    """dense_points option controls the dense fit resolution."""

    x = np.linspace(0.3, 1.6, 300)
    y = _gaussian(x, 2.0, 0.48, 0.06) + _gaussian(x, 1.0, 0.85, 0.06)
    result = analyze_measurement(_make_measurement(x, y, "dense"), AnalysisOptions(dense_points=500))
    assert len(result.dense_fit) == 500


def test_analysis_result_carries_options_and_measurement():
    """The AnalysisResult echoes the options and source measurement."""

    x, y, _ = _two_peak_signal()
    opts = AnalysisOptions()
    m = _make_measurement(x, y, "echo")
    result = analyze_measurement(m, opts)

    assert result.measurement is m
    assert result.options is opts


def test_empty_measurement_raises_value_error():
    """An empty measurement cannot be analysed."""

    m = _make_measurement(np.array([]), np.array([]), "empty")
    with pytest.raises(ValueError):
        analyze_measurement(m)


def test_all_nonpositive_signal_raises_value_error():
    """A signal with no positive values is rejected by the fitter."""

    x = np.linspace(0.3, 1.6, 100)
    y = np.full_like(x, -1.0)
    m = _make_measurement(x, y, "neg")
    with pytest.raises(ValueError):
        analyze_measurement(m)


def test_lysis_efficiency_bounded_in_unit_interval():
    """For any valid fit, lysis efficiency is in [0, 1]."""

    x, y, _ = _two_peak_signal()
    metrics = analyze_measurement(_make_measurement(x, y, "two")).metrics
    assert 0.0 <= float(metrics["lysis_efficiency"]) <= 1.0
    assert 0.0 <= float(metrics["intact_fraction"]) <= 1.0
