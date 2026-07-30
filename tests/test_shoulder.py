"""Tests for the objective shoulder-detection diagnostic.

Synthetic-only (no dependency on the gitignored ``data/``) so the suite passes in
CI. Each test pins one verdict: clean single peak -> 'none', a real shoulder or
resolved second peak -> 'shoulder', a cell-dominant trace -> 'n/a'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lysosense import AnalysisOptions, Measurement, analyze_measurement


def _gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _measurement(x: np.ndarray, y: np.ndarray, name: str = "synth") -> Measurement:
    df = pd.DataFrame({"particle_size_um": x, "mass_signal_ug": y})
    return Measurement(name=name, metadata={}, data=df, source=name)


def test_clean_single_peak_is_none():
    x = np.linspace(0.2, 1.2, 700)
    y = _gaussian(x, 10.0, 0.48, 0.06)
    result = analyze_measurement(_measurement(x, y))
    assert result.metrics["shoulder_verdict"] == "none"


def test_noisy_clean_peak_is_not_a_false_shoulder():
    """Instrument noise on a single peak must not manufacture a 'shoulder'."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.2, 1.2, 700)
    y = _gaussian(x, 10.0, 0.48, 0.06) + rng.normal(0.0, 0.02, size=x.size)
    result = analyze_measurement(_measurement(x, y))
    assert result.metrics["shoulder_verdict"] != "shoulder"


def test_real_shoulder_is_detected():
    x = np.linspace(0.2, 1.3, 700)
    y = _gaussian(x, 20.0, 0.70, 0.10) + _gaussian(x, 5.0, 1.00, 0.08)
    result = analyze_measurement(
        _measurement(x, y), AnalysisOptions(mu_ib_um=0.70, mu_cell_um=1.00)
    )
    assert result.metrics["shoulder_verdict"] == "shoulder"
    assert result.metrics["shoulder_excess_sigma"] >= 1.5


def test_resolved_second_peak_is_detected_as_shoulder():
    x = np.linspace(0.3, 1.6, 400)
    y = _gaussian(x, 2.0, 0.48, 0.06) + _gaussian(x, 1.0, 0.85, 0.06)
    result = analyze_measurement(_measurement(x, y))
    assert result.metrics["shoulder_verdict"] == "shoulder"


def test_cell_dominant_trace_is_not_applicable():
    """A lone peak at/after the cell target has no IB peak to define a shoulder from."""
    x = np.linspace(0.2, 1.4, 600)
    y = _gaussian(x, 60.0, 1.035, 0.075)
    result = analyze_measurement(
        _measurement(x, y), AnalysisOptions(mu_ib_um=0.70, mu_cell_um=1.00)
    )
    assert result.metrics["shoulder_verdict"] == "n/a"


def test_debris_dominant_trace_is_not_applicable():
    """A dominant low-size debris peak (out-of-framework sample) is not a shoulder case.

    Mirrors the real false-positive: refold/solution samples whose signal max
    lands at the low-size edge instead of at an IB peak. The verdict must be n/a,
    not a spurious 'shoulder' from a meaningless tail extrapolation.
    """
    x = np.linspace(0.2, 1.2, 700)
    y = _gaussian(x, 30.0, 0.25, 0.05) + _gaussian(x, 1.0, 0.90, 0.08)
    result = analyze_measurement(
        _measurement(x, y), AnalysisOptions(mu_ib_um=0.70, mu_cell_um=1.00)
    )
    assert result.metrics["shoulder_verdict"] == "n/a"


def test_shoulder_keys_always_present():
    x = np.linspace(0.3, 1.6, 400)
    y = _gaussian(x, 2.0, 0.48, 0.06) + _gaussian(x, 1.0, 0.85, 0.06)
    metrics = analyze_measurement(_measurement(x, y)).metrics
    assert "shoulder_verdict" in metrics
    assert "shoulder_excess_sigma" in metrics
