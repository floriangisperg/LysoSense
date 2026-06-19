"""Golden-master tests against real CPS Demo .dat files.

These lock down end-to-end parse + analyse metrics. They
``pytest.skip`` cleanly when the proprietary ``data/`` directory is
absent (e.g. a fresh CI clone), so they never fail there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lysosense import AnalysisOptions, analyze_measurement, load_dat_file

# Absolute path to the gitignored demo directory (proprietary sample data).
DEMO_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "CPS-FACS"
    / "CPS"
    / "Demo"
)

# Baseline metrics recorded from the current analyse_measurement implementation
# against the tracked Demo files (AnalysisOptions defaults). Floats are asserted
# with an absolute tolerance generous enough to absorb harmless BLAS / numpy
# minor-version differences but tight enough to catch real regressions.
GOLDEN: dict[str, dict[str, object]] = {
    "G7_CMC2_Homogenate_Pass_1_H11738.dat": {
        "fit_kind": "two",
        "intact_fraction": 0.584379737621651,
        "lysis_efficiency": 0.41562026237834904,
        "mean_ib_µm": 0.4743471780000001,
        "mean_cell_µm": 0.8305757459999998,
    },
    "G7_CMC2_IB_Seperation_1_Inclusion_Bodies_H11738.dat": {
        "fit_kind": "one",
        "intact_fraction": 0.0,
        "lysis_efficiency": 1.0,
        "mean_ib_µm": 0.4840554600000001,
        "mean_cell_µm": None,
    },
    "G7_CMC2_Resuspended_Biomass_H11738.dat": {
        "fit_kind": "two",
        "intact_fraction": 0.9922659703053512,
        "lysis_efficiency": 0.0077340296946487586,
        "mean_ib_µm": 0.5737646610006711,
        "mean_cell_µm": 0.8953214269999998,
    },
}

IB_KEY = "mean_ib_µm"
CELL_KEY = "mean_cell_µm"


def _require_demo_dir() -> Path:
    if not DEMO_DIR.is_dir():
        pytest.skip(f"Demo data directory absent: {DEMO_DIR}")
    return DEMO_DIR


@pytest.mark.parametrize("filename", sorted(GOLDEN))
def test_golden_demo_metrics(filename: str) -> None:
    """End-to-end metrics for each tracked Demo file stay within tolerance."""

    demo = _require_demo_dir()
    path = demo / filename
    if not path.exists():
        pytest.skip(f"{filename} not present in Demo directory")

    measurement = load_dat_file(path)
    result = analyze_measurement(measurement, AnalysisOptions())
    metrics = result.metrics
    expected = GOLDEN[filename]

    assert result.fit_kind == expected["fit_kind"]
    assert metrics["fit_kind"] == expected["fit_kind"]

    assert abs(metrics["intact_fraction"] - float(expected["intact_fraction"])) < 0.02
    assert abs(metrics["lysis_efficiency"] - float(expected["lysis_efficiency"])) < 0.02

    # Cell mean: None must stay None; otherwise stay within 0.05 um.
    expected_cell = expected[CELL_KEY]
    if expected_cell is None:
        assert metrics[CELL_KEY] is None
    else:
        assert metrics[CELL_KEY] is not None
        assert abs(metrics[CELL_KEY] - expected_cell) < 0.05
    # IB mean is always present for these files.
    assert metrics[IB_KEY] is not None
    assert abs(metrics[IB_KEY] - float(expected[IB_KEY])) < 0.05


def test_golden_areas_are_consistent() -> None:
    """area_total equals the sum of the two component areas for every file."""

    demo = _require_demo_dir()
    for filename in GOLDEN:
        path = demo / filename
        if not path.exists():
            continue
        metrics = analyze_measurement(load_dat_file(path), AnalysisOptions()).metrics
        summed = metrics["area_cells"] + metrics["area_inclusion_bodies"]
        assert abs(summed - metrics["area_total"]) < 1e-6, filename
