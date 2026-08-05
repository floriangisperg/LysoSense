"""Guards for the fit-pipeline performance optimizations.

These lock the "identical results, faster" contract:

* **L1** — the fit/finalize split is an identity: ``_finalize(_analyze_fit_only(m, o))``
  reproduces ``analyze_measurement(m, o)`` exactly (same fit, dense frame, metrics).
* **L2** — sharing model-independent hints does not change a fit: ``_fit_curve`` with a
  ``precomputed`` memo yields the same parameters as without it.
* **autofit** — the optimized 16-combo autofit (shared hints + memoized single-peak fits
  + finalize-on-the-winner + deferred overlap-robustness sweep) selects the same model
  and reports the same metrics as the original sequential grid.

If any of these break, a performance optimization has silently changed the numbers.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# `src` (lysosense) is added by conftest; also make `app` importable for the
# autofit test (streamlit_app holds the autofit entry point).
_REPO = Path(__file__).resolve().parent.parent
_APP = _REPO / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from lysosense import (  # noqa: E402
    AnalysisOptions,
    Measurement,
    analyze_measurement,
    calculate_r_squared,
)
from lysosense.analysis import (  # noqa: E402
    _analyze_fit_only,
    _build_precomputed_hints,
    _finalize,
    _fit_curve,
)


def _gauss(x: np.ndarray, a: float, mu: float, s: float) -> np.ndarray:
    return a * np.exp(-0.5 * ((x - mu) / s) ** 2)


def _measurement(x: np.ndarray, y: np.ndarray, name: str = "synth") -> Measurement:
    df = pd.DataFrame({"particle_size_um": x, "mass_signal_ug": y})
    return Measurement(name=name, metadata={}, data=df, source=name)


def _two_peak() -> Measurement:
    x = np.linspace(0.3, 1.6, 400)
    y = _gauss(x, 2.0, 0.48, 0.06) + _gauss(x, 1.0, 0.85, 0.06)
    return _measurement(x, y, "two")


def _single_ib() -> Measurement:
    x = np.linspace(0.3, 1.6, 400)
    y = _gauss(x, 2.0, 0.48, 0.06)
    return _measurement(x, y, "ib")


_OPTION_SETS = [
    AnalysisOptions(),
    AnalysisOptions(use_overlap_deconvolution=True),
]


@pytest.mark.parametrize("opts", _OPTION_SETS, ids=["default", "overlap"])
def test_l1_split_is_identity(opts: AnalysisOptions) -> None:
    """_finalize(_analyze_fit_only(...)) == analyze_measurement(...) exactly."""
    m = _two_peak()
    direct = analyze_measurement(m, opts)
    rebuilt = _finalize(_analyze_fit_only(m, opts))

    assert direct.fit_kind == rebuilt.fit_kind
    assert direct.metrics["fit_kind"] == rebuilt.metrics["fit_kind"]
    # Identical fit signal on the raw grid and on the dense grid => identical fit.
    assert np.allclose(
        direct.observed["fit_signal_ug"], rebuilt.observed["fit_signal_ug"]
    )
    assert np.allclose(
        direct.dense_fit["fit_signal_ug"], rebuilt.dense_fit["fit_signal_ug"]
    )
    # Spot-check the headline metric.
    assert direct.metrics["lysis_efficiency"] == pytest.approx(
        rebuilt.metrics["lysis_efficiency"]
    )


@pytest.mark.parametrize("opts", _OPTION_SETS, ids=["default", "overlap"])
def test_l2_precomputed_hints_do_not_change_fit(opts: AnalysisOptions) -> None:
    """A fit with the shared hints memo must match the fit without it."""
    m = _two_peak()
    x = m.data["particle_size_um"].to_numpy(dtype=float)
    y = m.data["mass_signal_ug"].to_numpy(dtype=float)

    plain = _fit_curve(x, y, opts)
    withMemo = _fit_curve(x, y, opts, precomputed=_build_precomputed_hints(x, y, opts))

    assert plain["kind"] == withMemo["kind"]
    assert np.allclose(plain["popt"], withMemo["popt"])


def _reference_autofit(measurement: Measurement, options: AnalysisOptions) -> object:
    """The original sequential 16-combo grid — the definition of correctness."""
    import streamlit_app  # noqa: E402

    models = ["gaussian", "lognormal", "splitgaussian", "gennormal"]
    best_r2 = -float("inf")
    best_rs = float("inf")
    best = None
    tol = 5e-4
    for mi in models:
        for mc in models:
            try:
                r = analyze_measurement(
                    measurement,
                    replace(options, model="gaussian", model_ib=mi, model_cell=mc),  # type: ignore[arg-type]
                )
            except Exception:
                continue
            if r.fit_kind in ("two", "overlap"):
                intact = streamlit_app.safe_float(
                    r.metrics.get("intact_fraction"), 0.0
                )
                if r.fit_kind != "overlap" and mi == "gennormal":
                    continue
                if r.fit_kind != "overlap" and mc == "gennormal" and intact < 0.15:
                    continue
            r2 = calculate_r_squared(r)
            rs = streamlit_app._fit_residual_score(r)
            if r2 > best_r2 + tol or (abs(r2 - best_r2) <= tol and rs < best_rs):
                best_r2, best_rs, best = r2, rs, r
    assert best is not None
    return best


def _signature(res: object) -> tuple:
    m = res.metrics  # type: ignore[attr-defined]
    return (
        res.fit_kind,  # type: ignore[attr-defined]
        round(float(m["lysis_efficiency"]), 10),
        None if m["mean_ib_µm"] is None else round(float(m["mean_ib_µm"]), 10),
        None if m["mean_cell_µm"] is None else round(float(m["mean_cell_µm"]), 10),
        m.get("area_robustness"),
    )


@pytest.mark.parametrize(
    "builder", [_two_peak, _single_ib], ids=["two_peak", "single_ib"]
)
@pytest.mark.parametrize("opts", _OPTION_SETS, ids=["default", "overlap"])
def test_autofit_matches_reference_grid(
    builder: object, opts: AnalysisOptions
) -> None:
    """Optimized autofit picks the same winner/metrics as the sequential grid."""
    import streamlit_app  # noqa: E402

    m = builder()  # type: ignore[operator]
    reference = _reference_autofit(m, opts)
    optimized = streamlit_app._autofit_measurement(m, opts)
    assert _signature(reference) == _signature(optimized)
