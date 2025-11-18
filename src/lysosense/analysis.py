"""Signal analysis helpers for LysoSense DCS datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from numpy import trapz
from scipy.optimize import curve_fit
from scipy.stats import lognorm, norm

from .io import Measurement


ModelType = Literal["gaussian", "lognormal"]


@dataclass
class AnalysisOptions:
    model: ModelType = "gaussian"
    mu_ib_um: float = 0.48
    mu_cell_um: float = 0.85
    allow_shift_fraction: float = 0.20
    sigma_bounds_gauss: Tuple[float, float] = (1e-3, 0.50)
    s_bounds_logn: Tuple[float, float] = (0.05, 0.80)
    second_peak_min_frac: float = 0.02
    dense_points: int = 1200


@dataclass
class AnalysisResult:
    measurement: Measurement
    observed: pd.DataFrame
    dense_fit: pd.DataFrame
    metrics: Dict[str, float | str | None]
    fit_kind: Literal["one", "two"]
    options: AnalysisOptions


class _SecondPeakTooSmall(Exception):
    """Raised when the fitted second peak contributes negligibly."""


def analyze_measurement(
    measurement: Measurement, options: AnalysisOptions | None = None
) -> AnalysisResult:
    opts = options or AnalysisOptions()
    df = measurement.data.copy()
    if df.empty:
        raise ValueError("Cannot analyse an empty measurement")

    x = df["particle_size_um"].to_numpy(dtype=float)
    y = df["mass_signal_ug"].to_numpy(dtype=float)

    fitres = _fit_curve(x, y, opts)
    observed = _augment_observed(df, x, fitres, opts)
    dense_fit = _build_dense_frame(x, fitres, opts)
    metrics = _derive_metrics(x, fitres, opts)

    return AnalysisResult(
        measurement=measurement,
        observed=observed,
        dense_fit=dense_fit,
        metrics=metrics,
        fit_kind=fitres["kind"],
        options=opts,
    )


def _augment_observed(df: pd.DataFrame, x: np.ndarray, fitres: Dict[str, np.ndarray], opts: AnalysisOptions) -> pd.DataFrame:
    observed = df.copy()
    total, cells, ibs, _ = _component_arrays(x, fitres, opts)
    observed["fit_signal_ug"] = total
    observed["cells_component_ug"] = cells
    observed["ibs_component_ug"] = ibs
    return observed


def _build_dense_frame(x: np.ndarray, fitres: Dict[str, np.ndarray], opts: AnalysisOptions) -> pd.DataFrame:
    dense_x = np.linspace(x.min(), x.max(), opts.dense_points)
    total, cells, ibs, baseline = _component_arrays(dense_x, fitres, opts)
    return pd.DataFrame(
        {
            "particle_size_um": dense_x,
            "fit_signal_ug": total,
            "cells_component_ug": cells,
            "ibs_component_ug": ibs,
            "baseline_ug": baseline,
        }
    )


def _derive_metrics(x: np.ndarray, fitres: Dict[str, np.ndarray], opts: AnalysisOptions) -> Dict[str, float | str | None]:
    total, cells, ibs, _ = _component_arrays(x, fitres, opts)
    area_cells = float(trapz(cells, x))
    area_ibs = float(trapz(ibs, x))
    area_total = max(area_cells + area_ibs, 1e-12)

    if fitres["kind"] == "two":
        _, m1, _, _, m2, _, _ = fitres["popt"]
        cell_first = abs(m1 - opts.mu_cell_um) < abs(m2 - opts.mu_cell_um)
        m_cell = float(m1 if cell_first else m2)
        m_ib = float(m2 if cell_first else m1)
    else:
        _, m_ib, _, _ = fitres["popt"]
        m_cell = None
        m_ib = float(m_ib)

    intact_fraction = float(area_cells / area_total)
    lysis_eff = float(1.0 - intact_fraction)

    return {
        "model": opts.model,
        "fit_kind": fitres["kind"],
        "area_cells": area_cells,
        "area_inclusion_bodies": area_ibs,
        "area_total": area_total,
        "intact_fraction": intact_fraction,
        "lysis_efficiency": lysis_eff,
        "mean_cell_µm": m_cell,
        "mean_ib_µm": m_ib,
    }


def _fit_curve(x: np.ndarray, y: np.ndarray, opts: AnalysisOptions) -> Dict[str, np.ndarray]:
    if np.max(y) <= 0:
        raise ValueError("Signal trace contains no positive values to fit")

    p0 = _initial_guesses(y, opts)
    bounds = _bounds_for_two_peak(opts)
    model_fn = _two_peak_model(opts)

    try:
        popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=100000)
        comp1, comp2, _ = _component_arrays_raw(x, popt, opts)
        area1 = trapz(comp1, x)
        area2 = trapz(comp2, x)
        frac2 = area2 / max(area1 + area2, 1e-12)
        if frac2 < opts.second_peak_min_frac:
            raise _SecondPeakTooSmall
        return {"kind": "two", "popt": popt, "pcov": pcov}
    except (RuntimeError, ValueError, _SecondPeakTooSmall):
        single_bounds = _bounds_for_one_peak(opts)
        single_model = _one_peak_model(opts)
        single_p0 = (p0[0], p0[1], p0[2], p0[-1])
        popt, pcov = curve_fit(
            single_model, x, y, p0=single_p0, bounds=single_bounds, maxfev=100000
        )
        return {"kind": "one", "popt": popt, "pcov": pcov}


def _initial_guesses(y: np.ndarray, opts: AnalysisOptions) -> Tuple[float, ...]:
    max_signal = float(np.max(y))
    A1 = max_signal * 0.8
    A2 = max_signal * 0.6
    if opts.model == "lognormal":
        s1 = np.clip(0.20, *opts.s_bounds_logn)
        s2 = np.clip(0.20, *opts.s_bounds_logn)
    else:
        s1 = np.clip(0.10, *opts.sigma_bounds_gauss)
        s2 = np.clip(0.15, *opts.sigma_bounds_gauss)
    c0 = float(max(0.0, np.percentile(y, 1)))
    return (A1, opts.mu_ib_um, s1, A2, opts.mu_cell_um, s2, c0)


def _bounds_for_two_peak(opts: AnalysisOptions) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    lo_mu_cell = opts.mu_cell_um * (1 - opts.allow_shift_fraction)
    hi_mu_cell = opts.mu_cell_um * (1 + opts.allow_shift_fraction)

    if opts.model == "lognormal":
        lo = [0, lo_mu_ib, opts.s_bounds_logn[0], 0, lo_mu_cell, opts.s_bounds_logn[0], 0]
        hi = [np.inf, hi_mu_ib, opts.s_bounds_logn[1], np.inf, hi_mu_cell, opts.s_bounds_logn[1], np.inf]
    else:
        lo = [0, lo_mu_ib, opts.sigma_bounds_gauss[0], 0, lo_mu_cell, opts.sigma_bounds_gauss[0], 0]
        hi = [np.inf, hi_mu_ib, opts.sigma_bounds_gauss[1], np.inf, hi_mu_cell, opts.sigma_bounds_gauss[1], np.inf]
    return (lo, hi)


def _bounds_for_one_peak(opts: AnalysisOptions) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    if opts.model == "lognormal":
        lo = [0, lo_mu_ib, opts.s_bounds_logn[0], 0]
        hi = [np.inf, hi_mu_ib, opts.s_bounds_logn[1], np.inf]
    else:
        lo = [0, lo_mu_ib, opts.sigma_bounds_gauss[0], 0]
        hi = [np.inf, hi_mu_ib, opts.sigma_bounds_gauss[1], np.inf]
    return (lo, hi)


def _two_peak_model(opts: AnalysisOptions):
    if opts.model == "lognormal":
        def model(x, A1, m1, s1, A2, m2, s2, c0):
            return (
                A1 * lognorm.pdf(x, s1, scale=m1)
                + A2 * lognorm.pdf(x, s2, scale=m2)
                + c0
            )
    else:
        def model(x, A1, m1, s1, A2, m2, s2, c0):
            return (
                A1 * norm.pdf(x, loc=m1, scale=s1)
                + A2 * norm.pdf(x, loc=m2, scale=s2)
                + c0
            )
    return model


def _one_peak_model(opts: AnalysisOptions):
    if opts.model == "lognormal":
        def model(x, A1, m1, s1, c0):
            return A1 * lognorm.pdf(x, s1, scale=m1) + c0
    else:
        def model(x, A1, m1, s1, c0):
            return A1 * norm.pdf(x, loc=m1, scale=s1) + c0
    return model


def _component_arrays(
    x: np.ndarray, fitres: Dict[str, np.ndarray], opts: AnalysisOptions
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fitres["kind"] == "two":
        comp1, comp2, baseline = _component_arrays_raw(x, fitres["popt"], opts)
        m1 = fitres["popt"][1]
        m2 = fitres["popt"][4]
        cell_first = abs(m1 - opts.mu_cell_um) < abs(m2 - opts.mu_cell_um)
        if cell_first:
            cells, ibs = comp1, comp2
        else:
            cells, ibs = comp2, comp1
    else:
        comp1, baseline = _single_component(x, fitres["popt"], opts)
        cells = np.zeros_like(comp1)
        ibs = comp1
    total = cells + ibs + baseline
    return total, cells, ibs, baseline


def _component_arrays_raw(
    x: np.ndarray, params: np.ndarray, opts: AnalysisOptions
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if opts.model == "lognormal":
        comp1 = params[0] * lognorm.pdf(x, params[2], scale=params[1])
        comp2 = params[3] * lognorm.pdf(x, params[5], scale=params[4])
    else:
        comp1 = params[0] * norm.pdf(x, loc=params[1], scale=params[2])
        comp2 = params[3] * norm.pdf(x, loc=params[4], scale=params[5])
    baseline = np.full_like(comp1, params[6])
    return comp1, comp2, baseline


def _single_component(
    x: np.ndarray, params: np.ndarray, opts: AnalysisOptions
) -> Tuple[np.ndarray, np.ndarray]:
    if opts.model == "lognormal":
        comp = params[0] * lognorm.pdf(x, params[2], scale=params[1])
    else:
        comp = params[0] * norm.pdf(x, loc=params[1], scale=params[2])
    baseline = np.full_like(comp, params[3])
    return comp, baseline
