"""Signal analysis helpers for LysoSense DCS datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import lognorm, norm

from .io import Measurement


ModelType = Literal["gaussian", "lognormal"]
_MIN_RELIABLE_SIGMA = 0.015  # �m; narrower hints are considered noise


@dataclass
class AnalysisOptions:
    model: ModelType = "gaussian"
    mu_ib_um: float = 0.48
    mu_cell_um: float = 0.85
    allow_shift_fraction: float = 0.20
    sigma_bounds_gauss_ib: Tuple[float, float] = (0.01, 0.25)
    sigma_bounds_gauss_cells: Tuple[float, float] = (0.05, 0.50)
    s_bounds_logn_ib: Tuple[float, float] = (0.03, 0.40)
    s_bounds_logn_cells: Tuple[float, float] = (0.05, 0.80)
    second_peak_min_frac: float = 0.02
    dense_points: int = 1200
    max_peak_fwhm_um: Optional[float] = None


@dataclass
class AnalysisResult:
    measurement: Measurement
    observed: pd.DataFrame
    dense_fit: pd.DataFrame
    metrics: Dict[str, float | str | None]
    fit_kind: Literal["one", "two"]
    options: AnalysisOptions


@dataclass
class _PeakHint:
    amplitude: float
    mu: float
    sigma: float


PeakHints = Dict[str, Optional[_PeakHint]]


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
    area_cells = float(np.trapz(cells, x))
    area_ibs = float(np.trapz(ibs, x))
    area_total = max(area_cells + area_ibs, 1e-12)

    if fitres["kind"] == "two":
        _, m1, _, _, m2, _, _ = fitres["popt"]
        cell_first = fitres.get("cell_first")
        if cell_first is None:
            cell_first = _cell_component_first(m1, m2, opts)
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

    p0, hints = _initial_guesses(x, y, opts)
    bounds = _bounds_for_two_peak(opts, hints)
    model_fn = _two_peak_model(opts)

    try:
        popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=100000)
        comp1, comp2, _ = _component_arrays_raw(x, popt, opts)
        area1 = np.trapz(comp1, x)
        area2 = np.trapz(comp2, x)
        frac2 = area2 / max(area1 + area2, 1e-12)
        if frac2 < opts.second_peak_min_frac:
            raise _SecondPeakTooSmall
        cell_first = _determine_cell_first(popt, hints, opts)
        return {"kind": "two", "popt": popt, "pcov": pcov, "cell_first": cell_first, "hints": hints}
    except (RuntimeError, ValueError, _SecondPeakTooSmall):
        single_bounds = _bounds_for_one_peak(opts, hints)
        single_model = _one_peak_model(opts)
        single_p0 = (p0[0], p0[1], p0[2], p0[-1])
        popt, pcov = curve_fit(
            single_model, x, y, p0=single_p0, bounds=single_bounds, maxfev=100000
        )
        return {"kind": "one", "popt": popt, "pcov": pcov, "cell_first": None, "hints": hints}


def _initial_guesses(x: np.ndarray, y: np.ndarray, opts: AnalysisOptions) -> Tuple[Tuple[float, ...], PeakHints]:
    max_signal = float(np.max(y))
    baseline_guess = float(np.percentile(y, 1))
    span = max(opts.allow_shift_fraction, 0.20)

    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    lo_mu_cell = opts.mu_cell_um * (1 - opts.allow_shift_fraction)
    hi_mu_cell = opts.mu_cell_um * (1 + opts.allow_shift_fraction)

    peak_candidates = _discover_peak_candidates(x, y, baseline_guess)
    ib_hint = _pop_candidate_near_target(peak_candidates, opts.mu_ib_um, span)
    cell_hint = _pop_candidate_near_target(peak_candidates, opts.mu_cell_um, span)
    if ib_hint is None:
        ib_hint = _estimate_peak_hint(x, y, opts.mu_ib_um, span, baseline_guess)
    if cell_hint is None:
        cell_hint = _estimate_peak_hint(x, y, opts.mu_cell_um, span, baseline_guess)

    # Amplitudes default to global estimates but prefer hint-derived values
    ib_amp_default = max(max_signal - baseline_guess, 1e-6) * 0.9
    cell_tail = max_signal * 0.2
    if cell_hint:
        cell_tail = cell_hint.amplitude
    else:
        tail_mask = x >= opts.mu_cell_um * (1 - 0.2)
        if np.any(tail_mask):
            tail_values = y[tail_mask]
            cell_tail = float(np.max(tail_values) - baseline_guess)
    cell_amp_default = max(cell_tail, (max_signal - baseline_guess) * 0.1)

    A1 = ib_hint.amplitude if ib_hint else ib_amp_default
    A2 = cell_hint.amplitude if cell_hint else cell_amp_default
    mu_ib_guess = (
        float(np.clip(ib_hint.mu, lo_mu_ib, hi_mu_ib)) if ib_hint else opts.mu_ib_um
    )
    mu_cell_guess = (
        float(np.clip(cell_hint.mu, lo_mu_cell, hi_mu_cell)) if cell_hint else opts.mu_cell_um
    )

    if opts.model == "lognormal":
        s1 = np.clip(
            _lognormal_shape_from_hint(ib_hint, mu_ib_guess), *opts.s_bounds_logn_ib
        )
        s2 = np.clip(
            _lognormal_shape_from_hint(cell_hint, mu_cell_guess), *opts.s_bounds_logn_cells
        )
    else:
        s1_hint = ib_hint.sigma if ib_hint else None
        s2_hint = cell_hint.sigma if cell_hint else None
        s1 = np.clip(
            s1_hint if s1_hint is not None else 0.08,
            *opts.sigma_bounds_gauss_ib,
        )
        s2 = np.clip(
            s2_hint if s2_hint is not None else 0.15,
            *opts.sigma_bounds_gauss_cells,
        )

    c0 = float(max(0.0, baseline_guess))
    params = (A1, mu_ib_guess, s1, A2, mu_cell_guess, s2, c0)
    return params, {"ib": ib_hint, "cell": cell_hint}


def _bounds_for_two_peak(
    opts: AnalysisOptions, hints: PeakHints
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    lo_mu_cell = opts.mu_cell_um * (1 - opts.allow_shift_fraction)
    hi_mu_cell = opts.mu_cell_um * (1 + opts.allow_shift_fraction)

    ib_hint = hints.get("ib")
    cell_hint = hints.get("cell")
    if ib_hint:
        lo_mu_ib, hi_mu_ib = _tight_bounds_from_hint(
            ib_hint.mu, lo_mu_ib, hi_mu_ib
        )
        sigma_ib_lo, sigma_ib_hi = _sigma_bounds_from_hint(
            ib_hint.sigma, opts.sigma_bounds_gauss_ib
        )
    else:
        sigma_ib_lo, sigma_ib_hi = opts.sigma_bounds_gauss_ib
    if cell_hint:
        lo_mu_cell, hi_mu_cell = _tight_bounds_from_hint(
            cell_hint.mu, lo_mu_cell, hi_mu_cell
        )
        sigma_cell_lo, sigma_cell_hi = _sigma_bounds_from_hint(
            cell_hint.sigma, opts.sigma_bounds_gauss_cells
        )
    else:
        sigma_cell_lo, sigma_cell_hi = opts.sigma_bounds_gauss_cells

    ib_mode_ref = float(np.clip(ib_hint.mu if ib_hint else opts.mu_ib_um, lo_mu_ib, hi_mu_ib))
    cell_mode_ref = float(np.clip(cell_hint.mu if cell_hint else opts.mu_cell_um, lo_mu_cell, hi_mu_cell))

    max_sigma_cap = _sigma_cap_from_fwhm(opts.max_peak_fwhm_um)
    if max_sigma_cap is not None:
        sigma_ib_hi = min(sigma_ib_hi, max_sigma_cap)
        sigma_cell_hi = min(sigma_cell_hi, max_sigma_cap)
        if sigma_ib_hi <= sigma_ib_lo:
            sigma_ib_hi = max(sigma_ib_lo * 1.01, sigma_ib_lo + 1e-4)
        if sigma_cell_hi <= sigma_cell_lo:
            sigma_cell_hi = max(sigma_cell_lo * 1.01, sigma_cell_lo + 1e-4)

    if opts.model == "lognormal":
        lo = [
            0,
            lo_mu_ib,
            opts.s_bounds_logn_ib[0],
            0,
            lo_mu_cell,
            opts.s_bounds_logn_cells[0],
            -np.inf,
        ]
        hi = [
            np.inf,
            hi_mu_ib,
            opts.s_bounds_logn_ib[1],
            np.inf,
            hi_mu_cell,
            opts.s_bounds_logn_cells[1],
            np.inf,
        ]
        max_shape_ib = _shape_cap_from_fwhm(opts.max_peak_fwhm_um, ib_mode_ref)
        max_shape_cell = _shape_cap_from_fwhm(opts.max_peak_fwhm_um, cell_mode_ref)
        if max_shape_ib is not None:
            hi[2] = min(hi[2], max_shape_ib)
            if hi[2] <= lo[2]:
                hi[2] = max(lo[2] * 1.01, lo[2] + 1e-4)
        if max_shape_cell is not None:
            hi[5] = min(hi[5], max_shape_cell)
            if hi[5] <= lo[5]:
                hi[5] = max(lo[5] * 1.01, lo[5] + 1e-4)
    else:
        lo = [
            0,
            lo_mu_ib,
            sigma_ib_lo,
            0,
            lo_mu_cell,
            sigma_cell_lo,
            -np.inf,
        ]
        hi = [
            np.inf,
            hi_mu_ib,
            sigma_ib_hi,
            np.inf,
            hi_mu_cell,
            sigma_cell_hi,
            np.inf,
        ]
    return (lo, hi)


def _bounds_for_one_peak(
    opts: AnalysisOptions, hints: PeakHints
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    ib_hint = hints.get("ib")
    if ib_hint:
        lo_mu_ib, hi_mu_ib = _tight_bounds_from_hint(
            ib_hint.mu, lo_mu_ib, hi_mu_ib
        )
        sigma_lo, sigma_hi = _sigma_bounds_from_hint(
            ib_hint.sigma, opts.sigma_bounds_gauss_ib
        )
    else:
        sigma_lo, sigma_hi = opts.sigma_bounds_gauss_ib
    ib_mode_ref = float(np.clip(ib_hint.mu if ib_hint else opts.mu_ib_um, lo_mu_ib, hi_mu_ib))

    if opts.model == "lognormal":
        lo = [0, lo_mu_ib, opts.s_bounds_logn_ib[0], -np.inf]
        hi = [np.inf, hi_mu_ib, opts.s_bounds_logn_ib[1], np.inf]
        max_shape_ib = _shape_cap_from_fwhm(opts.max_peak_fwhm_um, ib_mode_ref)
        if max_shape_ib is not None:
            hi[2] = min(hi[2], max_shape_ib)
            if hi[2] <= lo[2]:
                hi[2] = max(lo[2] * 1.01, lo[2] + 1e-4)
    else:
        max_sigma_cap = _sigma_cap_from_fwhm(opts.max_peak_fwhm_um)
        if max_sigma_cap is not None:
            sigma_hi = min(sigma_hi, max_sigma_cap)
            if sigma_hi <= sigma_lo:
                sigma_hi = max(sigma_lo * 1.01, sigma_lo + 1e-4)
        lo = [0, lo_mu_ib, sigma_lo, -np.inf]
        hi = [np.inf, hi_mu_ib, sigma_hi, np.inf]
    return (lo, hi)


def _two_peak_model(opts: AnalysisOptions):
    if opts.model == "lognormal":
        def model(x, A1, m1, s1, A2, m2, s2, c0):
            scale1 = _lognorm_scale_from_mode(m1, s1)
            scale2 = _lognorm_scale_from_mode(m2, s2)
            return (
                A1 * lognorm.pdf(x, s1, scale=scale1)
                + A2 * lognorm.pdf(x, s2, scale=scale2)
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
            scale = _lognorm_scale_from_mode(m1, s1)
            return A1 * lognorm.pdf(x, s1, scale=scale) + c0
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
        cell_first = fitres.get("cell_first")
        if cell_first is None:
            cell_first = _cell_component_first(m1, m2, opts)
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
        scale1 = _lognorm_scale_from_mode(params[1], params[2])
        scale2 = _lognorm_scale_from_mode(params[4], params[5])
        comp1 = params[0] * lognorm.pdf(x, params[2], scale=scale1)
        comp2 = params[3] * lognorm.pdf(x, params[5], scale=scale2)
    else:
        comp1 = params[0] * norm.pdf(x, loc=params[1], scale=params[2])
        comp2 = params[3] * norm.pdf(x, loc=params[4], scale=params[5])
    baseline = np.full_like(comp1, params[6])
    return comp1, comp2, baseline


def _single_component(
    x: np.ndarray, params: np.ndarray, opts: AnalysisOptions
) -> Tuple[np.ndarray, np.ndarray]:
    if opts.model == "lognormal":
        scale = _lognorm_scale_from_mode(params[1], params[2])
        comp = params[0] * lognorm.pdf(x, params[2], scale=scale)
    else:
        comp = params[0] * norm.pdf(x, loc=params[1], scale=params[2])
    baseline = np.full_like(comp, params[3])
    return comp, baseline


def _tight_bounds_from_hint(
    center: float, default_lo: float, default_hi: float
) -> Tuple[float, float]:
    epsilon = max(abs(center) * 1e-3, 1e-4)
    lo = max(center - epsilon, default_lo)
    hi = min(center + epsilon, default_hi)
    if hi <= lo:
        hi = lo + epsilon
    return lo, hi


def _sigma_bounds_from_hint(
    sigma: float, defaults: Tuple[float, float], tolerance: float = 0.2
) -> Tuple[float, float]:
    lo_default, hi_default = defaults
    if sigma <= 0 or sigma < _MIN_RELIABLE_SIGMA:
        return lo_default, hi_default
    if sigma < 0.05:
        tolerance = 0.05
    lo = max(sigma * (1 - tolerance), lo_default)
    hi = min(sigma * (1 + tolerance), hi_default)
    if hi <= lo:
        hi = lo * (1 + tolerance)
    return lo, hi


def _estimate_peak_hint(
    x: np.ndarray,
    y: np.ndarray,
    center: float,
    span: float,
    baseline: float,
) -> Optional[_PeakHint]:
    lo = center * (1 - span)
    hi = center * (1 + span)
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 4:
        return None
    region_x = x[mask]
    region_y = y[mask]
    idx = _select_peak_index(region_x, region_y)
    if idx is None:
        return None
    return _build_hint_from_region(region_x, region_y, idx, baseline)


def _discover_peak_candidates(
    x: np.ndarray, y: np.ndarray, baseline: float, max_candidates: int = 6
) -> List[_PeakHint]:
    if x.size == 0 or y.size == 0:
        return []
    adjusted = y - baseline
    adjusted = np.where(adjusted > 0, adjusted, 0.0)
    if not np.any(adjusted > 0):
        return []
    prominence = max(float(np.max(adjusted) * 0.05), 1e-9)
    distance = max(1, len(x) // 150)
    peak_indices, _ = find_peaks(adjusted, prominence=prominence, distance=distance)
    hints: List[_PeakHint] = []
    for idx in peak_indices:
        region_x, region_y, local_idx = _region_around_index(x, y, int(idx))
        if len(region_x) < 2:
            continue
        hint = _build_hint_from_region(region_x, region_y, local_idx, baseline)
        hints.append(hint)
    hints.sort(key=lambda h: h.amplitude, reverse=True)
    if len(hints) > max_candidates:
        hints = hints[:max_candidates]
    return hints


def _pop_candidate_near_target(
    candidates: List[_PeakHint], target: float, span: float
) -> Optional[_PeakHint]:
    if not candidates:
        return None

    window = max(abs(target) * span, 0.03)
    lo = min(target * (1 - span), target - window)
    hi = max(target * (1 + span), target + window)

    chosen_idx = None
    for idx, hint in enumerate(candidates):
        if lo <= hint.mu <= hi:
            chosen_idx = idx
            break

    if chosen_idx is None:
        tolerance = max(abs(target) * (span + 0.35), 0.08)
        best_idx = None
        best_distance = float("inf")
        for idx, hint in enumerate(candidates):
            dist = abs(hint.mu - target)
            if dist < best_distance:
                best_distance = dist
                best_idx = idx
        if best_idx is None or best_distance > tolerance:
            return None
        chosen_idx = best_idx

    return candidates.pop(chosen_idx)


def _region_around_index(
    x: np.ndarray, y: np.ndarray, idx: int, window_fraction: float = 0.08
) -> Tuple[np.ndarray, np.ndarray, int]:
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([]), 0
    half_window = max(2, int(len(x) * window_fraction))
    start = max(0, idx - half_window)
    stop = min(len(x), idx + half_window + 1)
    region_x = x[start:stop]
    region_y = y[start:stop]
    local_idx = idx - start
    return region_x, region_y, local_idx


def _build_hint_from_region(
    region_x: np.ndarray, region_y: np.ndarray, idx: int, baseline: float
) -> _PeakHint:
    amplitude = float(max(region_y[idx] - baseline, 1e-9))
    mu = float(region_x[idx])
    sigma = _estimate_sigma_from_region(region_x, region_y, idx)
    return _PeakHint(amplitude=amplitude, mu=mu, sigma=sigma)


def _estimate_sigma_from_region(
    region_x: np.ndarray, region_y: np.ndarray, idx: int
) -> float:
    if len(region_x) <= 1:
        return 1e-3
    target = region_y[idx]
    half_level = target * 0.5
    window_limit = max(1, int(0.15 * len(region_x)))
    left_idx = idx
    while (
        left_idx > 0
        and region_y[left_idx] > half_level
        and (idx - left_idx) < window_limit
    ):
        left_idx -= 1
    right_idx = idx
    while (
        right_idx < len(region_y) - 1
        and region_y[right_idx] > half_level
        and (right_idx - idx) < window_limit
    ):
        right_idx += 1
    if right_idx == left_idx:
        if len(region_x) > 1:
            right = min(idx + 1, len(region_x) - 1)
            left = max(idx - 1, 0)
            width = float(abs(region_x[right] - region_x[left]))
        else:
            width = 1e-3
    else:
        width = float(abs(region_x[right_idx] - region_x[left_idx]))
    return max(width / 2.354820045, 1e-3)


def _select_peak_index(region_x: np.ndarray, region_y: np.ndarray) -> Optional[int]:
    if len(region_y) == 0:
        return None
    if len(region_y) == 1:
        return 0
    dy = np.gradient(region_y, region_x)
    zero_crossings = np.where((dy[:-1] > 0) & (dy[1:] <= 0))[0] + 1
    if zero_crossings.size == 0:
        return int(np.argmax(region_y))
    local_idx = zero_crossings
    target_idx = int(local_idx[np.argmax(region_y[local_idx])])
    return target_idx


def _lognormal_shape_from_hint(
    hint: Optional[_PeakHint], mu_guess: float
) -> float:
    if not hint or hint.sigma <= 0 or mu_guess <= 0:
        return 0.20
    ratio = hint.sigma / max(mu_guess, 1e-6)
    return max(0.03, min(0.80, ratio))


def _sigma_cap_from_fwhm(limit_um: Optional[float]) -> Optional[float]:
    if limit_um is None or limit_um <= 0:
        return None
    return float(limit_um / 2.354820045)


def _shape_cap_from_fwhm(limit_um: Optional[float], mode_um: float) -> Optional[float]:
    if limit_um is None or limit_um <= 0 or mode_um <= 0:
        return None
    ratio = limit_um / (2.0 * mode_um)
    if ratio <= 0:
        return None
    return float(np.arcsinh(ratio) / np.sqrt(2.0 * np.log(2.0)))


def _lognorm_scale_from_mode(mode: float, shape: float) -> float:
    """
    Convert a desired lognormal mode to SciPy's `scale` parameter.

    SciPy's `lognorm.pdf` expects `scale = exp(mu)` where `mu` is the mean of
    the underlying normal distribution. The mode (peak) occurs at
    `scale * exp(-shape**2)`, so invert that relationship here.
    """
    safe_mode = max(mode, 1e-6)
    safe_shape = max(shape, 1e-6)
    return float(safe_mode * np.exp(safe_shape ** 2))


def _cell_component_first(m1: float, m2: float, opts: AnalysisOptions) -> bool:
    """
    Decide whether the first fitted component corresponds to intact cells.

    When the expected cell size exceeds the IB size (default case), prefer the
    component with the larger mean. Otherwise fall back to the closest-to-target
    heuristic so custom configurations keep working.
    """
    if opts.mu_cell_um > opts.mu_ib_um:
        return m1 >= m2
    return abs(m1 - opts.mu_cell_um) <= abs(m2 - opts.mu_cell_um)


def _determine_cell_first(
    popt: np.ndarray, hints: PeakHints, opts: AnalysisOptions
) -> bool:
    m1 = popt[1]
    m2 = popt[4]
    cell_hint = hints.get("cell") if hints else None
    ib_hint = hints.get("ib") if hints else None

    if cell_hint:
        return abs(m1 - cell_hint.mu) <= abs(m2 - cell_hint.mu)
    if ib_hint:
        return abs(m1 - ib_hint.mu) >= abs(m2 - ib_hint.mu)
    return _cell_component_first(m1, m2, opts)
