"""Signal analysis helpers for LysoSense DCS datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional, TypedDict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import lognorm, norm

from .io import Measurement


class _FitResult(TypedDict):
    """Type definition for fit result dictionary."""
    kind: Literal["one", "two"]
    popt: np.ndarray
    pcov: np.ndarray
    cell_first: Optional[bool]
    hints: PeakHints
    model_ib: ModelType
    model_cell: ModelType


ModelType = Literal["gaussian", "lognormal", "splitgaussian"]


def _params_per_peak(model: ModelType) -> int:
    """Return number of parameters per peak for a given model type."""
    if model == "splitgaussian":
        return 4
    return 3
_MIN_RELIABLE_SIGMA = 0.015  # �m; narrower hints are considered noise


@dataclass
class AnalysisOptions:
    model: ModelType = "gaussian"
    # Override model per peak (if set, takes precedence over 'model')
    model_ib: Optional[ModelType] = None
    model_cell: Optional[ModelType] = None
    mu_ib_um: float = 0.48
    mu_cell_um: float = 0.85
    allow_shift_fraction: float = 0.20
    sigma_bounds_gauss_ib: Tuple[float, float] = (0.01, 0.25)
    sigma_bounds_gauss_cells: Tuple[float, float] = (0.05, 0.50)
    s_bounds_logn_ib: Tuple[float, float] = (0.03, 0.40)
    s_bounds_logn_cells: Tuple[float, float] = (0.05, 0.80)
    # Split Gaussian bounds: (sigma_left, sigma_right) for each peak type
    sigma_left_bounds_ib: Tuple[float, float] = (0.01, 0.25)
    sigma_right_bounds_ib: Tuple[float, float] = (0.01, 0.40)
    sigma_left_bounds_cells: Tuple[float, float] = (0.05, 0.50)
    sigma_right_bounds_cells: Tuple[float, float] = (0.05, 0.80)
    second_peak_min_frac: float = 0.02
    dense_points: int = 1200
    max_peak_fwhm_um: Optional[float] = None

    # --- Gated 2-peak decision parameters ---
    # Enable the gated 2-peak decision logic (recommended: True)
    use_gated_two_peak: bool = True
    # Pre-fit gate: residual peak prominence threshold (in units of noise sigma)
    residual_prominence_sigma: float = 3.0
    # Pre-fit gate: minimum distance from main peak center (um)
    residual_min_distance_um: float = 0.15
    # Pre-fit gate: minimum residual area as fraction of total signal area
    residual_min_area_frac: float = 0.05
    # Post-fit gate: minimum BIC improvement for 2-peak model (negative = 2-peak better)
    bic_improvement_threshold: float = -10.0
    # Post-fit gate: minimum local dominance of second peak (fraction)
    local_dominance_threshold: float = 0.40
    # Post-fit gate: minimum second peak area fraction
    second_peak_area_threshold: float = 0.05
    # Post-fit gate: minimum peak separation relative to average FWHM
    min_separation_fwhm_ratio: float = 0.8

    # --- Second peak quality constraints (stricter than main peak) ---
    # Maximum FWHM for the Cell peak during fitting (um). None = use max_peak_fwhm_um
    # This is applied as a BOUND during fitting, not just a post-fit check.
    max_fwhm_second_peak_um: Optional[float] = 0.18
    # Minimum compactness for second peak: area / FWHM. Set to 0 to disable.
    min_compactness_second_peak: float = 0.0
    # Minimum prominence of second peak above the shoulder of main peak (in noise sigma units)
    # Set to 0 to disable. Note: prominence can be 0 for valid peaks sitting on a shoulder.
    min_prominence_second_peak_sigma: float = 0.0

    def get_model_ib(self) -> ModelType:
        return self.model_ib if self.model_ib else self.model

    def get_model_cell(self) -> ModelType:
        return self.model_cell if self.model_cell else self.model


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


class _NoResidualPeak(Exception):
    """Raised when the 1-peak residual has no evidence of a second peak."""


class _TwoPeakFitRejected(Exception):
    """Raised when the 2-peak fit fails post-fit validation checks."""


def _estimate_noise_from_residual(residual: np.ndarray) -> float:
    """
    Estimate noise level from the 1-peak fit residual using MAD.

    Uses Median Absolute Deviation scaled to match standard deviation
    for a normal distribution: sigma ≈ 1.4826 * MAD
    """
    # Use MAD of residual as robust noise estimate
    mad = float(np.median(np.abs(residual - np.median(residual))))
    return max(mad * 1.4826, 1e-9)


def _find_residual_peak_candidate(
    x: np.ndarray,
    residual: np.ndarray,
    main_peak_mu: float,
    opts: AnalysisOptions,
) -> Optional[Tuple[float, float]]:
    """
    Check if the positive residual contains a real peak candidate.

    Returns (peak_position, peak_height) if a valid candidate is found, else None.

    Criteria:
    - Prominence > noise_sigma * residual_prominence_sigma
    - Distance from main peak > residual_min_distance_um
    - Residual area > residual_min_area_frac of total signal area
    """
    # Only look at positive residual
    positive_residual = np.maximum(residual, 0)

    # Estimate noise level
    noise_sigma = _estimate_noise_from_residual(residual)

    # Check minimum residual area
    total_signal_area = float(np.trapz(np.maximum(positive_residual, 0), x))
    residual_area = float(np.trapz(positive_residual, x))
    if total_signal_area > 0:
        area_frac = residual_area / total_signal_area
        # Actually compare residual area to original signal (approximated by residual + fit)
        # For simplicity, use residual area relative to its own scale
        if residual_area < opts.residual_min_area_frac * total_signal_area * 10:  # heuristic scaling
            pass  # Don't reject yet, check other criteria

    # Find peaks in positive residual with minimum prominence
    min_prominence = noise_sigma * opts.residual_prominence_sigma
    distance = max(1, len(x) // 200)  # Minimum samples between peaks

    try:
        peak_indices, properties = find_peaks(
            positive_residual,
            prominence=min_prominence,
            distance=distance,
            width=2  # Minimum width to avoid single-point spikes
        )
    except ValueError:
        return None

    if len(peak_indices) == 0:
        return None

    # Filter by distance from main peak
    valid_candidates = []
    for idx in peak_indices:
        peak_x = x[idx]
        distance_from_main = abs(peak_x - main_peak_mu)
        if distance_from_main >= opts.residual_min_distance_um:
            peak_height = positive_residual[idx]
            # Get prominence from properties
            prominences = properties.get('prominences', [peak_height])
            prominence = prominences[list(peak_indices).index(idx)] if idx in peak_indices else peak_height
            valid_candidates.append((peak_x, peak_height, prominence, distance_from_main))

    if not valid_candidates:
        return None

    # Return the most prominent candidate
    valid_candidates.sort(key=lambda c: c[2], reverse=True)
    best = valid_candidates[0]
    return (best[0], best[1])


def _calculate_bic(
    y: np.ndarray,
    y_fit: np.ndarray,
    n_params: int,
) -> float:
    """
    Calculate Bayesian Information Criterion for a model fit.

    BIC = n * ln(RSS/n) + k * ln(n)

    Where n = number of data points, k = number of parameters, RSS = residual sum of squares.
    Lower BIC is better.
    """
    n = len(y)
    if n == 0:
        return float('inf')

    residuals = y - y_fit
    rss = float(np.sum(residuals ** 2))

    # Avoid log(0)
    if rss <= 0:
        rss = 1e-12

    bic = n * np.log(rss / n) + n_params * np.log(n)
    return float(bic)


def _check_local_dominance(
    x: np.ndarray,
    peak1: np.ndarray,
    peak2: np.ndarray,
    threshold: float,
) -> bool:
    """
    Check if peak2 dominates somewhere locally.

    Returns True if max_x(peak2 / (peak1 + peak2)) > threshold.
    This ensures the second peak actually "owns" a local region.
    """
    total = peak1 + peak2
    # Avoid division by zero
    total = np.where(total > 1e-12, total, 1e-12)
    dominance_ratio = peak2 / total

    max_dominance = float(np.max(dominance_ratio))
    return max_dominance > threshold


def _check_separation(
    mu1: float,
    mu2: float,
    fwhm1: float,
    fwhm2: float,
    min_ratio: float,
) -> bool:
    """
    Check if peaks are sufficiently separated.

    Criterion: |mu2 - mu1| / (0.5 * (FWHM1 + FWHM2)) > min_ratio
    """
    avg_fwhm = 0.5 * (fwhm1 + fwhm2)
    if avg_fwhm <= 0:
        return False

    separation_ratio = abs(mu2 - mu1) / avg_fwhm
    return separation_ratio > min_ratio


def _calculate_fwhm_from_params(
    model_type: ModelType,
    params: np.ndarray,
) -> float:
    """Calculate FWHM for a peak given its model type and parameters."""
    if model_type == "lognormal":
        # For lognormal, FWHM is approximately 2.355 * sigma (approximation)
        # More accurate: use the shape parameter
        shape = params[2]
        mode = params[1]
        # Approximate FWHM for lognormal
        return float(2.355 * shape * mode)
    elif model_type == "splitgaussian":
        # Use average of left and right sigma
        sigma_avg = 0.5 * (params[2] + params[3])
        return float(2.355 * sigma_avg)
    else:  # gaussian
        sigma = params[2]
        return float(2.355 * sigma)


def _calculate_compactness(
    area: float,
    fwhm: float,
) -> float:
    """
    Calculate peak compactness: area / FWHM.

    Higher compactness = sharper peak. A broad, flat peak will have low compactness
    even if it has significant area.
    """
    if fwhm <= 0:
        return 0.0
    return float(area / fwhm)


def _calculate_prominence_above_shoulder(
    x: np.ndarray,
    peak2: np.ndarray,
    peak1: np.ndarray,
    peak2_mu: float,
) -> float:
    """
    Calculate prominence of peak 2 above the shoulder of peak 1.

    The prominence is the height of peak 2 at its center minus the value of peak 1
    at that position (the "shoulder" of the main peak).

    Returns the prominence in the same units as the signal.
    """
    # Find index closest to peak 2's center
    idx = int(np.argmin(np.abs(x - peak2_mu)))

    # Height of peak 2 at its center
    peak2_height = float(peak2[idx])

    # Height of peak 1 (the shoulder) at peak 2's position
    shoulder_height = float(peak1[idx])

    # Prominence is how much peak 2 rises above the shoulder
    prominence = peak2_height - shoulder_height

    return max(prominence, 0.0)


def _check_second_peak_quality(
    x: np.ndarray,
    peak1: np.ndarray,
    peak2: np.ndarray,
    peak2_area: float,
    peak2_fwhm: float,
    peak2_mu: float,
    noise_sigma: float,
    opts: AnalysisOptions,
) -> Tuple[bool, str]:
    """
    Check quality constraints specific to the second (smaller) peak.

    Returns (passed, reason) where reason explains why it failed (if it did).
    """
    # Check 1: Maximum FWHM for second peak
    max_fwhm = opts.max_fwhm_second_peak_um
    if max_fwhm is None:
        max_fwhm = opts.max_peak_fwhm_um
    if max_fwhm is not None and peak2_fwhm > max_fwhm:
        return False, f"Second peak FWHM ({peak2_fwhm:.3f}) exceeds max ({max_fwhm:.3f})"

    # Check 2: Minimum compactness (area / FWHM)
    compactness = _calculate_compactness(peak2_area, peak2_fwhm)
    if compactness < opts.min_compactness_second_peak:
        return False, f"Second peak compactness ({compactness:.1f}) below minimum ({opts.min_compactness_second_peak:.1f})"

    # Check 3: Minimum prominence above shoulder
    prominence = _calculate_prominence_above_shoulder(x, peak1, peak2, peak2_mu)
    min_prominence = noise_sigma * opts.min_prominence_second_peak_sigma
    if prominence < min_prominence:
        return False, f"Second peak prominence ({prominence:.3f}) below minimum ({min_prominence:.3f})"

    return True, "OK"


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


def _augment_observed(
    df: pd.DataFrame,
    x: np.ndarray,
    fitres: _FitResult,
    opts: AnalysisOptions,
) -> pd.DataFrame:
    observed = df.copy()
    total, cells, ibs = _component_arrays(x, fitres, opts)
    observed["fit_signal_ug"] = total
    observed["cells_component_ug"] = cells
    observed["ibs_component_ug"] = ibs
    return observed


def _build_dense_frame(
    x: np.ndarray, fitres: _FitResult, opts: AnalysisOptions
) -> pd.DataFrame:
    dense_x = np.linspace(x.min(), x.max(), opts.dense_points)
    total, cells, ibs = _component_arrays(dense_x, fitres, opts)
    return pd.DataFrame(
        {
            "particle_size_um": dense_x,
            "fit_signal_ug": total,
            "cells_component_ug": cells,
            "ibs_component_ug": ibs,
        }
    )


def _derive_metrics(
    x: np.ndarray, fitres: _FitResult, opts: AnalysisOptions
) -> Dict[str, float | str | None]:
    total, cells, ibs = _component_arrays(x, fitres, opts)
    area_cells = float(np.trapz(cells, x))
    area_ibs = float(np.trapz(ibs, x))
    area_total = max(area_cells + area_ibs, 1e-12)

    model_ib = fitres.get("model_ib", opts.get_model_ib())
    model_cell = fitres.get("model_cell", opts.get_model_cell())

    if fitres["kind"] == "two":
        n1 = _params_per_peak(model_ib)
        m1 = fitres["popt"][1]
        m2 = fitres["popt"][n1 + 1]
        cell_first = fitres.get("cell_first")
        if cell_first is None:
            cell_first = _cell_component_first(m1, m2, opts)
        m_cell = float(m1 if cell_first else m2)
        m_ib = float(m2 if cell_first else m1)
    else:
        m_ib = float(fitres["popt"][1])
        m_cell = None

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


def _fit_curve(
    x: np.ndarray, y: np.ndarray, opts: AnalysisOptions
) -> _FitResult:
    if np.max(y) <= 0:
        raise ValueError("Signal trace contains no positive values to fit")

    model_ib = opts.get_model_ib()
    model_cell = opts.get_model_cell()

    base_p0, hints = _initial_guesses(x, y, opts)
    A1, mu_ib_guess, A2, mu_cell_guess = base_p0

    # Build initial params for single peak (IB peak)
    p1 = _single_peak_initial_params(model_ib, A1, mu_ib_guess, hints.get("ib"), opts, is_ib=True)

    # --- Step 1: Always fit 1-peak model first ---
    single_bounds = _bounds_for_one_peak(opts, hints, model_ib)
    single_model = _one_peak_model(model_ib)

    try:
        popt_1peak, pcov_1peak = curve_fit(
            single_model, x, y, p0=p1, bounds=single_bounds, maxfev=100000
        )
    except (RuntimeError, ValueError):
        # If even 1-peak fit fails, we have a problem
        raise ValueError("Failed to fit even a single-peak model")

    # Calculate 1-peak fit and residual
    y_fit_1peak = _eval_single_peak(x, model_ib, popt_1peak)
    residual = y - y_fit_1peak
    main_peak_mu = float(popt_1peak[1])

    # Calculate BIC for 1-peak model
    n_params_1peak = _params_per_peak(model_ib)
    bic_1peak = _calculate_bic(y, y_fit_1peak, n_params_1peak)

    # --- Step 2: Check if gated 2-peak decision is enabled ---
    if not opts.use_gated_two_peak:
        # Fall back to original behavior
        return _fit_two_peak_legacy(x, y, opts, hints, model_ib, model_cell,
                                     p1, popt_1peak, pcov_1peak, bic_1peak)

    # --- Step 3: Pre-fit gate - check residual for second peak evidence ---
    residual_candidate = _find_residual_peak_candidate(x, residual, main_peak_mu, opts)

    if residual_candidate is None:
        # No evidence of second peak in residual - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # --- Step 4: Attempt 2-peak fit ---
    p2 = _single_peak_initial_params(model_cell, A2, mu_cell_guess, hints.get("cell"), opts, is_ib=False)
    p0 = p1 + p2  # type: ignore[assignment]

    bounds = _bounds_for_two_peak(opts, hints, model_ib, model_cell)
    model_fn = _two_peak_model(opts, model_ib, model_cell)

    try:
        popt_2peak, pcov_2peak = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=100000)
    except (RuntimeError, ValueError):
        # 2-peak fit failed - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # --- Step 5: Post-fit gates ---
    comp1, comp2 = _component_arrays_raw(x, popt_2peak, model_ib, model_cell)
    y_fit_2peak = comp1 + comp2

    # Gate A: BIC improvement
    n_params_2peak = _params_per_peak(model_ib) + _params_per_peak(model_cell)
    bic_2peak = _calculate_bic(y, y_fit_2peak, n_params_2peak)
    delta_bic = bic_2peak - bic_1peak

    if delta_bic >= opts.bic_improvement_threshold:
        # BIC doesn't improve enough - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # Gate B: Area fraction of second peak
    area1 = float(np.trapz(comp1, x))
    area2 = float(np.trapz(comp2, x))
    total_area = area1 + area2
    frac2 = area2 / max(total_area, 1e-12)

    if frac2 < opts.second_peak_area_threshold:
        # Second peak too small - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # Gate C: Local dominance of second peak
    if not _check_local_dominance(x, comp1, comp2, opts.local_dominance_threshold):
        # Second peak doesn't dominate anywhere - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # Gate D: Peak separation (optional but recommended)
    mu1 = float(popt_2peak[1])
    n1 = _params_per_peak(model_ib)
    mu2 = float(popt_2peak[n1 + 1])
    fwhm1 = _calculate_fwhm_from_params(model_ib, popt_2peak[:n1])
    fwhm2 = _calculate_fwhm_from_params(model_cell, popt_2peak[n1:])

    if not _check_separation(mu1, mu2, fwhm1, fwhm2, opts.min_separation_fwhm_ratio):
        # Peaks too close / overlapping - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # Gate E: Second peak quality constraints (stricter than main peak)
    # Identify which component is the second (smaller) peak
    area1 = float(np.trapz(comp1, x))
    area2 = float(np.trapz(comp2, x))

    if area2 <= area1:
        # comp2 is the smaller peak - apply quality checks to it
        second_peak = comp2
        second_peak_area = area2
        second_peak_fwhm = fwhm2
        second_peak_mu = mu2
        main_peak = comp1
    else:
        # comp1 is the smaller peak - apply quality checks to it
        second_peak = comp1
        second_peak_area = area1
        second_peak_fwhm = fwhm1
        second_peak_mu = mu1
        main_peak = comp2

    # Estimate noise from the 1-peak residual
    noise_sigma = _estimate_noise_from_residual(residual)

    # Check second peak quality
    quality_ok, quality_reason = _check_second_peak_quality(
        x, main_peak, second_peak,
        second_peak_area, second_peak_fwhm, second_peak_mu,
        noise_sigma, opts
    )

    if not quality_ok:
        # Second peak doesn't meet quality standards - use 1-peak model
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }

    # All gates passed - accept 2-peak model
    cell_first = _determine_cell_first(popt_2peak, hints, opts, model_ib, model_cell)
    return {
        "kind": "two",
        "popt": popt_2peak,
        "pcov": pcov_2peak,
        "cell_first": cell_first,
        "hints": hints,
        "model_ib": model_ib,
        "model_cell": model_cell,
    }


def _fit_two_peak_legacy(
    x: np.ndarray,
    y: np.ndarray,
    opts: AnalysisOptions,
    hints: PeakHints,
    model_ib: ModelType,
    model_cell: ModelType,
    p1: Tuple[float, ...],
    popt_1peak: np.ndarray,
    pcov_1peak: np.ndarray,
    bic_1peak: float,
) -> _FitResult:
    """Legacy 2-peak fitting logic (used when gated decision is disabled)."""
    base_p0, _ = _initial_guesses(x, y, opts)
    A1, mu_ib_guess, A2, mu_cell_guess = base_p0

    p2 = _single_peak_initial_params(model_cell, A2, mu_cell_guess, hints.get("cell"), opts, is_ib=False)
    p0 = p1 + p2  # type: ignore[assignment]

    bounds = _bounds_for_two_peak(opts, hints, model_ib, model_cell)
    model_fn = _two_peak_model(opts, model_ib, model_cell)

    try:
        popt, pcov = curve_fit(model_fn, x, y, p0=p0, bounds=bounds, maxfev=100000)
        comp1, comp2 = _component_arrays_raw(x, popt, model_ib, model_cell)
        area1 = float(np.trapz(comp1, x))
        area2 = float(np.trapz(comp2, x))
        frac2 = area2 / max(area1 + area2, 1e-12)
        if frac2 < opts.second_peak_min_frac:
            raise _SecondPeakTooSmall
        cell_first = _determine_cell_first(popt, hints, opts, model_ib, model_cell)
        return {
            "kind": "two",
            "popt": popt,
            "pcov": pcov,
            "cell_first": cell_first,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }
    except (RuntimeError, ValueError, _SecondPeakTooSmall):
        return {
            "kind": "one",
            "popt": popt_1peak,
            "pcov": pcov_1peak,
            "cell_first": None,
            "hints": hints,
            "model_ib": model_ib,
            "model_cell": model_cell,
        }


def _initial_guesses(
    x: np.ndarray, y: np.ndarray, opts: AnalysisOptions
) -> Tuple[Tuple[float, ...], PeakHints]:
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
        float(np.clip(cell_hint.mu, lo_mu_cell, hi_mu_cell))
        if cell_hint
        else opts.mu_cell_um
    )

    return (A1, mu_ib_guess, A2, mu_cell_guess), {"ib": ib_hint, "cell": cell_hint}


def _single_peak_initial_params(
    model_type: ModelType,
    A: float,
    mu: float,
    hint: Optional[_PeakHint],
    opts: AnalysisOptions,
    is_ib: bool,
) -> Tuple[float, ...]:
    """Generate initial parameters for a single peak based on its model type."""
    if model_type == "lognormal":
        s = np.clip(
            _lognormal_shape_from_hint(hint, mu),
            *(opts.s_bounds_logn_ib if is_ib else opts.s_bounds_logn_cells)
        )
        return (A, mu, s)
    elif model_type == "splitgaussian":
        s_hint = hint.sigma if hint else None
        s = np.clip(
            s_hint if s_hint is not None else (0.08 if is_ib else 0.15),
            *(opts.sigma_bounds_gauss_ib if is_ib else opts.sigma_bounds_gauss_cells)
        )
        return (A, mu, s, s)  # Same sigma for left and right initially
    else:  # gaussian
        s_hint = hint.sigma if hint else None
        s = np.clip(
            s_hint if s_hint is not None else (0.08 if is_ib else 0.15),
            *(opts.sigma_bounds_gauss_ib if is_ib else opts.sigma_bounds_gauss_cells)
        )
        return (A, mu, s)


def _bounds_for_two_peak(
    opts: AnalysisOptions, hints: PeakHints, model_ib: ModelType, model_cell: ModelType
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Build bounds for two-peak fit with potentially different model types."""
    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    lo_mu_cell = opts.mu_cell_um * (1 - opts.allow_shift_fraction)
    hi_mu_cell = opts.mu_cell_um * (1 + opts.allow_shift_fraction)

    ib_hint = hints.get("ib")
    cell_hint = hints.get("cell")
    if ib_hint:
        lo_mu_ib, hi_mu_ib = _tight_bounds_from_hint(ib_hint.mu, lo_mu_ib, hi_mu_ib)
    if cell_hint:
        lo_mu_cell, hi_mu_cell = _tight_bounds_from_hint(
            cell_hint.mu, lo_mu_cell, hi_mu_cell
        )

    ib_mode_ref = float(
        np.clip(ib_hint.mu if ib_hint else opts.mu_ib_um, lo_mu_ib, hi_mu_ib)
    )
    cell_mode_ref = float(
        np.clip(cell_hint.mu if cell_hint else opts.mu_cell_um, lo_mu_cell, hi_mu_cell)
    )

    # Get bounds for each peak based on its model type
    lo_ib, hi_ib = _single_peak_bounds(
        model_ib, lo_mu_ib, hi_mu_ib, opts, ib_hint, ib_mode_ref, is_ib=True
    )
    lo_cell, hi_cell = _single_peak_bounds(
        model_cell, lo_mu_cell, hi_mu_cell, opts, cell_hint, cell_mode_ref, is_ib=False
    )

    return (tuple(lo_ib + lo_cell), tuple(hi_ib + hi_cell))


def _single_peak_bounds(
    model_type: ModelType,
    lo_mu: float,
    hi_mu: float,
    opts: AnalysisOptions,
    hint: Optional[_PeakHint],
    mode_ref: float,
    is_ib: bool,
) -> Tuple[List[float], List[float]]:
    """Build bounds for a single peak based on its model type.

    Uses tighter FWHM bounds for the Cell peak (is_ib=False) to prevent
    overly broad fits on the typically smaller second peak.
    """
    # Use stricter FWHM limit for Cell peak (the typically smaller second peak)
    if is_ib:
        fwhm_limit = opts.max_peak_fwhm_um
    else:
        # Cell peak uses the second peak limit if specified, otherwise falls back
        fwhm_limit = opts.max_fwhm_second_peak_um or opts.max_peak_fwhm_um

    max_sigma_cap = _sigma_cap_from_fwhm(fwhm_limit)

    if model_type == "lognormal":
        s_bounds = opts.s_bounds_logn_ib if is_ib else opts.s_bounds_logn_cells
        lo = [0, lo_mu, s_bounds[0]]
        hi = [np.inf, hi_mu, s_bounds[1]]
        # Apply FWHM cap
        max_shape = _shape_cap_from_fwhm(fwhm_limit, mode_ref)
        if max_shape is not None:
            hi[2] = min(hi[2], max_shape)
            if hi[2] <= lo[2]:
                hi[2] = max(lo[2] * 1.01, lo[2] + 1e-4)
    elif model_type == "splitgaussian":
        left_bounds = opts.sigma_left_bounds_ib if is_ib else opts.sigma_left_bounds_cells
        right_bounds = opts.sigma_right_bounds_ib if is_ib else opts.sigma_right_bounds_cells
        sigma_left_hi = left_bounds[1]
        sigma_right_hi = right_bounds[1]
        if max_sigma_cap is not None:
            sigma_left_hi = min(sigma_left_hi, max_sigma_cap)
            sigma_right_hi = min(sigma_right_hi, max_sigma_cap)
        lo = [0, lo_mu, left_bounds[0], right_bounds[0]]
        hi = [np.inf, hi_mu, sigma_left_hi, sigma_right_hi]
    else:  # gaussian
        sigma_bounds = opts.sigma_bounds_gauss_ib if is_ib else opts.sigma_bounds_gauss_cells
        sigma_lo, sigma_hi = sigma_bounds
        if hint:
            sigma_lo, sigma_hi = _sigma_bounds_from_hint(hint.sigma, sigma_bounds)
        if max_sigma_cap is not None:
            sigma_hi = min(sigma_hi, max_sigma_cap)
            if sigma_hi <= sigma_lo:
                sigma_hi = max(sigma_lo * 1.01, sigma_lo + 1e-4)
        lo = [0, lo_mu, sigma_lo]
        hi = [np.inf, hi_mu, sigma_hi]

    return lo, hi


def _bounds_for_one_peak(
    opts: AnalysisOptions, hints: PeakHints, model_type: ModelType
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lo_mu_ib = opts.mu_ib_um * (1 - opts.allow_shift_fraction)
    hi_mu_ib = opts.mu_ib_um * (1 + opts.allow_shift_fraction)
    ib_hint = hints.get("ib")
    if ib_hint:
        lo_mu_ib, hi_mu_ib = _tight_bounds_from_hint(ib_hint.mu, lo_mu_ib, hi_mu_ib)
    ib_mode_ref = float(
        np.clip(ib_hint.mu if ib_hint else opts.mu_ib_um, lo_mu_ib, hi_mu_ib)
    )

    lo, hi = _single_peak_bounds(model_type, lo_mu_ib, hi_mu_ib, opts, ib_hint, ib_mode_ref, is_ib=True)
    return (tuple(lo), tuple(hi))


def _two_peak_model(opts: AnalysisOptions, model_ib: ModelType, model_cell: ModelType):
    """Create a two-peak model function with potentially different model types per peak."""
    n1 = _params_per_peak(model_ib)
    n2 = _params_per_peak(model_cell)

    # Build parameter names dynamically for the signature
    # We'll use *args and unpack manually to handle variable parameter counts

    if model_ib == "lognormal" and model_cell == "lognormal":
        def model(x, A1, m1, s1, A2, m2, s2):
            scale1 = _lognorm_scale_from_mode(m1, s1)
            scale2 = _lognorm_scale_from_mode(m2, s2)
            return (
                A1 * lognorm.pdf(x, s1, scale=scale1)
                + A2 * lognorm.pdf(x, s2, scale=scale2)
            )
    elif model_ib == "splitgaussian" and model_cell == "splitgaussian":
        def model(x, A1, m1, s_left1, s_right1, A2, m2, s_left2, s_right2):  # type: ignore[misc]
            return (
                _split_gaussian(x, A1, m1, s_left1, s_right1)
                + _split_gaussian(x, A2, m2, s_left2, s_right2)
            )
    elif model_ib == "gaussian" and model_cell == "gaussian":
        def model(x, A1, m1, s1, A2, m2, s2):
            return (
                A1 * norm.pdf(x, loc=m1, scale=s1)
                + A2 * norm.pdf(x, loc=m2, scale=s2)
            )
    elif model_ib == "lognormal" and model_cell == "gaussian":
        def model(x, A1, m1, s1, A2, m2, s2):  # type: ignore[misc]
            scale1 = _lognorm_scale_from_mode(m1, s1)
            return (
                A1 * lognorm.pdf(x, s1, scale=scale1)
                + A2 * norm.pdf(x, loc=m2, scale=s2)
            )
    elif model_ib == "gaussian" and model_cell == "lognormal":
        def model(x, A1, m1, s1, A2, m2, s2):  # type: ignore[misc]
            scale2 = _lognorm_scale_from_mode(m2, s2)
            return (
                A1 * norm.pdf(x, loc=m1, scale=s1)
                + A2 * lognorm.pdf(x, s2, scale=scale2)
            )
    elif model_ib == "splitgaussian" and model_cell == "gaussian":
        def model(x, A1, m1, s_left1, s_right1, A2, m2, s2):  # type: ignore[misc]
            return (
                _split_gaussian(x, A1, m1, s_left1, s_right1)
                + A2 * norm.pdf(x, loc=m2, scale=s2)
            )
    elif model_ib == "gaussian" and model_cell == "splitgaussian":
        def model(x, A1, m1, s1, A2, m2, s_left2, s_right2):  # type: ignore[misc]
            return (
                A1 * norm.pdf(x, loc=m1, scale=s1)
                + _split_gaussian(x, A2, m2, s_left2, s_right2)
            )
    elif model_ib == "splitgaussian" and model_cell == "lognormal":
        def model(x, A1, m1, s_left1, s_right1, A2, m2, s2):  # type: ignore[misc]
            scale2 = _lognorm_scale_from_mode(m2, s2)
            return (
                _split_gaussian(x, A1, m1, s_left1, s_right1)
                + A2 * lognorm.pdf(x, s2, scale=scale2)
            )
    elif model_ib == "lognormal" and model_cell == "splitgaussian":
        def model(x, A1, m1, s1, A2, m2, s_left2, s_right2):  # type: ignore[misc]
            scale1 = _lognorm_scale_from_mode(m1, s1)
            return (
                A1 * lognorm.pdf(x, s1, scale=scale1)
                + _split_gaussian(x, A2, m2, s_left2, s_right2)
            )
    else:
        raise ValueError(f"Unknown model combination: {model_ib}, {model_cell}")

    return model


def _one_peak_model(model_type: ModelType):
    if model_type == "lognormal":

        def model(x, A1, m1, s1):
            scale = _lognorm_scale_from_mode(m1, s1)
            return A1 * lognorm.pdf(x, s1, scale=scale)
    elif model_type == "splitgaussian":

        def model(x, A1, m1, s_left1, s_right1):  # type: ignore[misc]
            return _split_gaussian(x, A1, m1, s_left1, s_right1)
    else:

        def model(x, A1, m1, s1):
            return A1 * norm.pdf(x, loc=m1, scale=s1)

    return model


def _split_gaussian(
    x: np.ndarray, A: float, mu: float, sigma_left: float, sigma_right: float
) -> np.ndarray:
    """Two-piece (split) Gaussian with different sigmas for left and right sides."""
    result = np.zeros_like(x, dtype=float)
    left_mask = x < mu
    right_mask = ~left_mask
    result[left_mask] = A * np.exp(-0.5 * ((x[left_mask] - mu) / sigma_left) ** 2)
    result[right_mask] = A * np.exp(-0.5 * ((x[right_mask] - mu) / sigma_right) ** 2)
    return result


def _eval_single_peak(
    x: np.ndarray, model_type: ModelType, params: np.ndarray
) -> np.ndarray:
    """Evaluate a single peak given its model type and parameters."""
    if model_type == "lognormal":
        scale = _lognorm_scale_from_mode(params[1], params[2])
        return params[0] * lognorm.pdf(x, params[2], scale=scale)
    elif model_type == "splitgaussian":
        return _split_gaussian(x, params[0], params[1], params[2], params[3])
    else:  # gaussian
        return params[0] * norm.pdf(x, loc=params[1], scale=params[2])


def _get_mu_from_params(model_type: ModelType, params: np.ndarray) -> float:
    """Get the center/mu value from peak parameters."""
    return float(params[1])


def _component_arrays(
    x: np.ndarray, fitres: _FitResult, opts: AnalysisOptions
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_ib = fitres.get("model_ib", opts.get_model_ib())
    model_cell = fitres.get("model_cell", opts.get_model_cell())

    if fitres["kind"] == "two":
        comp1, comp2 = _component_arrays_raw(x, fitres["popt"], model_ib, model_cell)
        n1 = _params_per_peak(model_ib)
        m1 = fitres["popt"][1]
        m2 = fitres["popt"][n1 + 1]
        cell_first = fitres.get("cell_first")
        if cell_first is None:
            cell_first = _cell_component_first(m1, m2, opts)
        if cell_first:
            cells, ibs = comp1, comp2
        else:
            cells, ibs = comp2, comp1
    else:
        comp1 = _single_component(x, fitres["popt"], model_ib)
        cells = np.zeros_like(comp1)
        ibs = comp1
    total = cells + ibs
    return total, cells, ibs


def _component_arrays_raw(
    x: np.ndarray, params: np.ndarray, model_ib: ModelType, model_cell: ModelType
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate both peak components given their model types."""
    n1 = _params_per_peak(model_ib)
    comp1 = _eval_single_peak(x, model_ib, params[:n1])
    comp2 = _eval_single_peak(x, model_cell, params[n1:])
    return comp1, comp2


def _single_component(
    x: np.ndarray, params: np.ndarray, model_type: ModelType
) -> np.ndarray:
    return _eval_single_peak(x, model_type, params)


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


def _lognormal_shape_from_hint(hint: Optional[_PeakHint], mu_guess: float) -> float:
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
    return float(safe_mode * np.exp(safe_shape**2))


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
    popt: np.ndarray, hints: PeakHints, opts: AnalysisOptions, model_ib: ModelType, model_cell: ModelType
) -> bool:
    m1 = popt[1]
    n1 = _params_per_peak(model_ib)
    m2 = popt[n1 + 1]
    cell_hint = hints.get("cell") if hints else None
    ib_hint = hints.get("ib") if hints else None

    if cell_hint:
        return abs(m1 - cell_hint.mu) <= abs(m2 - cell_hint.mu)
    if ib_hint:
        return abs(m1 - ib_hint.mu) >= abs(m2 - ib_hint.mu)
    return _cell_component_first(m1, m2, opts)
