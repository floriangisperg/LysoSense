"""Signal analysis helpers for LysoSense."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .io import Measurement


@dataclass
class AnalysisOptions:
    baseline_percentile: float = 5.0
    smooth_window: int = 11


@dataclass
class AnalysisResult:
    measurement: Measurement
    enriched: pd.DataFrame
    metrics: Dict[str, float]
    options: AnalysisOptions


def analyze_measurement(measurement: Measurement, options: AnalysisOptions | None = None) -> AnalysisResult:
    opts = options or AnalysisOptions()

    enriched = _enrich_signal(measurement.data, opts)
    metrics = _compute_metrics(enriched)

    return AnalysisResult(
        measurement=measurement,
        enriched=enriched,
        metrics=metrics,
        options=opts,
    )


def _enrich_signal(df: pd.DataFrame, options: AnalysisOptions) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot analyse an empty measurement")

    enriched = df.sort_values("wavenumber").reset_index(drop=True).copy()
    intensities = enriched["intensity"].to_numpy(dtype=float)

    baseline = np.percentile(intensities, options.baseline_percentile)
    corrected = intensities - baseline
    corrected = np.where(corrected < 0, 0, corrected)

    smooth_window = max(1, int(options.smooth_window))
    smooth_window = min(smooth_window, len(corrected))
    if smooth_window % 2 == 0 and smooth_window > 1:
        smooth_window -= 1  # keep the window symmetric where possible

    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        smoothed = np.convolve(corrected, kernel, mode="same")
    else:
        smoothed = corrected

    peak = smoothed.max()
    if peak > 0:
        normalized = smoothed / peak
    else:
        normalized = np.zeros_like(smoothed)

    enriched["baseline_level"] = baseline
    enriched["intensity_corrected"] = corrected
    enriched["intensity_smooth"] = smoothed
    enriched["normalized_intensity"] = normalized

    return enriched


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    x = df["wavenumber"].to_numpy(dtype=float)
    y_corrected = df["intensity_corrected"].to_numpy(dtype=float)
    y_smoothed = df["intensity_smooth"].to_numpy(dtype=float)
    baseline = float(df["baseline_level"].iloc[0])

    positive_signal = np.clip(y_corrected, 0, None)
    area = float(np.trapz(positive_signal, x))
    span = float(x[-1] - x[0]) if len(x) > 1 else 1.0
    normalized_area = float(area / span) if span else float("nan")

    peak_idx = int(np.argmax(y_smoothed))
    peak_position = float(x[peak_idx])
    peak_intensity = float(y_smoothed[peak_idx])

    centroid = _weighted_centroid(x, positive_signal)
    fwhm = _full_width_half_max(x, y_smoothed)

    tail_start = int(len(y_corrected) * 0.9)
    noise_slice = y_corrected[tail_start:] if tail_start < len(y_corrected) else y_corrected
    noise_std = float(np.std(noise_slice)) if len(noise_slice) else 0.0
    snr = float(peak_intensity / noise_std) if noise_std else float("inf") if peak_intensity > 0 else 0.0

    if np.any(positive_signal):
        positives = positive_signal[np.nonzero(positive_signal)]
        if len(positives):
            denom = float(max(np.min(positives), 1e-9))
            dynamic_range = float(peak_intensity / denom) if denom else float("inf")
        else:
            dynamic_range = float("inf") if peak_intensity > 0 else 0.0
    else:
        dynamic_range = float("inf") if peak_intensity > 0 else 0.0

    return {
        "baseline_level": baseline,
        "area": area,
        "normalized_area": normalized_area,
        "peak_position": peak_position,
        "peak_intensity": peak_intensity,
        "centroid": centroid,
        "fwhm": fwhm,
        "noise_std": noise_std,
        "snr": snr,
        "dynamic_range": dynamic_range,
    }


def _weighted_centroid(x: np.ndarray, y: np.ndarray) -> float:
    total = np.trapz(y, x)
    if not np.isfinite(total) or total == 0:
        return float("nan")
    weighted = np.trapz(x * y, x)
    return float(weighted / total)


def _full_width_half_max(x: np.ndarray, y: np.ndarray) -> float:
    if not len(y):
        return float("nan")
    peak = float(np.max(y))
    if peak <= 0:
        return float("nan")
    half = peak / 2
    indices = np.where(y >= half)[0]
    if not len(indices):
        return float("nan")
    left_idx = indices[0]
    right_idx = indices[-1]

    left_cross = _interpolate_half_crossing(x, y, left_idx, -1, half)
    right_cross = _interpolate_half_crossing(x, y, right_idx, 1, half)
    if left_cross is None or right_cross is None:
        return float("nan")
    return float(right_cross - left_cross)


def _interpolate_half_crossing(
    x: np.ndarray, y: np.ndarray, start_idx: int, step: int, level: float
) -> Optional[float]:
    idx = start_idx
    next_idx = idx + step
    if next_idx < 0 or next_idx >= len(y):
        return float(x[idx])
    y0 = y[idx]
    y1 = y[next_idx]
    if y0 == y1:
        return float(x[idx])
    fraction = (level - y0) / (y1 - y0)
    return float(x[idx] + fraction * (x[next_idx] - x[idx]))
