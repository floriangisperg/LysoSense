"""Pure data transforms for CPS/DCS measurements.

These helpers operate on :class:`~lysosense.io.Measurement` and
:class:`~lysosense.analysis.AnalysisResult` objects and contain no Streamlit
dependencies, so they can be unit-tested directly. The Streamlit app imports
them from here instead of defining them inline.
"""

from __future__ import annotations

import numpy as np

from .analysis import AnalysisResult
from .io import Measurement


class NormalizationSkipped(ValueError):
    """Raised by :func:`normalize_measurement` when the signal max is non-positive.

    The Streamlit app catches this and surfaces ``str(exc)`` as a warning while
    keeping the original measurement, mirroring the previous inline behaviour.
    """

    def __init__(self, factor: float, name: str) -> None:
        self.factor = factor
        self.name = name
        super().__init__(
            f"Normalization factor is {factor:.2e}, "
            f"skipping normalization for {name}"
        )


def calculate_r_squared(result: AnalysisResult) -> float:
    """Coefficient of determination R² between the observed and fitted signal."""

    observed = result.observed
    y_actual = observed["mass_signal_ug"].to_numpy()
    y_predicted = observed["fit_signal_ug"].to_numpy()

    ss_res = float(np.sum((y_actual - y_predicted) ** 2))
    ss_tot = float(np.sum((y_actual - float(np.mean(y_actual))) ** 2))

    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)
    return max(0.0, r_squared)  # Ensure non-negative


def subtract_baseline(
    measurement: Measurement, method: str = "minimum"
) -> Measurement:
    """Return a new measurement with the baseline subtracted from the signal."""

    if measurement.data.empty:
        return measurement

    df = measurement.data.copy()
    x = df["particle_size_um"].to_numpy()
    y = df["mass_signal_ug"].to_numpy()

    if method == "minimum":
        baseline_value = np.min(y)
        y_corrected = y - baseline_value

    elif method == "percentile":
        baseline_value = np.percentile(y, 1)
        y_corrected = y - baseline_value

    elif method == "linear":
        # Fit line through first 10% and last 10% of points
        n_points = len(x)
        n_edge = max(10, int(0.1 * n_points))

        # Get edge points for baseline fitting
        x_edge = np.concatenate([x[:n_edge], x[-n_edge:]])
        y_edge = np.concatenate([y[:n_edge], y[-n_edge:]])

        # Fit linear baseline
        coeffs = np.polyfit(x_edge, y_edge, 1)
        baseline = np.polyval(coeffs, x)
        y_corrected = y - baseline

    else:
        raise ValueError(f"Unknown baseline method: {method}")

    # Ensure non-negative values
    y_corrected = np.maximum(y_corrected, 0)

    # Create new measurement with baseline-corrected data
    corrected_data = df.copy()
    corrected_data["mass_signal_ug"] = y_corrected

    # Store baseline info in metadata
    corrected_metadata = measurement.metadata.copy()
    corrected_metadata["baseline_subtracted"] = True
    corrected_metadata["baseline_method"] = method

    return Measurement(
        name=f"{measurement.name}_baseline_corrected",
        metadata=corrected_metadata,
        data=corrected_data,
        source=measurement.source,
        notes=measurement.notes + [f"Baseline corrected using {method} method"],
    )


def normalize_measurement(measurement: Measurement) -> Measurement:
    """Return a new measurement with the signal scaled to a maximum of 1.

    Raises :class:`NormalizationSkipped` when the maximum signal is
    non-positive; callers are expected to catch it and warn the user while
    keeping the original measurement.
    """

    if measurement.data.empty:
        return measurement

    df = measurement.data.copy()
    y = df["mass_signal_ug"].to_numpy()

    # Use maximum signal value for normalization
    normalization_factor = np.max(y)

    # Avoid division by zero / negative scaling
    if normalization_factor <= 0:
        raise NormalizationSkipped(float(normalization_factor), measurement.name)

    # Apply normalization
    y_normalized = y / normalization_factor

    # Create new measurement with normalized data
    normalized_data = df.copy()
    normalized_data["mass_signal_ug"] = y_normalized

    # Store normalization info in metadata
    normalized_metadata = measurement.metadata.copy()
    normalized_metadata["normalized"] = True
    normalized_metadata["normalization_method"] = "max_intensity"
    normalized_metadata["normalization_factor"] = normalization_factor

    return Measurement(
        name=f"{measurement.name}_normalized",
        metadata=normalized_metadata,
        data=normalized_data,
        source=measurement.source,
        notes=measurement.notes
        + [f"Normalized to max intensity (factor: {normalization_factor:.2e})"],
    )


def clip_measurement_range(
    measurement: Measurement, min_size: float, max_size: float
) -> Measurement:
    """Restrict the measurement to the ``[min_size, max_size]`` particle-size window."""

    if measurement.data.empty:
        return measurement

    df = measurement.data.copy()
    mask = df["particle_size_um"].between(min_size, max_size)

    if not mask.any():
        # If no points fall in range, keep the original to avoid empty fits.
        return measurement

    clipped_data = df.loc[mask].reset_index(drop=True)

    clipped_metadata = measurement.metadata.copy()
    clipped_metadata["clip_min_um"] = min_size
    clipped_metadata["clip_max_um"] = max_size

    return Measurement(
        name=measurement.name,
        metadata=clipped_metadata,
        data=clipped_data,
        source=measurement.source,
        notes=measurement.notes
        + [f"Clipped to {min_size:.2f}-{max_size:.2f} µm range"],
    )
