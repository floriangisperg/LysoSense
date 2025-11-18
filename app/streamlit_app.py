from __future__ import annotations

import io
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except Exception:  # pragma: no cover
    UploadedFile = Any  # type: ignore

from lysosense import (
    AnalysisOptions,
    AnalysisResult,
    analyze_measurement,
    parse_dat_bytes,
)
from scipy.optimize import curve_fit
import numpy as np

ARTICLE_URL = "https://www.sciencedirect.com/science/article/pii/S0168165625002706"


def main() -> None:
    st.set_page_config(page_title="LysoSense CPS Analyzer", layout="wide")
    st.title("LysoSense CPS Analyzer")
    st.markdown(
        "Differential centrifugal sedimentation workflow for tracking intact cells and inclusion bodies "
        "during homogenisation (method adapted from [Klausser et al., 2025](%s))."
        % ARTICLE_URL
    )

    options, show_fit, show_components, view_mode, compare_models, baseline_subtraction, baseline_method, normalize_data, uploaded_files = _render_sidebar()

    if not uploaded_files:
        st.info("📁 Upload .dat files in the sidebar to begin the analysis.")
        return

    results = _analyze_uploads(uploaded_files, options, normalize_data)
    if not results:
        st.warning("All uploaded files failed to parse. Please verify the file format.")
        return

    labels = [label for label, _ in results]
    active_labels = st.multiselect(
        "Traces to analyze",
        labels,
        default=labels,
        help="Use this control to focus on a subset of uploaded measurements.",
    )

    active_results = [(label, res) for label, res in results if label in active_labels]
    if not active_results:
        st.warning("Select at least one measurement to render plots and metrics.")
        return

    # Create tab interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Individual Samples",
        "📈 Results Table",
        "ℹ️ Detailed Information"
    ])

    with tab1:
        _render_overview_tab(active_results, show_fit, show_components, view_mode)

    with tab2:
        _render_individual_samples_tab(active_results, show_fit, show_components, view_mode)

    with tab3:
        summary_df = _render_results_tab(active_results)

    with tab4:
        _render_details_tab(active_results)

    # Download button stays at bottom
    _render_download(summary_df)


def _render_sidebar() -> Tuple[AnalysisOptions, bool, bool, str, bool, bool, bool, bool]:
    # Data upload section (always expanded)
    with st.sidebar.expander("📁 Data Upload", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload CPS/DCS .dat files",
            type=["dat"],
            accept_multiple_files=True,
            help="Drop multiple runs at once to compare peak areas and lysis efficiency.",
            key="file_uploader"
        )

    # Only show other sections if files are uploaded
    if uploaded_files:
        # Data preprocessing section
        with st.sidebar.expander("🔧 Data Preprocessing", expanded=True):
            baseline_subtraction = st.checkbox(
                "Baseline subtraction",
                value=False,
                help="Subtract baseline from raw data before fitting",
                key="baseline_subtraction"
            )

            baseline_method = st.selectbox(
                "Baseline method",
                ("minimum", "percentile", "linear"),
                help="• Minimum: Use minimum signal value\n• Percentile: Use 1st percentile\n• Linear: Linear fit to edges",
                disabled=not baseline_subtraction,
                key="baseline_method"
            )

            st.markdown("---")  # Separator

            normalize_data = st.checkbox(
                "Normalize data",
                value=False,
                help="Normalize data to enable comparison between samples with different concentrations",
                key="normalize_data"
            )

            if normalize_data:
                st.markdown("**Method**: Max intensity normalization (scales to maximum signal value)")

        # Model settings section
        with st.sidebar.expander("⚙️ Model Settings", expanded=True):
            model = st.radio("Peak model", ("gaussian", "lognormal", "autofit"), horizontal=True, key="model")
            compare_models = (model == "autofit")

            st.markdown("**Peak Parameters**")
            mu_ib = st.number_input("IB target size (µm)", value=0.48, min_value=0.1, max_value=2.0, step=0.01, key="mu_ib")
            mu_cell = st.number_input("Cell target size (µm)", value=0.85, min_value=0.1, max_value=3.0, step=0.01, key="mu_cell")

            st.markdown("**Fitting Constraints**")
            allow_shift = st.slider("Allowed peak shift (%)", min_value=5, max_value=40, value=20, step=1, key="allow_shift")
            second_peak = st.slider("Min 2nd peak fraction", min_value=0.0, max_value=0.20, value=0.02, step=0.01, key="second_peak")

        # Visualization section (merged with display options)
        with st.sidebar.expander("📊 Visualization", expanded=True):
            view_mode = st.radio(
                "View mode",
                ("Combined", "Fit Overview", "Raw Data Only"),
                help="• Combined: Raw data + fits + components\n• Fit Overview: Only fitted components\n• Raw Data Only: Just raw measurements",
                key="view_mode"
            )

            st.markdown("**Display Options**")
            show_fit = st.checkbox("Show fitted envelope", value=True, key="show_fit")
            show_components = st.checkbox("Show component contributions", value=True, key="show_components")

        # Quick actions section
        with st.sidebar.expander("⚡ Quick Actions", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset All", help="Reset all settings to defaults"):
                    # Clear all widget state to reset to defaults
                    for key in list(st.session_state.keys()):
                        if key.startswith(('view_mode', 'model', 'autofit', 'mu_ib', 'mu_cell',
                                        'allow_shift', 'second_peak', 'show_fit', 'show_components',
                                        'baseline_subtraction', 'baseline_method')):
                            del st.session_state[key]
                    st.rerun()

            with col2:
                if st.button("💾 Save Config", help="Save current configuration"):
                    st.info("Configuration saved (feature coming soon)")
    else:
        # Return defaults when no files uploaded
        model = "gaussian"
        compare_models = False
        mu_ib = 0.48
        mu_cell = 0.85
        allow_shift = 20
        second_peak = 0.02
        show_fit = True
        show_components = True
        baseline_subtraction = False
        baseline_method = "minimum"
        view_mode = "Combined"
        normalize_data = False

    # Don't create AnalysisOptions here anymore since it depends on the model choice
    # Create a placeholder with default values that will be overridden in analysis
    options = AnalysisOptions(
        model="gaussian",  # placeholder, will be overridden
        mu_ib_um=float(mu_ib),
        mu_cell_um=float(mu_cell),
        allow_shift_fraction=allow_shift / 100.0,
        second_peak_min_frac=float(second_peak),
    )
    return options, show_fit, show_components, view_mode, compare_models, baseline_subtraction, baseline_method, normalize_data, uploaded_files


def _analyze_uploads(
    uploaded_files: Sequence[UploadedFile],
    options: AnalysisOptions,
    normalize_data: bool,
) -> List[Tuple[str, AnalysisResult]]:
    results: List[Tuple[str, AnalysisResult]] = []
    model = st.session_state.get('model', 'gaussian')
    compare_models = (model == 'autofit')
    baseline_subtraction = st.session_state.get('baseline_subtraction', False)
    baseline_method = st.session_state.get('baseline_method', 'minimum')

    for file in uploaded_files:
        try:
            measurement = parse_dat_bytes(file.getvalue(), source_name=file.name)

            # Apply baseline subtraction if requested
            if baseline_subtraction:
                measurement = _subtract_baseline(measurement, baseline_method)

            # Apply normalization if requested
            if normalize_data:
                measurement = _normalize_data(measurement)

            if compare_models:
                # Fit both models and choose the better one
                gaussian_result = analyze_measurement(measurement, AnalysisOptions(
                    model="gaussian",
                    mu_ib_um=options.mu_ib_um,
                    mu_cell_um=options.mu_cell_um,
                    allow_shift_fraction=options.allow_shift_fraction,
                    second_peak_min_frac=options.second_peak_min_frac,
                ))

                lognormal_result = analyze_measurement(measurement, AnalysisOptions(
                    model="lognormal",
                    mu_ib_um=options.mu_ib_um,
                    mu_cell_um=options.mu_cell_um,
                    allow_shift_fraction=options.allow_shift_fraction,
                    second_peak_min_frac=options.second_peak_min_frac,
                ))

                # Calculate R² for both models
                gaussian_r2 = _calculate_r_squared(gaussian_result)
                lognormal_r2 = _calculate_r_squared(lognormal_result)

                # Choose the better model
                if gaussian_r2 > lognormal_r2:
                    analysis = gaussian_result
                else:
                    analysis = lognormal_result
            else:
                # Use selected model (but not "autofit" since that's handled above)
                actual_model = "gaussian" if model == "autofit" else model
                actual_options = AnalysisOptions(
                    model=actual_model,  # type: ignore[arg-type]
                    mu_ib_um=options.mu_ib_um,
                    mu_cell_um=options.mu_cell_um,
                    allow_shift_fraction=options.allow_shift_fraction,
                    second_peak_min_frac=options.second_peak_min_frac,
                )
                analysis = analyze_measurement(measurement, actual_options)

            results.append((file.name, analysis))
        except Exception as exc:
            st.error(f"{file.name}: {exc}")
    return results


def _calculate_r_squared(result: AnalysisResult) -> float:
    """Calculate R² for the fit."""
    observed = result.observed
    y_actual = observed["mass_signal_ug"].values
    y_predicted = observed["fit_signal_ug"].values

    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if ss_tot == 0:
        return 0.0

    r_squared = 1 - (ss_res / ss_tot)
    return max(0.0, r_squared)  # Ensure non-negative


def _subtract_baseline(measurement, method: str = "minimum"):
    """Subtract baseline from measurement data."""
    from lysosense.io import Measurement

    if measurement.data.empty:
        return measurement

    df = measurement.data.copy()
    x = df["particle_size_um"].values
    y = df["mass_signal_ug"].values

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
        notes=measurement.notes + [f"Baseline corrected using {method} method"]
    )


def _normalize_data(measurement):
    """Normalize measurement data to maximum intensity."""
    from lysosense.io import Measurement

    if measurement.data.empty:
        return measurement

    df = measurement.data.copy()
    y = df["mass_signal_ug"].values

    # Use maximum signal value for normalization
    normalization_factor = np.max(y)

    # Avoid division by zero
    if normalization_factor <= 0:
        st.warning(f"Normalization factor is {normalization_factor:.2e}, skipping normalization for {measurement.name}")
        return measurement

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
        notes=measurement.notes + [f"Normalized to max intensity (factor: {normalization_factor:.2e})"]
    )


def _render_raw_data_plot(entries: Sequence[Tuple[str, AnalysisResult]]) -> None:
    """Plot only raw data traces with distinct colors per sample."""
    fig = go.Figure()

    # Color palette for samples
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (label, analysis) in enumerate(entries):
        color = colors[i % len(colors)]
        observed = analysis.observed

        # Extract sample name (remove file extension for cleaner legend)
        sample_name = label.replace('.dat', '')

        fig.add_trace(
            go.Scatter(
                x=observed["particle_size_um"],
                y=observed["mass_signal_ug"],
                name=sample_name,
                mode="lines",
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title="D * Wd (µg)",
        legend_title="Sample",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis=dict(range=[0.2, 1.2]),
    )
    # Check if any samples are normalized
    normalized_samples = [label for label, analysis in entries
                         if analysis.measurement.metadata.get('normalized', False)]

    title = "Raw Particle Size Distributions"
    if normalized_samples:
        title += " (Normalized Data)"

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


def _render_fit_overview(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
) -> None:
    """Organized plot showing only fitted data with sample-specific colors and grouped legends."""
    fig = go.Figure()

    # Color palette for samples (same as raw data view for consistency)
    sample_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (label, analysis) in enumerate(entries):
        color = sample_colors[i % len(sample_colors)]
        sample_name = label.replace('.dat', '')
        group_name = f"group_{i}"

        # Show fit envelope if requested
        if show_fit:
            fig.add_trace(
                go.Scatter(
                    x=analysis.dense_fit["particle_size_um"],
                    y=analysis.dense_fit["fit_signal_ug"],
                    name="Fit",
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    legendgroup=group_name,
                    legendgrouptitle_text=sample_name,
                )
            )

        # Show components if requested
        if show_components:
            # Show cells component if it exists
            if analysis.dense_fit["cells_component_ug"].any():
                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["cells_component_ug"],
                        name="Cells",
                        mode="lines",
                        line=dict(color=color, width=2),
                        legendgroup=group_name,
                        legendgrouptitle_text=sample_name,
                    )
                )

            # Always show IBs component
            fig.add_trace(
                go.Scatter(
                    x=analysis.dense_fit["particle_size_um"],
                    y=analysis.dense_fit["ibs_component_ug"],
                    name="IBs",
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=group_name,
                    legendgrouptitle_text=sample_name,
                )
            )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title="D * Wd (µg)",
        legend_title="Samples & Components",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis=dict(range=[0.2, 1.2]),
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggleothers",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        )
    )

    # Add legend guide
    st.markdown("**Legend Guide:** Click sample names to toggle all traces • Click individual traces to toggle • Line styles: solid=cells, dashed=fit, dotted=IBs")

    # Check if any samples are normalized
    normalized_samples = [label for label, analysis in entries
                         if analysis.measurement.metadata.get('normalized', False)]

    title = "Fit Components Overview"
    if normalized_samples:
        title += " (Normalized Data)"

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


def _render_plot(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
) -> None:
    """Combined plot with grouped legends for better organization."""
    fig = go.Figure()

    # Color palette for samples
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (label, analysis) in enumerate(entries):
        color = colors[i % len(colors)]
        sample_name = label.replace('.dat', '')
        group_name = f"group_{i}"
        observed = analysis.observed

        # Raw data
        fig.add_trace(
            go.Scatter(
                x=observed["particle_size_um"],
                y=observed["mass_signal_ug"],
                name="Raw",
                mode="lines",
                line=dict(color=color, width=2),
                legendgroup=group_name,
                legendgrouptitle_text=sample_name,
            )
        )

        if show_fit:
            fig.add_trace(
                go.Scatter(
                    x=analysis.dense_fit["particle_size_um"],
                    y=analysis.dense_fit["fit_signal_ug"],
                    name="Fit",
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    legendgroup=group_name,
                    legendgrouptitle_text=sample_name,
                )
            )

        if show_components:
            if analysis.dense_fit["cells_component_ug"].any():
                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["cells_component_ug"],
                        name="Cells",
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        legendgroup=group_name,
                        legendgrouptitle_text=sample_name,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=analysis.dense_fit["particle_size_um"],
                    y=analysis.dense_fit["ibs_component_ug"],
                    name="IBs",
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    legendgroup=group_name,
                    legendgrouptitle_text=sample_name,
                )
            )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title="D * Wd (µg)",
        legend_title="Samples & Trace Types",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis=dict(range=[0.2, 1.2]),
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggleothers",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        )
    )

    # Add legend guide
    st.markdown("**Legend Guide:** Click sample names to toggle all traces • Line styles: solid=raw/cells, dashed=fit, dotted=IBs")

    # Check if any samples are normalized
    normalized_samples = [label for label, analysis in entries
                         if analysis.measurement.metadata.get('normalized', False)]

    title = "Combined Particle Size Distribution"
    if normalized_samples:
        title += " (Normalized Data)"

    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(entries: Sequence[Tuple[str, AnalysisResult]]) -> pd.DataFrame:
    records: List[Dict[str, float | str | None]] = []
    for label, analysis in entries:
        row = {"measurement": label}
        row.update(analysis.metrics)

        # Add fit quality metrics
        r_squared = _calculate_r_squared(analysis)
        row["r_squared"] = r_squared
        row["fit_quality"] = _get_fit_quality_label(r_squared)

        # Add baseline information
        baseline_subtracted = analysis.measurement.metadata.get('baseline_subtracted', False)
        if baseline_subtracted:
            baseline_method = analysis.measurement.metadata.get('baseline_method', 'unknown')
            row["baseline_corrected"] = f"Yes ({baseline_method})"
        else:
            row["baseline_corrected"] = "No"

        # Add normalization information
        normalized = analysis.measurement.metadata.get('normalized', False)
        if normalized:
            norm_method = analysis.measurement.metadata.get('normalization_method', 'unknown')
            norm_factor = analysis.measurement.metadata.get('normalization_factor', 1.0)
            row["normalized"] = f"Yes ({norm_method}, {norm_factor:.2e})"
        else:
            row["normalized"] = "No"

        records.append(row)

    summary = pd.DataFrame(records).set_index("measurement")

    # Reorder columns for better readability
    column_order = [
        "model", "fit_kind", "baseline_corrected", "normalized", "r_squared", "fit_quality",
        "area_cells", "area_inclusion_bodies", "area_total",
        "intact_fraction", "lysis_efficiency",
        "mean_cell_µm", "mean_ib_µm"
    ]

    # Only include columns that exist
    existing_columns = [col for col in column_order if col in summary.columns]
    other_columns = [col for col in summary.columns if col not in column_order]
    final_columns = existing_columns + other_columns

    summary = summary[final_columns]

    st.subheader("Relative abundance and lysis efficiency")
    numeric_cols = summary.select_dtypes(include="number").columns

    # Custom formatters
    formatters = {col: "{:.4g}" for col in numeric_cols if col != "r_squared"}
    formatters["r_squared"] = "{:.4f}"

    # Style the dataframe with conditional formatting for fit quality
    styled_summary = summary.style.format(formatters)

    # Add color coding for R² values
    def highlight_r_squared(val):
        if val >= 0.95:
            return 'background-color: #d4edda'  # Green - excellent
        elif val >= 0.90:
            return 'background-color: #fff3cd'  # Yellow - good
        elif val >= 0.80:
            return 'background-color: #f8d7da'  # Light red - fair
        else:
            return 'background-color: #f5c6cb'  # Dark red - poor

    styled_summary = styled_summary.applymap(highlight_r_squared, subset=['r_squared'])

    st.dataframe(styled_summary)

    # Add fit quality legend
    st.markdown("""
    **Fit Quality Legend:**
    - 🟢 R² ≥ 0.95: Excellent fit
    - 🟡 R² ≥ 0.90: Good fit
    - 🟠 R² ≥ 0.80: Fair fit
    - 🔴 R² < 0.80: Poor fit
    """)

    return summary.reset_index()


def _get_fit_quality_label(r_squared: float) -> str:
    """Get a descriptive label for the fit quality."""
    if r_squared >= 0.95:
        return "Excellent"
    elif r_squared >= 0.90:
        return "Good"
    elif r_squared >= 0.80:
        return "Fair"
    else:
        return "Poor"


def _render_download(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
    buffer.seek(0)
    st.download_button(
        "Download summary (XLSX)",
        data=buffer.getvalue(),
        file_name="lysosense_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _render_details(entries: Sequence[Tuple[str, AnalysisResult]]) -> None:
    st.subheader("Detailed run information")
    for label, analysis in entries:
        measurement = analysis.measurement
        with st.expander(label):
            meta_df = pd.DataFrame(
                sorted(measurement.metadata.items()), columns=["Field", "Value"]
            )
            st.markdown("**Metadata**")
            st.dataframe(meta_df, hide_index=True, use_container_width=True)

            st.markdown("**Observed trace (first 15 points)**")
            preview = analysis.observed.head(15)[
                [
                    "particle_size_um",
                    "mass_signal_ug",
                    "fit_signal_ug",
                    "cells_component_ug",
                    "ibs_component_ug",
                ]
            ]
            st.dataframe(preview, use_container_width=True)


def _render_overview_tab(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    view_mode: str
) -> None:
    """Render the Overview tab with combined plots."""
    st.markdown("### Combined Analysis Overview")
    st.markdown("Overview of all selected samples with fitted components and metrics.")

    # Render based on view mode
    if view_mode == "Raw Data Only":
        _render_raw_data_plot(entries)
    elif view_mode == "Fit Overview":
        _render_fit_overview(entries, show_fit, show_components)
    else:  # Combined view (original)
        _render_plot(entries, show_fit, show_components)


def _render_individual_samples_tab(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    view_mode: str
) -> None:
    """Render individual samples in a grid layout."""
    st.markdown("### Individual Sample Analysis")
    st.markdown("Detailed view of each selected sample in a grid layout.")

    if not entries:
        st.info("No samples to display.")
        return

    # Calculate grid layout
    n_samples = len(entries)
    n_cols = min(3, max(1, n_samples))  # 1-3 columns based on sample count
    n_rows = (n_samples + n_cols - 1) // n_cols

    # Create grid
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < n_samples:
                with cols[col]:
                    label, analysis = entries[idx]
                    sample_name = label.replace('.dat', '')

                    st.markdown(f"**{sample_name}**")

                    # Create individual plot for this sample
                    fig = _create_individual_sample_plot(
                        [(label, analysis)], show_fit, show_components, view_mode, sample_name
                    )
                    st.plotly_chart(fig, use_container_width=True)


def _create_individual_sample_plot(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    view_mode: str,
    sample_name: str
) -> go.Figure:
    """Create a plot for a single sample."""
    fig = go.Figure()

    # Use a consistent color for the sample
    color = '#1f77b4'

    for label, analysis in entries:
        observed = analysis.observed

        if view_mode == "Raw Data Only":
            # Only raw data
            fig.add_trace(
                go.Scatter(
                    x=observed["particle_size_um"],
                    y=observed["mass_signal_ug"],
                    name="Raw Data",
                    mode="lines",
                    line=dict(color=color, width=2),
                )
            )
        elif view_mode == "Fit Overview":
            # Only fitted components
            if show_fit:
                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["fit_signal_ug"],
                        name="Fit",
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                    )
                )

            if show_components:
                if analysis.dense_fit["cells_component_ug"].any():
                    fig.add_trace(
                        go.Scatter(
                            x=analysis.dense_fit["particle_size_um"],
                            y=analysis.dense_fit["cells_component_ug"],
                            name="Cells",
                            mode="lines",
                            line=dict(color=color, width=1.5),
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["ibs_component_ug"],
                        name="IBs",
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="dot"),
                    )
                )
        else:  # Combined view
            # Raw data
            fig.add_trace(
                go.Scatter(
                    x=observed["particle_size_um"],
                    y=observed["mass_signal_ug"],
                    name="Raw",
                    mode="lines",
                    line=dict(color=color, width=2),
                )
            )

            if show_fit:
                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["fit_signal_ug"],
                        name="Fit",
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                    )
                )

            if show_components:
                if analysis.dense_fit["cells_component_ug"].any():
                    fig.add_trace(
                        go.Scatter(
                            x=analysis.dense_fit["particle_size_um"],
                            y=analysis.dense_fit["cells_component_ug"],
                            name="Cells",
                            mode="lines",
                            line=dict(color=color, width=1.5),
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["ibs_component_ug"],
                        name="IBs",
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="dot"),
                    )
                )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title="D * Wd (µg)",
        template="plotly_white",
        margin=dict(l=40, r=10, t=20, b=40),  # Reduced top margin since no title
        height=300,  # Compact height for grid layout
        xaxis=dict(range=[0.2, 1.2]),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        )
    )

    return fig


def _render_results_tab(entries: Sequence[Tuple[str, AnalysisResult]]) -> pd.DataFrame:
    """Render the Results Table tab."""
    st.markdown("### Analysis Results")
    st.markdown("Detailed metrics and fit quality for all selected samples.")

    summary_df = _render_metrics(entries)
    return summary_df


def _render_details_tab(entries: Sequence[Tuple[str, AnalysisResult]]) -> None:
    """Render the Detailed Information tab."""
    st.markdown("### Detailed Run Information")
    st.markdown("Comprehensive metadata and data preview for each sample.")

    _render_details(entries)


if __name__ == "__main__":
    main()
