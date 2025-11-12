from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

import altair as alt
import pandas as pd
import streamlit as st

try:  # Streamlit exposes the UploadedFile type at runtime
    from streamlit.runtime.uploaded_file_manager import UploadedFile
except Exception:  # pragma: no cover - fallback for type checkers
    UploadedFile = Any  # type: ignore

from lysosense import (
    AnalysisOptions,
    AnalysisResult,
    analyze_measurement,
    list_dat_files,
    load_dat_file,
    parse_dat_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (PROJECT_ROOT / "data").resolve()


@st.cache_data(show_spinner=False)
def _load_sample_cached(path_str: str, baseline_percentile: float, smooth_window: int) -> AnalysisResult:
    measurement = load_dat_file(Path(path_str))
    options = AnalysisOptions(
        baseline_percentile=baseline_percentile, smooth_window=int(smooth_window)
    )
    return analyze_measurement(measurement, options)


def main() -> None:
    st.set_page_config(page_title="LysoSense Analyzer", layout="wide")
    st.title("LysoSense Raman Analyzer")
    st.caption(
        "Upload .dat exports or explore the bundled samples to inspect spectra, overlay traces, "
        "and review automated quality metrics."
    )

    options, uploaded_files, selected_sample_paths = _render_sidebar(DATA_DIR)

    results = _collect_results(uploaded_files, selected_sample_paths, options)
    if not results:
        st.info("Add uploads or select sample files in the sidebar to begin.")
        return

    labels = _label_results(results)

    available_labels = [label for label, _ in labels]
    active_labels = st.sidebar.multiselect(
        "Traces to visualize",
        available_labels,
        default=available_labels,
        key="trace-selector",
    )
    active_entries = [(label, res) for label, res in labels if label in active_labels]

    if not active_entries:
        st.warning("Select at least one trace to render the plots.")
        return

    signal_mode = st.radio(
        "Signal view",
        ("Raw intensity", "Baseline corrected", "Smoothed (normalized)"),
        horizontal=True,
    )

    _render_plot(active_entries, signal_mode)
    _render_metrics(active_entries)
    _render_details(active_entries)


def _render_sidebar(data_dir: Path) -> Tuple[AnalysisOptions, Sequence[UploadedFile], List[Path]]:
    st.sidebar.header("Inputs")

    baseline = st.sidebar.slider(
        "Baseline percentile", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )
    smooth_window = st.sidebar.slider(
        "Smoothing window (points)", min_value=3, max_value=51, value=11, step=2
    )
    options = AnalysisOptions(baseline_percentile=baseline, smooth_window=smooth_window)

    uploaded_files = st.sidebar.file_uploader(
        "Upload instrument .dat files",
        type=["dat"],
        accept_multiple_files=True,
    )

    sample_paths = list_dat_files(data_dir)
    sample_names = [path.name for path in sample_paths]
    default_samples = sample_names[:3]
    chosen_samples = st.sidebar.multiselect(
        "Sample data", sample_names, default=default_samples, key="sample-select"
    )

    selected = [data_dir / name for name in chosen_samples if (data_dir / name).exists()]

    st.sidebar.caption(f"Data folder: {data_dir}")

    return options, uploaded_files or [], selected


def _collect_results(
    uploaded_files: Sequence[UploadedFile],
    sample_paths: Sequence[Path],
    options: AnalysisOptions,
) -> List[Tuple[str, AnalysisResult]]:
    results: List[Tuple[str, AnalysisResult]] = []

    for file in uploaded_files:
        measurement = parse_dat_bytes(file.getvalue(), source_name=file.name)
        analysis = analyze_measurement(measurement, options)
        results.append((file.name, analysis))

    for path in sample_paths:
        analysis = _load_sample_cached(
            str(path), options.baseline_percentile, options.smooth_window
        )
        results.append((path.name, analysis))

    return results


def _label_results(results: Sequence[Tuple[str, AnalysisResult]]) -> List[Tuple[str, AnalysisResult]]:
    used: set[str] = set()
    labelled: List[Tuple[str, AnalysisResult]] = []
    for origin, analysis in results:
        base = _format_label(analysis.measurement.name, origin)
        label = _ensure_unique_label(base, used)
        used.add(label)
        labelled.append((label, analysis))
    return labelled


def _format_label(name: str, origin: str) -> str:
    if origin and origin not in (name or ""):
        return f"{name} ({origin})"
    return name or origin


def _ensure_unique_label(base: str, used: set[str]) -> str:
    label = base
    suffix = 2
    while label in used:
        label = f"{base} ({suffix})"
        suffix += 1
    return label


def _render_plot(entries: Sequence[Tuple[str, AnalysisResult]], mode: str) -> None:
    value_column, y_title = {
        "Raw intensity": ("intensity", "Raw intensity"),
        "Baseline corrected": ("intensity_corrected", "Baseline corrected"),
        "Smoothed (normalized)": ("normalized_intensity", "Normalized (smoothed)"),
    }[mode]

    frames: List[pd.DataFrame] = []
    for label, analysis in entries:
        subset = analysis.enriched[
            [
                "wavenumber",
                "intensity",
                "intensity_corrected",
                "intensity_smooth",
                "normalized_intensity",
            ]
        ].copy()
        subset["measurement"] = label
        frames.append(subset)

    plot_df = pd.concat(frames, ignore_index=True)

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("wavenumber:Q", title="Wavenumber (a.u.)"),
            y=alt.Y(f"{value_column}:Q", title=y_title),
            color=alt.Color("measurement:N", title="Trace"),
            tooltip=[
                alt.Tooltip("measurement:N", title="Trace"),
                alt.Tooltip("wavenumber:Q", title="Wavenumber", format=".4f"),
                alt.Tooltip("intensity:Q", title="Raw"),
                alt.Tooltip("intensity_corrected:Q", title="Baseline corrected"),
                alt.Tooltip("intensity_smooth:Q", title="Smoothed"),
            ],
        )
        .interactive()
    )

    st.subheader("Signal overlay")
    st.altair_chart(chart, use_container_width=True)


def _render_metrics(entries: Sequence[Tuple[str, AnalysisResult]]) -> None:
    rows = []
    for label, analysis in entries:
        row = {"measurement": label}
        row.update(analysis.metrics)
        rows.append(row)
    metrics_df = pd.DataFrame(rows).set_index("measurement")
    st.subheader("Automated metrics")
    st.dataframe(metrics_df.style.format("{:.4g}"))


def _render_details(entries: Sequence[Tuple[str, AnalysisResult]]) -> None:
    st.subheader("Per-measurement detail")
    for label, analysis in entries:
        measurement = analysis.measurement
        with st.expander(label):
            meta_df = pd.DataFrame(
                sorted(measurement.metadata.items()), columns=["Field", "Value"]
            )
            st.write("Metadata")
            st.dataframe(meta_df, hide_index=True, use_container_width=True)

            st.write("Sampled signal (first 15 points)")
            preview = analysis.enriched.head(15)[
                ["wavenumber", "intensity", "intensity_corrected", "intensity_smooth"]
            ]
            st.dataframe(preview, use_container_width=True)


if __name__ == "__main__":
    main()
