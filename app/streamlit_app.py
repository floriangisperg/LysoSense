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

ARTICLE_URL = "https://www.sciencedirect.com/science/article/pii/S0168165625002706"


def main() -> None:
    st.set_page_config(page_title="LysoSense CPS Analyzer", layout="wide")
    st.title("LysoSense CPS Analyzer")
    st.markdown(
        "Differential centrifugal sedimentation workflow for tracking intact cells and inclusion bodies "
        "during homogenisation (method adapted from [Klausser et al., 2025](%s))."
        % ARTICLE_URL
    )

    options, show_fit, show_components = _render_sidebar()

    uploaded_files = st.file_uploader(
        "Upload CPS/DCS .dat files",
        type=["dat"],
        accept_multiple_files=True,
        help="Drop multiple runs at once to compare peak areas and lysis efficiency.",
    )

    if not uploaded_files:
        st.info("Upload one or more .dat exports to begin the analysis.")
        return

    results = _analyze_uploads(uploaded_files, options)
    if not results:
        st.warning("All uploaded files failed to parse. Please verify the file format.")
        return

    labels = [label for label, _ in results]
    active_labels = st.multiselect(
        "Traces to visualise",
        labels,
        default=labels,
        help="Use this control to focus on a subset of uploaded measurements.",
    )

    active_results = [(label, res) for label, res in results if label in active_labels]
    if not active_results:
        st.warning("Select at least one measurement to render plots and metrics.")
        return

    _render_plot(active_results, show_fit, show_components)
    summary_df = _render_metrics(active_results)
    _render_download(summary_df)
    _render_details(active_results)


def _render_sidebar() -> Tuple[AnalysisOptions, bool, bool]:
    st.sidebar.header("Model settings")
    model = st.sidebar.radio("Peak model", ("gaussian", "lognormal"), horizontal=True)
    mu_ib = st.sidebar.number_input("IB target size (um)", value=0.48, min_value=0.1, max_value=2.0, step=0.01)
    mu_cell = st.sidebar.number_input("Cell target size (um)", value=0.85, min_value=0.1, max_value=3.0, step=0.01)
    allow_shift = st.sidebar.slider("Allowed peak shift (%)", min_value=5, max_value=40, value=20, step=1)
    second_peak = st.sidebar.slider(
        "Minimum 2nd peak fraction", min_value=0.0, max_value=0.20, value=0.02, step=0.01
    )
    show_fit = st.sidebar.checkbox("Show fitted envelope", value=True)
    show_components = st.sidebar.checkbox("Show component contributions", value=True)

    options = AnalysisOptions(
        model=model,  # type: ignore[arg-type]
        mu_ib_um=float(mu_ib),
        mu_cell_um=float(mu_cell),
        allow_shift_fraction=allow_shift / 100.0,
        second_peak_min_frac=float(second_peak),
    )
    return options, show_fit, show_components


def _analyze_uploads(
    uploaded_files: Sequence[UploadedFile],
    options: AnalysisOptions,
) -> List[Tuple[str, AnalysisResult]]:
    results: List[Tuple[str, AnalysisResult]] = []
    for file in uploaded_files:
        try:
            measurement = parse_dat_bytes(file.getvalue(), source_name=file.name)
            analysis = analyze_measurement(measurement, options)
            results.append((file.name, analysis))
        except Exception as exc:
            st.error(f"{file.name}: {exc}")
    return results


def _render_plot(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
) -> None:
    fig = go.Figure()
    for label, analysis in entries:
        observed = analysis.observed
        fig.add_trace(
            go.Scatter(
                x=observed["particle_size_um"],
                y=observed["mass_signal_ug"],
                name=f"{label} - raw",
                mode="lines",
            )
        )
        if show_fit:
            fig.add_trace(
                go.Scatter(
                    x=analysis.dense_fit["particle_size_um"],
                    y=analysis.dense_fit["fit_signal_ug"],
                    name=f"{label} - fit",
                    mode="lines",
                    line=dict(dash="dash"),
                )
            )
        if show_components:
            if analysis.dense_fit["cells_component_ug"].any():
                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["cells_component_ug"],
                        name=f"{label} - cells",
                        mode="lines",
                        line=dict(width=1.5),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=analysis.dense_fit["particle_size_um"],
                    y=analysis.dense_fit["ibs_component_ug"],
                    name=f"{label} - IBs",
                    mode="lines",
                    line=dict(width=1.5),
                )
            )

    fig.update_layout(
        xaxis_title="Particle size (um)",
        yaxis_title="D * Wd (ug)",
        legend_title="Trace",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
    )
    st.subheader("Particle size distribution")
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(entries: Sequence[Tuple[str, AnalysisResult]]) -> pd.DataFrame:
    records: List[Dict[str, float | str | None]] = []
    for label, analysis in entries:
        row = {"measurement": label}
        row.update(analysis.metrics)
        records.append(row)
    summary = pd.DataFrame(records).set_index("measurement")
    st.subheader("Relative abundance and lysis efficiency")
    numeric_cols = summary.select_dtypes(include="number").columns
    formatters = {col: "{:.4g}" for col in numeric_cols}
    st.dataframe(summary.style.format(formatters))
    return summary.reset_index()


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


if __name__ == "__main__":
    main()
