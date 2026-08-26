from __future__ import annotations

import io
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

# Import the repo's `src/lysosense` package, not an older copy from site-packages.
# Streamlit (re)runs this file with sys.path already populated; if `src` is present
# but not first, a pip-installed `lysosense` can win and lack newer AnalysisOptions.
_repo_root = Path(__file__).resolve().parent.parent
_src = str(_repo_root / "src")
if _src in sys.path:
    sys.path.remove(_src)
sys.path.insert(0, _src)

# Only evict an already-loaded `lysosense` when it is NOT the copy under this
# repo's `src/` (e.g. a stale pip-installed build). Evicting unconditionally used
# to break `st.cache_data`: every rerun re-imported the package and created *new*
# `Measurement`/`AnalysisResult` class objects, so pickling a cached analysis
# failed with "it's not the same object as lysosense.io.Measurement" and every
# uploaded file errored. With a single stable class identity the cache pickles
# and unpickles cleanly across reruns.
import lysosense as _lysosense  # noqa: E402
_lysosense_file = getattr(_lysosense, "__file__", None)
if not _lysosense_file or not Path(_lysosense_file).resolve().is_relative_to(_repo_root):
    for _name in list(sys.modules):
        if _name == "lysosense" or _name.startswith("lysosense."):
            del sys.modules[_name]

try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile  # noqa: E402
except Exception:  # pragma: no cover
    UploadedFile = Any  # type: ignore

from lysosense import (  # noqa: E402
    AnalysisOptions,
    AnalysisResult,
    NormalizationSkipped,
    analyze_measurement,
    calculate_r_squared,
    clip_measurement_range,
    normalize_measurement,
    parse_dat_bytes,
    subtract_baseline,
)
from lysosense._version import CHANGELOG, __version__  # noqa: E402
from lysosense.analysis import (  # noqa: E402
    _ALL_MODELS,
    _FitSnapshot,
    _analyze_fit_only,
    _build_precomputed_hints,
    _finalize,
    _parallel_map,
)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on error."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_normalized_entries(entries: Sequence[Tuple[str, AnalysisResult]]) -> bool:
    return any(
        analysis.measurement.metadata.get("normalized", False)
        for _, analysis in entries
    )


def _signal_yaxis_title(entries: Sequence[Tuple[str, AnalysisResult]]) -> str:
    return "Rel Weight" if _has_normalized_entries(entries) else "D * Wd (µg)"


def _apply_size_axis_scale(fig: go.Figure, log_size_axis: bool) -> None:
    """Switch the particle-size x-axis to logarithmic.

    Display only — the fit and lysis metrics are computed in linear µm and are
    unaffected. Any non-positive sizes (rare in CPS/DCS exports) are dropped by
    Plotly on a log axis rather than rendered.
    """
    if log_size_axis:
        fig.update_xaxes(type="log")


ARTICLE_URL = "https://www.sciencedirect.com/science/article/pii/S0168165625002706"
CDLAB_URL = "https://www.tuwien.at/en/cdl/ibp4"
CDG_URL = "https://www.cdg.ac.at/en/research-units/labor/inclusion-body-processing-40"
BI_URL = "https://www.boehringer-ingelheim.com/"
IBD_URL = "https://www.tuwien.at/en/tch/icebe/ibdgroup"
TUWIEN_URL = "https://www.tuwien.at/en/"


def _render_credit_logos() -> None:
    """Clickable logo strip for the page footer.

    Raster logos (PNG/JPG/WebP) are embedded as base64 data-URI ``<img>`` tags.
    The Boehringer Ingelheim logo is an SVG and is inlined directly: an SVG
    loaded via a data-URI ``<img>`` can be blocked by the component iframe's
    content-security policy, whereas inlined ``<svg>`` markup renders reliably.
    """
    import base64
    import re
    import streamlit.components.v1 as components

    base = Path(__file__).parent / "assets"
    items = [
        ("ibd_logo.png", IBD_URL),
        ("tuwien_logo.webp", TUWIEN_URL),
        ("bi_logo.svg", BI_URL),
        ("cdg_logo.png", CDG_URL),
    ]
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    parts: List[str] = []
    for name, url in items:
        path = base / name
        if not path.exists():
            continue
        if path.suffix.lower() == ".svg":
            svg = re.sub(r"<\?xml[^>]*\?>", "", path.read_text(encoding="utf-8"), count=1).strip()
            svg = svg.replace("<svg ", '<svg style="height:54px;width:auto;" ', 1)
            inner = svg
        else:
            uri = "data:" + mime.get(path.suffix.lower(), "image/png") + ";base64,"
            uri += base64.b64encode(path.read_bytes()).decode()
            inner = f'<img src="{uri}" style="max-height:54px;max-width:180px;width:auto;">'
        parts.append(
            f'<a href="{url}" target="_blank" rel="noopener" '
            f'style="display:inline-flex;align-items:center;height:54px;margin:0 16px;">'
            f"{inner}</a>"
        )
    if parts:
        components.html(
            '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;">'
            + "".join(parts)
            + "</div>",
            height=72,
        )


def _render_footer() -> None:
    """Credits footer shown at the bottom of the analyzer page."""
    st.markdown("---")
    _render_credit_logos()
    st.caption(
        "A tool of the CD Laboratory for [Inclusion Body Processing 4.0](%s), "
        "developed in the [IBD Group — Integrated Bioprocess Development](%s) "
        "at [TU Wien](%s). Funded by the "
        "[Christian Doppler Gesellschaft](%s) and [Boehringer Ingelheim](%s)."
        % (CDLAB_URL, IBD_URL, TUWIEN_URL, CDG_URL, BI_URL)
    )


def _hero_svg() -> str:
    """Build the start-page hero illustration SVG (process + distribution).

    Top row: intact cells → homogeniser → inclusion bodies. Bottom: the
    resulting particle-size distribution with the two populations and a "lysis"
    arrow. Vivid colors, no fixed background — reads on light and dark themes.
    """
    import numpy as np

    width, height = 900, 390
    x0, x1 = 70.0, 850.0
    base, span = 340.0, 110.0  # distribution axis baseline / peak-height span

    def sx(mu: float) -> float:
        return x0 + (mu - 0.2) * (x1 - x0)

    def peak_path(amp: float, mu: float, sigma: float) -> str:
        xs = np.linspace(0.2, 1.2, 240)
        ys = amp * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        pts = [f"{sx(xv):.1f},{base - yv * span:.1f}" for xv, yv in zip(xs, ys)]
        return "M " + pts[0] + " L " + " L ".join(pts[1:]) + f" L {x1:.1f},{base:.1f} L {x0:.1f},{base:.1f} Z"

    # --- process row: intact cells -> homogeniser -> inclusion bodies ---
    cells = [(95, 80, 15), (120, 96, 13), (110, 64, 12), (84, 102, 11)]
    cell_svg = "".join(
        f'<circle cx="{cx}" cy="{cy}" r="{cr}" fill="rgba(34,197,94,0.22)" stroke="#22c55e" stroke-width="2"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{cr * 0.35:.1f}" fill="#16a34a"/>'
        for cx, cy, cr in cells
    )
    ib_dots = [
        (678, 72), (692, 90), (685, 58), (702, 98), (712, 72), (700, 108), (722, 86), (718, 62),
        (734, 96), (730, 70), (744, 84), (740, 58), (756, 98), (762, 74), (770, 90), (756, 64),
    ]
    ib_svg = "".join(f'<circle cx="{dx}" cy="{dy}" r="3.5" fill="#f97316"/>' for dx, dy in ib_dots)
    homogeniser = (
        '<rect x="288" y="76" width="14" height="12" fill="#94a3b8"/>'
        '<rect x="302" y="52" width="210" height="60" rx="10" fill="#eef2ff" stroke="#1f77b4" stroke-width="2"/>'
        '<rect x="512" y="76" width="14" height="12" fill="#94a3b8"/>'
        '<path d="M 352,60 L 408,82 L 352,104" fill="none" stroke="#1f77b4" stroke-width="2"/>'
        '<path d="M 462,60 L 406,82 L 462,104" fill="none" stroke="#1f77b4" stroke-width="2"/>'
        '<text x="407" y="132" font-size="13" font-weight="600" fill="#1f77b4" text-anchor="middle">homogeniser</text>'
    )
    flow = (
        '<line x1="160" y1="82" x2="284" y2="82" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow)"/>'
        '<line x1="530" y1="82" x2="664" y2="82" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow)"/>'
    )

    # --- particle-size distribution ---
    ticks = "".join(
        f'<line x1="{sx(t):.1f}" y1="{base}" x2="{sx(t):.1f}" y2="{base + 6}" stroke="#cbd5e1"/>'
        f'<text x="{sx(t):.1f}" y="{base + 22}" font-size="12" fill="#94a3b8" text-anchor="middle">{t:.1f}</text>'
        for t in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    )
    ib = peak_path(1.0, 0.48, 0.06)
    cell = peak_path(0.34, 0.85, 0.09)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:860px;display:block;margin:0 auto;font-family:system-ui,sans-serif;">'
        f'<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="#3b82f6"/></marker></defs>'
        f'{cell_svg}{flow}{homogeniser}{ib_svg}'
        f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="#94a3b8" stroke-width="1.5"/>{ticks}'
        f'<path d="{cell}" fill="rgba(34,197,94,0.20)" stroke="#22c55e" stroke-width="2"/>'
        f'<path d="{ib}" fill="rgba(249,115,22,0.20)" stroke="#f97316" stroke-width="2"/>'
        f'<text x="{sx(0.48):.1f}" y="{base - 1.0 * span - 10:.1f}" font-size="13" font-weight="600" '
        f'fill="#f97316" text-anchor="middle">Inclusion bodies</text>'
        f'<text x="{sx(0.85):.1f}" y="{base - 0.34 * span - 10:.1f}" font-size="13" font-weight="600" '
        f'fill="#22c55e" text-anchor="middle">Intact cells</text>'
        f'<path d="M {sx(0.80):.1f},{base - 0.20 * span:.1f} Q {sx(0.66):.1f},{base - 0.62 * span:.1f} '
        f'{sx(0.55):.1f},{base - 0.55 * span:.1f}" fill="none" stroke="#3b82f6" stroke-width="1.6" '
        f'stroke-dasharray="5 4" marker-end="url(#arrow)"/>'
        f'<text x="{sx(0.665):.1f}" y="{base - 0.70 * span:.1f}" font-size="12" fill="#3b82f6" '
        f'text-anchor="middle">lysis</text>'
        f'<text x="{(x0 + x1) / 2:.1f}" y="{base + 40}" font-size="12" fill="#94a3b8" '
        f'text-anchor="middle">particle size (µm)</text>'
        f"</svg>"
    )
    return svg


def _render_hero() -> None:
    """Render the start-page hero illustration."""
    import streamlit.components.v1 as components

    components.html(_hero_svg(), height=390)


@st.dialog("✨ What's new in LysoSense")
def whats_new_dialog() -> None:
    """Show release notes; newest version first. ``__version__``/``CHANGELOG``
    come from ``lysosense._version``, so this never needs manual syncing."""
    st.caption(f"LysoSense v{__version__}")
    for idx, (version, date, bullets) in enumerate(CHANGELOG):
        st.markdown(f"**v{version}** — {date}")
        st.markdown("\n".join(f"- {bullet}" for bullet in bullets))
        if idx != len(CHANGELOG) - 1:
            st.divider()


@st.cache_data(show_spinner=False)
def _cached_guide_html(guide_mtime: float) -> str:
    """Build (and cache) the standalone guide HTML.

    Keyed on ``guide_html.py``'s mtime so edits to the guide content invalidate
    the cache — without it, Streamlit caches on this function's own code (which
    doesn't change) and would keep serving a stale guide.
    """
    from guide_html import build_guide_html

    return build_guide_html()


def main() -> None:
    # Multipage entry point. ``set_page_config`` must be the first Streamlit
    # command; ``st.navigation`` makes each page a real served route, so the
    # Home landing page, the Analyzer, and the Guide each have a shareable URL —
    # which works the same locally and on a hosted deployment
    # (lysosense.streamlit.app), unlike a server-side ``webbrowser.open`` that
    # only works for local ``streamlit run``.
    st.set_page_config(page_title="LysoSense CPS Analyzer", page_icon="🔬", layout="wide")
    home = st.Page(home_page, title="Home", url_path="", default=True, icon="🏠")
    analyzer = st.Page(analyzer_page, title="Analyzer", url_path="analyzer", icon="🔬")
    guide = st.Page(guide_page, title="Guide", url_path="guide", icon="📖")
    # Stash the sibling Page objects so ``home_page`` can build call-to-action
    # links to the Analyzer and Guide. A function-based page can only be linked
    # via its ``st.Page`` object, which a page function can't otherwise receive.
    st.session_state["_nav_pages"] = {"analyzer": analyzer, "guide": guide}
    st.navigation([home, analyzer, guide]).run()


def guide_page() -> None:
    """Render the user guide page (prose + interactive example plots)."""
    import streamlit.components.v1 as components

    st.title("📖 LysoSense — User Guide")
    st.caption(
        "Tip: to read this alongside the analyzer, right-click the 'Guide' entry in "
        "the sidebar and choose 'Open link in new tab'."
    )
    guide_mtime = (Path(__file__).parent / "guide_html.py").stat().st_mtime
    with st.spinner("Building guide…"):
        html = _cached_guide_html(guide_mtime)
    # The guide bundles inline Plotly.js so it works offline; allow-scripts on the
    # component iframe lets the interactive plots render.
    components.html(html, height=1000, scrolling=True)


def home_page() -> None:
    """Landing page: what LysoSense does, the process illustration, and onward links."""
    st.title("LysoSense CPS Analyzer")
    _render_hero()
    st.markdown(
        "Differential centrifugal sedimentation (DCS) workflow for tracking intact cells and "
        "inclusion bodies during homogenisation (method adapted from [Klausser et al., 2025](%s)).\n\n"
        "LysoSense fits each trace to separate the intact-cell and inclusion-body populations and "
        "reports their relative abundances and **lysis efficiency** (the share of total signal outside "
        "the cell peak). Compare runs side by side, inspect every fit, and export the results to XLSX."
        % ARTICLE_URL
    )

    # Call-to-action links to the other pages. Function-based pages can only be
    # linked via their ``st.Page`` object, which ``main`` stashed in session state.
    pages = st.session_state.get("_nav_pages", {})
    col1, col2 = st.columns(2)
    if col1.button(
        "🔬 Open the Analyzer",
        type="primary",
        use_container_width=True,
        help="Upload .dat files and run the analysis.",
    ):
        st.switch_page(pages["analyzer"])
    col2.page_link(
        pages["guide"],
        label="📖 Read the Guide",
        use_container_width=True,
        help="A walkthrough with example plots — opens alongside the analyzer.",
    )

    st.caption(f"LysoSense v{__version__}")
    _render_footer()


def analyzer_page() -> None:
    """The analysis tool: sidebar controls, fitted plots, metrics, and downloads."""
    st.title("LysoSense CPS Analyzer")
    _render_analyzer_body()


def _render_analyzer_body() -> None:
    (
        options,
        show_fit,
        show_components,
        view_mode,
        compare_models,
        baseline_subtraction,
        baseline_method,
        normalize_data,
        limit_size_range,
        size_min_um,
        size_max_um,
        log_size_axis,
        peak_name_cell,
        peak_name_ib,
        uploaded_files,
    ) = _render_sidebar()

    if not uploaded_files:
        st.info("📁 Upload .dat files in the sidebar to begin the analysis.")
        return
    if limit_size_range and size_min_um >= size_max_um:
        st.error("The particle-size range is invalid: min size must be smaller than max size.")
        return

    results = _analyze_uploads(
        uploaded_files,
        options,
        normalize_data,
        limit_size_range,
        size_min_um,
        size_max_um,
    )
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

    _render_run_summary(active_results)

    # Create tab interface
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Overview",
            "🔍 Individual Samples",
            "📈 Results Table",
            "ℹ️ Detailed Information",
        ]
    )

    with tab1:
        _render_overview_tab(
            active_results, show_fit, show_components, view_mode, log_size_axis,
            peak_name_cell, peak_name_ib,
        )

    with tab2:
        _render_individual_samples_tab(
            active_results, show_fit, show_components, view_mode, log_size_axis,
            peak_name_cell, peak_name_ib,
        )

    with tab3:
        summary_df = _render_results_tab(
            active_results, peak_name_cell, peak_name_ib
        )

    with tab4:
        _render_details_tab(active_results, peak_name_cell, peak_name_ib)

    # Download buttons stay at bottom
    st.markdown("---")
    st.markdown("### Downloads")
    col1, col2 = st.columns(2)
    with col1:
        _render_download(summary_df)
    with col2:
        _render_experimental_data_download(
            active_results, peak_name_cell, peak_name_ib
        )


def _render_sidebar() -> Tuple[
    AnalysisOptions, bool, bool, str, bool, bool, str, bool, bool, float, float, bool, str, str, List[Any]
]:
    # Data upload section (always expanded)
    with st.sidebar.expander("📁 Data Upload", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload CPS/DCS .dat files",
            type=["dat"],
            accept_multiple_files=True,
            help="Drop multiple runs at once to compare peak areas and lysis efficiency.",
            key="file_uploader",
        )

    # Only show other sections if files are uploaded
    if uploaded_files:
        # Peaks & sample setup — the inputs that most directly determine lysis
        # (peak labels, target sizes, and the analysis window), so they sit first.
        with st.sidebar.expander("🎯 Peaks & Sample", expanded=True):
            st.markdown("**Peak labels**")
            peak_name_ib = st.text_input(
                "IB peak name",
                value="IBs",
                key="peak_name_ib",
                help="Display label for the inclusion-body peak (smaller). Shown in plots, the results table, and XLSX exports.",
            )
            peak_name_cell = st.text_input(
                "Cell peak name",
                value="Cells",
                key="peak_name_cell",
                help="Display label for the cell peak (larger). Lysis% is always reported for this peak.",
            )

            st.markdown("**Peak targets**")
            mu_ib = st.number_input(
                "IB target size (µm)",
                value=0.48,
                min_value=0.1,
                max_value=2.0,
                step=0.01,
                key="mu_ib",
            )
            mu_cell = st.number_input(
                "Cell target size (µm)",
                value=0.85,
                min_value=0.1,
                max_value=3.0,
                step=0.01,
                key="mu_cell",
            )

            st.markdown("**Particle-size window**")
            limit_size_range = st.checkbox(
                "Limit particle-size range for fitting",
                value=True,
                help="Restrict analysis to a selected particle-size window. Leave off to use the full uploaded CPS/DCS trace.",
                key="limit_size_range",
            )
            range_col1, range_col2 = st.columns(2)
            with range_col1:
                size_min_um = st.number_input(
                    "Min size (µm)",
                    value=0.2,
                    min_value=0.0,
                    max_value=50.0,
                    step=0.1,
                    disabled=not limit_size_range,
                    key="size_min_um",
                )
            with range_col2:
                size_max_um = st.number_input(
                    "Max size (µm)",
                    value=1.2,
                    min_value=0.1,
                    max_value=50.0,
                    step=0.1,
                    disabled=not limit_size_range,
                    key="size_max_um",
                )

        # Data preprocessing (rarely changed — collapsed by default)
        with st.sidebar.expander("🔧 Data Preprocessing", expanded=False):
            baseline_subtraction = st.checkbox(
                "Baseline subtraction",
                value=False,
                help="Subtract baseline from raw data before fitting. Usually not necessary.",
                key="baseline_subtraction",
            )

            baseline_method = st.selectbox(
                "Baseline method",
                ("minimum", "percentile", "linear"),
                help="• Minimum: Use minimum signal value\n• Percentile: Use 1st percentile\n• Linear: Linear fit to edges",
                disabled=not baseline_subtraction,
                key="baseline_method",
            )

            st.markdown("---")  # Separator

            normalize_data = st.checkbox(
                "Normalize data",
                value=False,
                help="Normalize data to enable comparison between samples with different concentrations",
                key="normalize_data",
            )

            if normalize_data:
                st.markdown(
                    "**Method**: Max intensity normalization (scales to maximum signal value)"
                )

        # Fitting — model shape, width handling, and detection mode
        with st.sidebar.expander("⚙️ Fitting", expanded=True):
            model_options = (
                "gaussian",
                "lognormal",
                "splitgaussian",
                "gennormal",
                "autofit",
            )
            default_model = st.session_state.get("model", "autofit")
            default_index = (
                model_options.index(default_model)
                if default_model in model_options
                else 0
            )
            model = st.radio(
                "Peak model",
                model_options,
                index=default_index,
                key="model",
            )
            compare_models = model == "autofit"

            st.checkbox(
                "Relax peak-width constraints (broad / overlapping peaks)",
                value=False,
                key="relax_widths",
                help=(
                    "Off (default): use the standard tight peak-width bounds. "
                    "Turn this on for instruments/data where peaks sit broader than "
                    "the defaults expect (typical symptom: the fit is sharper/narrower "
                    "than the data and R² is poor). When on, a tight fit is tried first "
                    "and the widths are only relaxed if that fit scores below R²=0.92, "
                    "so clean traces are left unchanged."
                ),
            )

            st.markdown("**Peak detection**")
            detection_modes = (
                "Automatic",
                "Resolved peaks only",
                "Allow overlapping peaks",
                "Single peak only",
            )
            default_detection_mode = st.session_state.get(
                "peak_detection_mode", "Automatic"
            )
            default_detection_index = (
                detection_modes.index(default_detection_mode)
                if default_detection_mode in detection_modes
                else 0
            )
            peak_detection_mode = st.radio(
                "Peak detection mode",
                detection_modes,
                index=default_detection_index,
                help=(
                    "Automatic tries resolved peaks first and overlap deconvolution only if needed. "
                    "Resolved peaks only is stricter. Single peak only disables two-component fits."
                ),
                key="peak_detection_mode",
            )

            with st.expander("Advanced fitting settings", expanded=False):
                sensitivity = st.select_slider(
                    "Sensitivity",
                    options=[
                        "Low (strict)",
                        "Medium (default)",
                        "High (sensitive)",
                        "Custom",
                    ],
                    value="Medium (default)",
                    key="sensitivity",
                    help=(
                        "Low = fewer false positives, High = catch more 2-peaks. "
                        "Select 'Custom' to adjust individual parameters."
                    ),
                )

                if sensitivity == "Custom":
                    st.markdown("**Resolved peak gates**")
                    st.slider(
                        "Residual prominence (× noise σ)",
                        min_value=1.0,
                        max_value=6.0,
                        value=3.0,
                        step=0.5,
                        help="Minimum prominence of residual peak candidate (higher = stricter)",
                        key="residual_prominence",
                    )
                    st.slider(
                        "Min residual distance (µm)",
                        min_value=0.05,
                        max_value=0.30,
                        value=0.15,
                        step=0.01,
                        help="Minimum distance from main peak for residual candidate",
                        key="residual_distance",
                    )
                    st.slider(
                        "Min residual area (%)",
                        min_value=1.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.5,
                        help="Minimum residual area as fraction of total signal",
                        key="residual_area",
                    )
                    st.slider(
                        "BIC improvement threshold",
                        min_value=-20.0,
                        max_value=-2.0,
                        value=-10.0,
                        step=1.0,
                        help="2-peak model must improve BIC by this much (more negative = stricter)",
                        key="bic_threshold",
                    )
                    st.slider(
                        "Local dominance (%)",
                        min_value=20.0,
                        max_value=60.0,
                        value=40.0,
                        step=5.0,
                        help="Second peak must dominate this much somewhere locally",
                        key="local_dominance",
                    )
                    st.slider(
                        "Min 2nd peak area (%)",
                        min_value=1.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.5,
                        help="Minimum area fraction for second peak",
                        key="second_area",
                    )
                    st.slider(
                        "Min separation (× avg FWHM)",
                        min_value=0.3,
                        max_value=1.5,
                        value=0.8,
                        step=0.1,
                        help="Peak separation relative to average FWHM (higher = stricter)",
                        key="separation_ratio",
                    )
                    st.slider(
                        "Max Cell peak FWHM (µm)",
                        min_value=0.08,
                        max_value=0.30,
                        value=0.25,
                        step=0.01,
                        help="Maximum FWHM for the Cell peak during fitting.",
                        key="max_fwhm_second",
                    )
                    st.slider(
                        "Min compactness (area/FWHM)",
                        min_value=0.0,
                        max_value=30.0,
                        value=0.0,
                        step=1.0,
                        help="Post-fit check: minimum compactness for second peak. 0 = disabled.",
                        key="min_compactness",
                    )
                    st.slider(
                        "Min prominence (× noise σ)",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.0,
                        step=0.5,
                        help="Post-fit check: minimum prominence above the main peak shoulder. 0 = disabled.",
                        key="min_prominence_sigma",
                    )

                st.markdown("**Overlap deconvolution**")
                st.slider(
                    "Cell center shift (%)",
                    min_value=5,
                    max_value=25,
                    value=12,
                    step=1,
                    help="Allowed cell peak center shift around the configured cell target.",
                    key="overlap_cell_shift",
                )
                st.slider(
                    "Max overlap IB FWHM (µm)",
                    min_value=0.15,
                    max_value=0.50,
                    value=0.35,
                    step=0.01,
                    help="Maximum IB peak width in overlap deconvolution.",
                    key="overlap_max_ib_fwhm",
                )
                st.slider(
                    "Max overlap cell FWHM (µm)",
                    min_value=0.10,
                    max_value=0.45,
                    value=0.30,
                    step=0.01,
                    help="Maximum cell peak width in overlap deconvolution.",
                    key="overlap_max_cell_fwhm",
                )
                st.slider(
                    "Min overlap cell area (%)",
                    min_value=1.0,
                    max_value=15.0,
                    value=3.0,
                    step=0.5,
                    help="Minimum fitted cell area needed to accept an overlap deconvolution.",
                    key="overlap_min_area",
                )

                st.markdown("**Model per peak**")
                use_mixed_models = st.checkbox(
                    "Use different models per peak",
                    value=st.session_state.get("use_mixed_models", False),
                    key="use_mixed_models",
                    disabled=compare_models,
                    help="Fit IB and cell peaks with different model types (autofit does this automatically)",
                )

                if use_mixed_models and not compare_models:
                    single_model_options = (
                        "gaussian",
                        "lognormal",
                        "splitgaussian",
                        "gennormal",
                    )
                    st.selectbox(
                        "IB peak model",
                        single_model_options,
                        index=single_model_options.index(model)
                        if model in single_model_options
                        else 0,
                        key="model_ib",
                    )
                    st.selectbox(
                        "Cell peak model",
                        single_model_options,
                        index=single_model_options.index(model)
                        if model in single_model_options
                        else 0,
                        key="model_cell",
                    )

                st.markdown("**Fitting constraints**")
                allow_shift = st.slider(
                    "Allowed peak shift (%)",
                    min_value=5,
                    max_value=40,
                    value=20,
                    step=1,
                    key="allow_shift",
                )
                second_peak_percent = (
                    st.slider(
                        "Min 2nd peak fraction (%)",
                        min_value=0.0,
                        max_value=8.0,
                        value=2.0,
                        step=0.5,
                        help="Minimum share of total area required to keep the cell peak.",
                        key="second_peak",
                    )
                    / 100.0
                )
                limit_peak_width = st.checkbox(
                    "Limit max peak width",
                    value=True,
                    help="Apply a full-width-at-half-maximum (FWHM) cap to both peaks to avoid overly broad fits.",
                    key="limit_peak_width",
                )
                if limit_peak_width:
                    max_peak_width_value = st.slider(
                        "Max peak width (um)",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.3,
                        step=0.01,
                        key="max_peak_width",
                    )
                else:
                    max_peak_width_value = None

                fit_weight_power = st.slider(
                    "Peak-top weighting",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Give higher-signal points more influence during fitting. 0 = ordinary least squares.",
                    key="fit_weight_power",
                )

        # Visualization section (merged with display options)
        with st.sidebar.expander("📊 Visualization", expanded=True):
            view_mode = st.radio(
                "View mode",
                ("Combined", "Fit Overview", "Raw Data Only"),
                help="• Combined: Raw data + fits + components\n• Fit Overview: Only fitted components\n• Raw Data Only: Just raw measurements",
                key="view_mode",
            )

            st.markdown("**Display Options**")
            show_fit = st.checkbox("Show fitted envelope", value=True, key="show_fit")
            show_components = st.checkbox(
                "Show component contributions", value=True, key="show_components"
            )
            log_size_axis = st.checkbox(
                "Log particle-size axis",
                value=False,
                key="log_size_axis",
                help=(
                    "Display only — switch the size axis to logarithmic. "
                    "The fit and lysis% are computed in linear space and are unaffected."
                ),
            )

        # Quick actions section
        with st.sidebar.expander("⚡ Quick Actions", expanded=False):
            if st.button("🔄 Reset All", help="Reset all settings to defaults"):
                # Clear all widget state to reset to defaults
                for key in list(st.session_state.keys()):
                    if isinstance(key, str) and key.startswith(
                        (
                            "view_mode",
                            "model",
                            "autofit",
                            "mu_ib",
                            "mu_cell",
                            "peak_name_ib",
                            "peak_name_cell",
                            "allow_shift",
                            "second_peak",
                            "limit_peak_width",
                            "max_peak_width",
                            "fit_weight_power",
                            "show_fit",
                            "show_components",
                            "log_size_axis",
                            "baseline_subtraction",
                            "baseline_method",
                            "limit_size_range",
                            "size_min_um",
                            "size_max_um",
                            "peak_detection_mode",
                            "sensitivity",
                            "residual_prominence",
                            "residual_distance",
                            "residual_area",
                            "bic_threshold",
                            "local_dominance",
                            "second_area",
                            "separation_ratio",
                            "max_fwhm_second",
                            "min_compactness",
                            "min_prominence_sigma",
                            "overlap_cell_shift",
                            "overlap_max_ib_fwhm",
                            "overlap_max_cell_fwhm",
                            "overlap_min_area",
                        )
                    ):
                        del st.session_state[key]
                st.rerun()
    else:
        # Return defaults when no files uploaded
        model = "autofit"
        compare_models = True
        mu_ib = 0.48
        mu_cell = 0.85
        peak_name_ib = "IBs"
        peak_name_cell = "Cells"
        allow_shift = 20
        second_peak_percent = 0.02
        limit_peak_width = True
        max_peak_width_value = 0.3
        fit_weight_power = 0.2
        show_fit = True
        show_components = True
        log_size_axis = False
        baseline_subtraction = False
        baseline_method = "minimum"
        view_mode = "Combined"
        normalize_data = False
        limit_size_range = True
        size_min_um = 0.2
        size_max_um = 1.2
        peak_detection_mode = "Automatic"
        sensitivity = "Medium (default)"
        # Second peak quality defaults
        residual_prominence = 3.0
        residual_distance = 0.15
        residual_area = 5.0
        bic_threshold = -10.0
        local_dominance = 40.0
        second_area = 3.0
        separation_ratio = 0.8
        max_fwhm_second = 0.25
        min_compactness = 0.0
        min_prominence_sigma = 0.0

    # Don't create AnalysisOptions here anymore since it depends on the model choice
    # Create a placeholder with default values that will be overridden in analysis
    peak_width_cap = None
    if limit_peak_width and max_peak_width_value is not None:
        peak_width_cap = safe_float(max_peak_width_value)
        if peak_width_cap == 0.0:  # safe_float returns 0.0 on error by default
            peak_width_cap = None

    peak_detection_mode = st.session_state.get(
        "peak_detection_mode", peak_detection_mode
    )
    force_single_peak = peak_detection_mode == "Single peak only"
    use_overlap_deconvolution = peak_detection_mode in (
        "Automatic",
        "Allow overlapping peaks",
    )
    overlap_cell_shift = st.session_state.get("overlap_cell_shift", 12)
    overlap_max_ib_fwhm = st.session_state.get("overlap_max_ib_fwhm", 0.35)
    overlap_max_cell_fwhm = st.session_state.get("overlap_max_cell_fwhm", 0.30)
    overlap_min_area = st.session_state.get("overlap_min_area", 3.0)

    # Sensitivity presets
    sensitivity_presets = {
        "Low (strict)": {
            "residual_prominence": 4.0,
            "residual_distance": 0.20,
            "residual_area": 8.0,
            "bic_threshold": -15.0,
            "local_dominance": 50.0,
            "second_area": 8.0,
            "separation_ratio": 1.0,
            "max_fwhm_second": 0.15,
            "min_compactness": 5.0,
            "min_prominence_sigma": 2.0,
        },
        "Medium (default)": {
            "residual_prominence": 3.0,
            "residual_distance": 0.15,
            "residual_area": 5.0,
            "bic_threshold": -10.0,
            "local_dominance": 40.0,
            "second_area": 3.0,
            "separation_ratio": 0.8,
            "max_fwhm_second": 0.25,
            "min_compactness": 0.0,
            "min_prominence_sigma": 0.0,
        },
        "High (sensitive)": {
            "residual_prominence": 2.0,
            "residual_distance": 0.10,
            "residual_area": 3.0,
            "bic_threshold": -5.0,
            "local_dominance": 30.0,
            "second_area": 3.0,
            "separation_ratio": 0.5,
            "max_fwhm_second": 0.30,
            "min_compactness": 0.0,
            "min_prominence_sigma": 0.0,
        },
    }

    sensitivity = st.session_state.get("sensitivity", "Medium (default)")
    if sensitivity in sensitivity_presets:
        # Use preset values
        preset = sensitivity_presets[sensitivity]
        residual_prominence = preset["residual_prominence"]
        residual_distance = preset["residual_distance"]
        residual_area = preset["residual_area"]
        bic_threshold = preset["bic_threshold"]
        local_dominance = preset["local_dominance"]
        second_area = preset["second_area"]
        separation_ratio = preset["separation_ratio"]
        max_fwhm_second = preset["max_fwhm_second"]
        min_compactness = preset["min_compactness"]
        min_prominence_sigma = preset["min_prominence_sigma"]
    else:
        # Use custom values from session state
        residual_prominence = st.session_state.get("residual_prominence", 3.0)
        residual_distance = st.session_state.get("residual_distance", 0.15)
        residual_area = st.session_state.get("residual_area", 5.0)
        bic_threshold = st.session_state.get("bic_threshold", -10.0)
        local_dominance = st.session_state.get("local_dominance", 40.0)
        second_area = st.session_state.get("second_area", 3.0)
        separation_ratio = st.session_state.get("separation_ratio", 0.8)
        max_fwhm_second = st.session_state.get("max_fwhm_second", 0.25)
        min_compactness = st.session_state.get("min_compactness", 0.0)
        min_prominence_sigma = st.session_state.get("min_prominence_sigma", 0.0)

    options = AnalysisOptions(
        model="gaussian",  # placeholder, will be overridden
        mu_ib_um=safe_float(mu_ib, 0.48),
        mu_cell_um=safe_float(mu_cell, 0.85),
        allow_shift_fraction=safe_float(allow_shift, 20.0) / 100.0,
        second_peak_min_frac=safe_float(second_peak_percent, 0.02),
        max_peak_fwhm_um=peak_width_cap,
        fit_weight_power=safe_float(fit_weight_power, 0.2),
        force_single_peak=force_single_peak,
        use_gated_two_peak=True,
        residual_prominence_sigma=safe_float(residual_prominence, 3.0),
        residual_min_distance_um=safe_float(residual_distance, 0.15),
        residual_min_area_frac=safe_float(residual_area, 5.0) / 100.0,
        bic_improvement_threshold=safe_float(bic_threshold, -10.0),
        local_dominance_threshold=safe_float(local_dominance, 40.0) / 100.0,
        second_peak_area_threshold=safe_float(second_area, 3.0) / 100.0,
        min_separation_fwhm_ratio=safe_float(separation_ratio, 0.8),
        # Second peak quality constraints
        max_fwhm_second_peak_um=safe_float(max_fwhm_second, 0.25),
        min_compactness_second_peak=safe_float(min_compactness, 0.0),
        min_prominence_second_peak_sigma=safe_float(min_prominence_sigma, 0.0),
        use_overlap_deconvolution=bool(use_overlap_deconvolution),
        overlap_cell_shift_fraction=safe_float(overlap_cell_shift, 12.0) / 100.0,
        overlap_max_ib_fwhm_um=safe_float(overlap_max_ib_fwhm, 0.35),
        overlap_max_cell_fwhm_um=safe_float(overlap_max_cell_fwhm, 0.30),
        overlap_min_area_frac=safe_float(overlap_min_area, 3.0) / 100.0,
        relax_peak_widths=bool(st.session_state.get("relax_widths", False)),
    )
    # Version footer + release notes — always shown, independent of uploads.
    st.sidebar.markdown("---")
    if st.sidebar.button(
        "✨ What's new?",
        key="whats_new",
        help="See recent changes and the current app version.",
    ):
        whats_new_dialog()

    return (
        options,
        show_fit,
        show_components,
        view_mode,
        compare_models,
        baseline_subtraction,
        baseline_method,
        normalize_data,
        bool(limit_size_range),
        safe_float(size_min_um, 0.2) or 0.2,
        safe_float(size_max_um, 1.2) or 1.2,
        log_size_axis,
        peak_name_cell,
        peak_name_ib,
        uploaded_files,
    )


def _analyze_uploads(
    uploaded_files: Sequence[Any],
    options: AnalysisOptions,
    normalize_data: bool,
    limit_size_range: bool,
    size_min_um: float,
    size_max_um: float,
) -> List[Tuple[str, AnalysisResult]]:
    results: List[Tuple[str, AnalysisResult]] = []
    selected_model: str = str(st.session_state.get("model", "autofit"))
    baseline_subtraction = st.session_state.get("baseline_subtraction", False)
    baseline_method = st.session_state.get("baseline_method", "minimum")
    use_mixed = st.session_state.get("use_mixed_models", False)
    model_ib_val = st.session_state.get("model_ib")
    model_cell_val = st.session_state.get("model_cell")
    options_key = _analysis_options_cache_key(options)

    for file in uploaded_files:
        try:
            analysis, warning = _analyze_one_upload_cached(
                file.getvalue(),
                file.name,
                options_key,
                selected_model,
                bool(baseline_subtraction),
                str(baseline_method),
                bool(normalize_data),
                bool(limit_size_range),
                float(size_min_um),
                float(size_max_um),
                bool(use_mixed),
                model_ib_val,
                model_cell_val,
            )
        except Exception:
            # ``st.cache_data`` serializes the return value, which can fail for
            # reasons unrelated to the fit (class-identity drift after a module
            # reload, or a value the cache simply can't pickle). Fall back to an
            # uncached run so the user still gets a result instead of a per-file
            # error. A genuine fit failure re-raises here and is reported below.
            try:
                analysis, warning = _analyze_one_upload(
                    file.getvalue(),
                    file.name,
                    options,
                    selected_model,
                    bool(baseline_subtraction),
                    str(baseline_method),
                    bool(normalize_data),
                    bool(limit_size_range),
                    float(size_min_um),
                    float(size_max_um),
                    bool(use_mixed),
                    model_ib_val,
                    model_cell_val,
                )
            except Exception as exc:
                st.error(f"{file.name}: {exc}")
                continue
        if warning:
            st.warning(warning)
        results.append((file.name, analysis))
    return results


AnalysisOptionsKey = Tuple[Tuple[str, Any], ...]


def _analysis_options_cache_key(options: AnalysisOptions) -> AnalysisOptionsKey:
    return tuple((field.name, getattr(options, field.name)) for field in fields(options))


def _analysis_options_from_cache_key(key: AnalysisOptionsKey) -> AnalysisOptions:
    return AnalysisOptions(**dict(key))


def _analyze_one_upload(
    file_bytes: bytes,
    source_name: str,
    options: AnalysisOptions,
    selected_model: str,
    baseline_subtraction: bool,
    baseline_method: str,
    normalize_data: bool,
    limit_size_range: bool,
    size_min_um: float,
    size_max_um: float,
    use_mixed: bool,
    model_ib_val: Optional[str],
    model_cell_val: Optional[str],
) -> Tuple[AnalysisResult, Optional[str]]:
    """Run one file through the full pipeline (parse -> preprocess -> fit).

    Pure worker with no ``@st.cache_data`` so callers can use it directly when the
    cache must be bypassed (see ``_analyze_uploads``'s fallback).
    """
    measurement = parse_dat_bytes(file_bytes, source_name=source_name)
    if limit_size_range and size_min_um < size_max_um:
        measurement = clip_measurement_range(measurement, size_min_um, size_max_um)
    if baseline_subtraction:
        measurement = subtract_baseline(measurement, baseline_method)

    warning: Optional[str] = None
    if normalize_data:
        try:
            measurement = normalize_measurement(measurement)
        except NormalizationSkipped as exc:
            warning = str(exc)

    analysis = _fit_measurement_from_ui_options(
        measurement,
        options,
        selected_model,
        use_mixed,
        model_ib_val,
        model_cell_val,
    )
    return analysis, warning


@st.cache_data(show_spinner=False)
def _analyze_one_upload_cached(
    file_bytes: bytes,
    source_name: str,
    options_key: AnalysisOptionsKey,
    selected_model: str,
    baseline_subtraction: bool,
    baseline_method: str,
    normalize_data: bool,
    limit_size_range: bool,
    size_min_um: float,
    size_max_um: float,
    use_mixed: bool,
    model_ib_val: Optional[str],
    model_cell_val: Optional[str],
) -> Tuple[AnalysisResult, Optional[str]]:
    options = _analysis_options_from_cache_key(options_key)
    return _analyze_one_upload(
        file_bytes,
        source_name,
        options,
        selected_model,
        baseline_subtraction,
        baseline_method,
        normalize_data,
        limit_size_range,
        size_min_um,
        size_max_um,
        use_mixed,
        model_ib_val,
        model_cell_val,
    )


def _fit_measurement_from_ui_options(
    measurement: Any,
    options: AnalysisOptions,
    selected_model: str,
    use_mixed: bool,
    model_ib_val: Optional[str],
    model_cell_val: Optional[str],
) -> AnalysisResult:
    if selected_model == "autofit":
        return _autofit_measurement(measurement, options)

    actual_model = "gaussian" if selected_model == "autofit" else selected_model
    if use_mixed and model_ib_val and model_cell_val:
        actual_options = replace(
            options,
            model=actual_model,  # type: ignore[arg-type]
            model_ib=model_ib_val,  # type: ignore[arg-type]
            model_cell=model_cell_val,  # type: ignore[arg-type]
        )
    else:
        actual_options = replace(
            options,
            model=actual_model,  # type: ignore[arg-type]
            model_ib=None,
            model_cell=None,
        )
    return analyze_measurement(measurement, actual_options)


def _autofit_measurement(
    measurement: Any, options: AnalysisOptions
) -> AnalysisResult:
    """Score every model combination (4 IB x 4 cell = 16) and keep the best by R2.

    Optimized for speed without changing the result:
      * peak hints and the 8 single-peak candidate fits are computed once and
        reused across the grid (they are model-independent),
      * each combo is scored from the cheap fit-only snapshot -- the dense frame,
        full metrics and shoulder diagnostic are built only for the winner,
      * the 16 combos run in a thread pool (scipy's curve_fit releases the GIL).

    Selection is an order-dependent R2 tie-break fold, and the pool preserves
    submission order, so the chosen winner is identical to a sequential grid.
    """
    df = measurement.data
    x = df["particle_size_um"].to_numpy(dtype=float)
    y = df["mass_signal_ug"].to_numpy(dtype=float)
    shared = _build_precomputed_hints(x, y, options)

    def _run_combo(
        combo: Any,
    ) -> Tuple[str, str, Optional[_FitSnapshot]]:
        model_ib, model_cell = combo
        try:
            opts_i = replace(
                options,
                model="gaussian",
                model_ib=model_ib,  # type: ignore[arg-type]
                model_cell=model_cell,  # type: ignore[arg-type]
            )
            snapshot = _analyze_fit_only(measurement, opts_i, precomputed=shared)
        except Exception:
            return (model_ib, model_cell, None)
        return (model_ib, model_cell, snapshot)

    combos = [(mi, mc) for mi in _ALL_MODELS for mc in _ALL_MODELS]
    scored = _parallel_map(_run_combo, combos)

    best_r2 = -float("inf")
    best_residual_score = float("inf")
    best_snapshot: Optional[_FitSnapshot] = None
    r2_tie_tolerance = 5e-4

    for model_ib, model_cell, snapshot in scored:
        if snapshot is None:
            continue
        fit_kind = snapshot.fitres["kind"]
        if fit_kind in ("two", "overlap"):
            intact_fraction = safe_float(snapshot.intact_fraction, 0.0)
            if fit_kind != "overlap" and model_ib == "gennormal":
                continue
            if (
                fit_kind != "overlap"
                and model_cell == "gennormal"
                and intact_fraction < 0.15
            ):
                continue
        # Minimal result carrying just what the selection helpers read (the
        # observed frame); dense_fit/metrics are placeholders, not used for ranking.
        result = AnalysisResult(
            measurement=measurement,
            observed=snapshot.observed,
            dense_fit=pd.DataFrame(),
            metrics={"intact_fraction": snapshot.intact_fraction},
            fit_kind=fit_kind,
            options=snapshot.opts,
        )
        r2 = calculate_r_squared(result)
        residual_score = _fit_residual_score(result)
        if r2 > best_r2 + r2_tie_tolerance or (
            abs(r2 - best_r2) <= r2_tie_tolerance
            and residual_score < best_residual_score
        ):
            best_r2 = r2
            best_residual_score = residual_score
            best_snapshot = snapshot

    if best_snapshot is None:
        raise RuntimeError("All autofit attempts failed")
    return _finalize(best_snapshot)


def _fit_residual_score(result: AnalysisResult) -> float:
    """Residual tie-break score for near-identical autofit R² values."""
    observed = result.observed
    residual = observed["mass_signal_ug"] - observed["fit_signal_ug"]
    peak_height = max(safe_float(observed["mass_signal_ug"].max()), 1e-12)
    max_abs = safe_float(residual.abs().max())
    mean_abs = safe_float(residual.abs().mean())
    return (max_abs / peak_height) + 0.25 * (mean_abs / peak_height)


def _cell_component_label(
    analysis: AnalysisResult, peak_name_cells: str = "Cells"
) -> str:
    if analysis.fit_kind == "overlap":
        return f"{peak_name_cells} (overlap fit)"
    return peak_name_cells


def _render_run_summary(entries: Sequence[Tuple[str, AnalysisResult]]) -> None:
    """Show a compact status band for the selected analysis run."""
    if not entries:
        return

    r_squared_values = [calculate_r_squared(analysis) for _, analysis in entries]
    lysis_values = [
        safe_float(analysis.metrics.get("lysis_efficiency"), 0.0)
        for _, analysis in entries
    ]
    two_peak_count = sum(1 for _, analysis in entries if analysis.fit_kind == "two")
    overlap_count = sum(
        1 for _, analysis in entries if analysis.fit_kind == "overlap"
    )
    low_quality_count = sum(1 for value in r_squared_values if value < 0.90)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected traces", len(entries))
    with col2:
        st.metric(
            "2-peak fits",
            f"{two_peak_count + overlap_count}/{len(entries)}",
            delta=f"{overlap_count} overlap" if overlap_count else None,
        )
    with col3:
        mean_lysis = sum(lysis_values) / max(len(lysis_values), 1)
        st.metric("Mean lysis efficiency", f"{mean_lysis:.1%}")
    with col4:
        mean_r2 = sum(r_squared_values) / max(len(r_squared_values), 1)
        st.metric("Mean R²", f"{mean_r2:.4f}")

    if low_quality_count:
        st.warning(
            f"{low_quality_count} selected trace(s) have R² below 0.90. "
            "Review the individual sample plots before interpreting those results."
        )


def _render_raw_data_plot(
    entries: Sequence[Tuple[str, AnalysisResult]], log_size_axis: bool = False
) -> None:
    """Plot only raw data traces with distinct colors per sample."""
    fig = go.Figure()

    # Color palette for samples
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, (label, analysis) in enumerate(entries):
        color = colors[i % len(colors)]
        observed = analysis.observed

        # Extract sample name (remove file extension for cleaner legend)
        sample_name = label.replace(".dat", "")

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
        yaxis_title=_signal_yaxis_title(entries),
        legend_title="Sample",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
    )
    _apply_size_axis_scale(fig, log_size_axis)
    # Check if any samples are normalized
    normalized_samples = [
        label
        for label, analysis in entries
        if analysis.measurement.metadata.get("normalized", False)
    ]

    title = "Raw Particle Size Distributions"
    if normalized_samples:
        title += " (Normalized Data)"

    st.subheader(title)
    st.plotly_chart(fig, width="stretch")


def _render_fit_overview(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    log_size_axis: bool = False,
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> None:
    """Organized plot showing only fitted data with sample-specific colors and grouped legends."""
    fig = go.Figure()

    # Color palette for samples (same as raw data view for consistency)
    sample_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, (label, analysis) in enumerate(entries):
        color = sample_colors[i % len(sample_colors)]
        sample_name = label.replace(".dat", "")
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
                        name=_cell_component_label(analysis, peak_name_cell),
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
                    name=peak_name_ib,
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=group_name,
                    legendgrouptitle_text=sample_name,
                )
            )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title=_signal_yaxis_title(entries),
        legend_title="Samples & Components",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggleothers",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    _apply_size_axis_scale(fig, log_size_axis)

    # Add legend guide
    st.markdown(
        f"**Legend Guide:** Click sample names to toggle all traces • Click individual traces to toggle • Line styles: solid={peak_name_cell.lower()}, dashed=fit, dotted={peak_name_ib.lower()}"
    )

    # Check if any samples are normalized
    normalized_samples = [
        label
        for label, analysis in entries
        if analysis.measurement.metadata.get("normalized", False)
    ]

    title = "Fit Components Overview"
    if normalized_samples:
        title += " (Normalized Data)"

    st.subheader(title)
    st.plotly_chart(fig, width="stretch")


def _render_plot(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    log_size_axis: bool = False,
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> None:
    """Combined plot with grouped legends for better organization."""
    fig = go.Figure()

    # Color palette for samples
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, (label, analysis) in enumerate(entries):
        color = colors[i % len(colors)]
        sample_name = label.replace(".dat", "")
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
                        name=_cell_component_label(analysis, peak_name_cell),
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
                    name=peak_name_ib,
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    legendgroup=group_name,
                    legendgrouptitle_text=sample_name,
                )
            )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title=_signal_yaxis_title(entries),
        legend_title="Samples & Trace Types",
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
        legend=dict(
            groupclick="toggleitem",
            itemclick="toggleothers",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    _apply_size_axis_scale(fig, log_size_axis)

    # Add legend guide
    st.markdown(
        f"**Legend Guide:** Click sample names to toggle all traces • Line styles: solid=raw/{peak_name_cell.lower()}, dashed=fit, dotted={peak_name_ib.lower()}"
    )

    # Check if any samples are normalized
    normalized_samples = [
        label
        for label, analysis in entries
        if analysis.measurement.metadata.get("normalized", False)
    ]

    title = "Combined Particle Size Distribution"
    if normalized_samples:
        title += " (Normalized Data)"

    st.subheader(title)
    st.plotly_chart(fig, width="stretch")


def _render_metrics(
    entries: Sequence[Tuple[str, AnalysisResult]],
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> pd.DataFrame:
    records: List[Dict[str, float | str | None]] = []
    for label, analysis in entries:
        row: Dict[str, float | str | None] = {"measurement": label}
        row.update(analysis.metrics)  # type: ignore[arg-type]

        # Add fit quality metrics
        r_squared = calculate_r_squared(analysis)
        row["r_squared"] = r_squared
        row["fit_quality"] = _get_fit_quality_label(r_squared)

        # Add baseline information
        baseline_subtracted = analysis.measurement.metadata.get(
            "baseline_subtracted", False
        )
        if baseline_subtracted:
            baseline_method = analysis.measurement.metadata.get(
                "baseline_method", "unknown"
            )
            row["baseline_corrected"] = f"Yes ({baseline_method})"
        else:
            row["baseline_corrected"] = "No"

        # Add normalization information
        normalized = analysis.measurement.metadata.get("normalized", False)
        if normalized:
            norm_method = analysis.measurement.metadata.get(
                "normalization_method", "unknown"
            )
            norm_factor = analysis.measurement.metadata.get("normalization_factor", 1.0)
            row["normalized"] = f"Yes ({norm_method}, {norm_factor:.2e})"
        else:
            row["normalized"] = "No"

        records.append(row)

    summary = pd.DataFrame(records).set_index("measurement")

    # Surface the user-supplied peak names in the display + export headers.
    # Internal metric keys are left untouched (they are asserted in the test
    # suite); only this display/export frame is renamed.
    mean_cell_col = f"mean {peak_name_cell} (µm)"
    mean_ib_col = f"mean {peak_name_ib} (µm)"
    fwhm_cell_col = f"fwhm {peak_name_cell} (µm)"
    fwhm_ib_col = f"fwhm {peak_name_ib} (µm)"
    area_cell_col = f"area ({peak_name_cell})"
    area_ib_col = f"area ({peak_name_ib})"
    summary = summary.rename(
        columns={
            "area_cells": area_cell_col,
            "area_inclusion_bodies": area_ib_col,
            "mean_cell_µm": mean_cell_col,
            "mean_ib_µm": mean_ib_col,
            "fwhm_cell_µm": fwhm_cell_col,
            "fwhm_ib_µm": fwhm_ib_col,
        }
    )

    # Split the metrics into a Results table (the answer — lysis and the peak
    # characteristics it derives from) and a Diagnostics table (how reliable the
    # fit is). Lysis efficiency leads the Results table so it is the first thing
    # the eye lands on, instead of buried at the far right of one wide table.
    results_cols = [
        "lysis_efficiency",
        "intact_fraction",
        "fit_kind",
        mean_cell_col,
        mean_ib_col,
        fwhm_cell_col,
        fwhm_ib_col,
        area_cell_col,
        area_ib_col,
        "area_total",
    ]
    diagnostics_cols = [
        "r_squared",
        "fit_quality",
        "model",
        "shoulder_verdict",
        "shoulder_excess_sigma",
        "area_robustness",
        "baseline_corrected",
        "normalized",
    ]

    results_cols = [c for c in results_cols if c in summary.columns]
    diagnostics_cols = [c for c in diagnostics_cols if c in summary.columns]

    # The exported frame keeps every column, Results first then Diagnostics, so
    # the XLSX reads the same way as the app (lysis up front).
    summary = summary[results_cols + diagnostics_cols]

    results_df = summary[results_cols]
    diagnostics_df = summary[diagnostics_cols]

    def highlight_r_squared(val: float) -> str:
        if val >= 0.95:
            return "background-color: #d4edda"  # Green - excellent
        elif val >= 0.90:
            return "background-color: #fff3cd"  # Yellow - good
        elif val >= 0.80:
            return "background-color: #f8d7da"  # Light red - fair
        else:
            return "background-color: #f5c6cb"  # Dark red - poor

    # --- Results table: the answer, lysis first ---
    st.subheader("Results")
    st.caption(
        "Lysis efficiency and the peak positions, widths and areas it is derived "
        "from. `fit_kind` is one / two / overlap — a one-peak fit forces lysis "
        "to 0% or 100%."
    )
    results_numeric = results_df.select_dtypes(include="number").columns
    results_formatters: Dict[str, str] = {
        col: "{:.4g}"
        for col in results_numeric
        if col not in ("lysis_efficiency", "intact_fraction")
    }
    if "lysis_efficiency" in results_df.columns:
        results_formatters["lysis_efficiency"] = "{:.1%}"
    if "intact_fraction" in results_df.columns:
        results_formatters["intact_fraction"] = "{:.1%}"
    st.dataframe(
        results_df.style.format(results_formatters, na_rep="—"),  # type: ignore[arg-type]
        width="stretch",
    )

    # --- Diagnostics table: fit quality + reliability ---
    st.subheader("Diagnostics")
    st.caption(
        "Fit quality and reliability indicators. Use these to judge how much to "
        "trust the results above before interpreting lysis efficiency.\n\n"
        f"**model** — peak model(s) actually fitted: 'A + B' means model A for the "
        f"{peak_name_ib.lower()} peak and model B for the {peak_name_cell.lower()} "
        "peak; a single name = one-peak fit of that model.\n"
        "**shoulder_verdict** — objective second-component check, independent of "
        "the fit: *shoulder* = detected, *none* = below the detection limit (not "
        "zero cells), *indeterminate* = near the limit, *n/a* = dominant peak at/"
        "after the cell target. On a resolved two-peak fit it fires trivially "
        "(the second peak is plain to see) — it is informative on one-peak and "
        "overlap fits.\n"
        "**shoulder_excess_sigma** — how strongly the right tail rises above the "
        "best single-peak prediction, in units of the measurement noise (σ). "
        "Bigger = more confident a second component is present. It measures "
        "confidence, not size — the amount of "
        f"{peak_name_cell.lower()} is the area / intact fraction. A high σ with "
        "verdict *indeterminate* means the tail bulges but the shape test did not "
        "confirm it, so do not call it a shoulder yet.\n"
        "**area_robustness** — overlap fits only: how much the cell-area split "
        "moves when the overlap assumptions vary (*stable* / *moderate* / "
        "*uncertain*)."
    )
    diag_numeric = diagnostics_df.select_dtypes(include="number").columns
    diag_formatters: Dict[str, str] = {
        col: "{:.4g}" for col in diag_numeric if col != "r_squared"
    }
    if "r_squared" in diagnostics_df.columns:
        diag_formatters["r_squared"] = "{:.4f}"
    styled_diag = diagnostics_df.style.format(diag_formatters, na_rep="—")  # type: ignore[arg-type]
    if "r_squared" in diagnostics_df.columns:
        styled_diag = styled_diag.map(highlight_r_squared, subset=["r_squared"])  # type: ignore[arg-type]
    st.dataframe(styled_diag, width="stretch")

    st.markdown(
        "**R² legend:** 🟢 ≥ 0.95 excellent · 🟡 ≥ 0.90 good · "
        "🟠 ≥ 0.80 fair · 🔴 < 0.80 poor"
    )

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
        file_name=f"lysosense_summary_v{__version__}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _render_experimental_data_download(
    results: List[Tuple[str, AnalysisResult]],
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> None:
    """Download button for experimental data with fits.

    Creates an Excel file where each sheet corresponds to one uploaded data file.
    Each sheet contains:
    - particle_size_um: Original x values
    - mass_signal_ug: Original y values (raw signal)
    - fit_signal_ug: Total fitted signal
    - {peak_name_cell} component (µg): Cells component of the fit
    - {peak_name_ib} component (µg): Inclusion bodies component of the fit
    """
    if not results:
        return

    buffer = io.BytesIO()
    used_sheet_names: set = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for label, analysis in results:
            # Create sheet name from filename (remove .dat extension)
            # Excel sheet names are limited to 31 characters. Long filenames that
            # share a 31-char prefix (e.g. ..._pass_1 / ..._pass_10) would collide;
            # openpyxl rejects duplicates, so de-duplicate with a numeric suffix.
            base_name = label.replace(".dat", "")[:31]
            sheet_name = base_name
            suffix = 2
            while sheet_name in used_sheet_names:
                tail = f"~{suffix}"
                sheet_name = base_name[: 31 - len(tail)] + tail
                suffix += 1
            used_sheet_names.add(sheet_name)

            # Use observed DataFrame which contains original data and fitted values.
            # Surface the user-supplied peak names in the exported column headers;
            # the in-memory column names are left unchanged.
            df = analysis.observed.copy().rename(
                columns={
                    "cells_component_ug": f"{peak_name_cell} component (µg)",
                    "ibs_component_ug": f"{peak_name_ib} component (µg)",
                }
            )

            # Write to sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    st.download_button(
        "Download experimental data (XLSX)",
        data=buffer.getvalue(),
        file_name=f"lysosense_experimental_data_v{__version__}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download an Excel file with each sample as a separate sheet, containing original data and fitted values.",
    )


def _render_details(
    entries: Sequence[Tuple[str, AnalysisResult]],
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> None:
    st.subheader("Detailed run information")
    for label, analysis in entries:
        measurement = analysis.measurement
        with st.expander(label):
            # Convert all values to strings to avoid Arrow serialization issues with mixed types
            meta_items = [
                (k, str(v) if not isinstance(v, str) else v)
                for k, v in sorted(measurement.metadata.items())
            ]
            meta_df = pd.DataFrame(meta_items, columns=["Field", "Value"])
            st.markdown("**Metadata**")
            st.dataframe(meta_df, hide_index=True, width="stretch")

            st.markdown("**Observed trace (first 15 points)**")
            preview = analysis.observed.head(15)[
                [
                    "particle_size_um",
                    "mass_signal_ug",
                    "fit_signal_ug",
                    "cells_component_ug",
                    "ibs_component_ug",
                ]
            ].rename(
                columns={
                    "cells_component_ug": f"{peak_name_cell} (µg)",
                    "ibs_component_ug": f"{peak_name_ib} (µg)",
                }
            )
            st.dataframe(preview, width="stretch")


def _render_overview_tab(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    view_mode: str,
    log_size_axis: bool = False,
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> None:
    """Render the Overview tab with combined plots."""
    st.markdown("### Combined Analysis Overview")
    st.markdown("Overview of all selected samples with fitted components and metrics.")

    # Render based on view mode
    if view_mode == "Raw Data Only":
        _render_raw_data_plot(entries, log_size_axis)
    elif view_mode == "Fit Overview":
        _render_fit_overview(
            entries, show_fit, show_components, log_size_axis,
            peak_name_cell, peak_name_ib,
        )
    else:  # Combined view (original)
        _render_plot(
            entries, show_fit, show_components, log_size_axis,
            peak_name_cell, peak_name_ib,
        )


def _render_individual_samples_tab(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    view_mode: str,
    log_size_axis: bool = False,
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
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
                    sample_name = label.replace(".dat", "")

                    st.markdown(f"**{sample_name}**")

                    # Create individual plot for this sample
                    fig = _create_individual_sample_plot(
                        [(label, analysis)],
                        show_fit,
                        show_components,
                        view_mode,
                        sample_name,
                        log_size_axis,
                        peak_name_cell,
                        peak_name_ib,
                    )
                    st.plotly_chart(fig, width="stretch")


def _create_individual_sample_plot(
    entries: Sequence[Tuple[str, AnalysisResult]],
    show_fit: bool,
    show_components: bool,
    view_mode: str,
    sample_name: str,
    log_size_axis: bool = False,
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> go.Figure:
    """Create a plot for a single sample."""
    fig = go.Figure()

    # Use a consistent color for the sample
    color = "#1f77b4"

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
                            name=_cell_component_label(analysis, peak_name_cell),
                            mode="lines",
                            line=dict(color=color, width=1.5),
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["ibs_component_ug"],
                        name=peak_name_ib,
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
                            name=_cell_component_label(analysis, peak_name_cell),
                            mode="lines",
                            line=dict(color=color, width=1.5),
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=analysis.dense_fit["particle_size_um"],
                        y=analysis.dense_fit["ibs_component_ug"],
                        name=peak_name_ib,
                        mode="lines",
                        line=dict(color=color, width=1.5, dash="dot"),
                    )
                )

    fig.update_layout(
        xaxis_title="Particle size (µm)",
        yaxis_title=_signal_yaxis_title(entries),
        template="plotly_white",
        margin=dict(l=40, r=10, t=20, b=40),  # Reduced top margin since no title
        height=300,  # Compact height for grid layout
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    _apply_size_axis_scale(fig, log_size_axis)

    return fig


def _render_results_tab(
    entries: Sequence[Tuple[str, AnalysisResult]],
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> pd.DataFrame:
    """Render the Results Table tab."""
    st.markdown("### Analysis Results")
    st.markdown("Detailed metrics and fit quality for all selected samples.")

    summary_df = _render_metrics(entries, peak_name_cell, peak_name_ib)
    return summary_df


def _render_details_tab(
    entries: Sequence[Tuple[str, AnalysisResult]],
    peak_name_cell: str = "Cells",
    peak_name_ib: str = "IBs",
) -> None:
    """Render the Detailed Information tab."""
    st.markdown("### Detailed Run Information")
    st.markdown("Comprehensive metadata and data preview for each sample.")

    _render_details(entries, peak_name_cell, peak_name_ib)


if __name__ == "__main__":
    main()
