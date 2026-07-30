"""Build the standalone LysoSense user guide as a self-contained HTML page.

The guide is opened in its own browser window (see ``streamlit_app``) so it can
sit beside the running app. It bundles:

* rich HTML prose (``GUIDE_BODY``) describing the whole workflow, and
* interactive Plotly example figures (SYNTHETIC DCS traces) embedded inline.

The example traces are generated from simple peak functions — they are
illustrative, not real measurements. Plotly.js is inlined once via
``plotly.offline.get_plotlyjs`` so the page works fully offline. No new runtime
dependencies: only numpy and plotly (both already used by the app).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly import offline as _plotly_offline  # type: ignore[import-untyped]

# Visual conventions kept consistent with the app's charts.
_RAW_COLOR = "#6b7280"
_FIT_COLOR = "#1f77b4"
_CELL_COLOR = "#2ca02c"
_IB_COLOR = "#ff7f0e"
_TIGHT_COLOR = "#d62728"


def _peak(x: np.ndarray, height: float, mu: float, sigma: float) -> np.ndarray:
    """Symmetric peak of given ``height`` (max value) at ``mu`` with width ``sigma``."""
    return height * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Particle size (µm)",
        yaxis_title="Signal",
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=20, t=45, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
    )
    fig.update_xaxes(range=[0.2, 1.2])
    return fig


def _noisy(total: np.ndarray, amp: float, rng: np.random.Generator) -> np.ndarray:
    return total + rng.normal(0.0, amp, size=total.shape)


def _fig_clean_two_peak(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400)
    ib = _peak(x, 1.0, 0.48, 0.06)
    cell = _peak(x, 0.32, 0.86, 0.09)
    raw = _noisy(ib + cell, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib + cell, name="Fit", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=cell, name="Cells", line=dict(color=_CELL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib, name="IBs", line=dict(color=_IB_COLOR, width=2, dash="dot")))
    return _base_layout(fig, "A · Clean two-peak fit")


def _fig_overlap(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400)
    ib = _peak(x, 1.0, 0.50, 0.09)
    cell = _peak(x, 0.22, 0.70, 0.07)  # sits as a shoulder on the IB slope
    raw = _noisy(ib + cell, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib + cell, name="Fit", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=cell, name="Cells (shoulder)", line=dict(color=_CELL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib, name="IBs", line=dict(color=_IB_COLOR, width=2, dash="dot")))
    return _base_layout(fig, "B · Overlapping peaks (shoulder)")


def _fig_single_ib(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400)
    ib = _peak(x, 1.0, 0.48, 0.07)
    raw = _noisy(ib, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib, name="Fit (IB only)", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    return _base_layout(fig, "C · Lone IB peak → lysis ≈ 100%")


def _fig_lone_cell(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400)
    cell = _peak(x, 1.0, 0.85, 0.08)
    raw = _noisy(cell, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=cell, name="Fit (cells only)", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    return _base_layout(fig, "E · Lone cell peak → lysis ≈ 0%")


def _fig_broad(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400)
    raw = _noisy(_peak(x, 1.0, 0.55, 0.16), 0.012, rng)
    tight = _peak(x, 1.0, 0.55, 0.075)  # default tight bounds underfit a broad peak
    relaxed = _peak(x, 1.0, 0.55, 0.16)
    r2_tight = _r_squared(raw, tight)
    r2_relaxed = _r_squared(raw, relaxed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(
        go.Scatter(
            x=x, y=tight, name=f"Tight fit (R²={r2_tight:.2f})", line=dict(color=_TIGHT_COLOR, width=2.5, dash="dash")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=relaxed, name=f"Relaxed fit (R²={r2_relaxed:.2f})", line=dict(color=_FIT_COLOR, width=2.5)
        )
    )
    return _base_layout(fig, "D · Broad peak: tight vs. relaxed width")


def _fig_not_evaluable(rng: np.random.Generator) -> go.Figure:
    """Synthetic stand-in for a non-evaluable trace (modelled on a real file that
    showed a strongly negative, drifting baseline). Not for quantitative use."""
    x = np.linspace(0.2, 1.2, 400)
    drift = -3.0 + 3.2 * (x - 0.2)  # non-physical baseline: ~-3 at small sizes → ~+0.2
    bumps = _peak(x, 0.6, 0.30, 0.04) + _peak(x, 0.4, 0.95, 0.06)
    raw = drift + bumps + rng.normal(0.0, 0.04, size=x.shape)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_hline(y=0.0, line=dict(color="#9ca3af", width=1, dash="dot"))
    fig.add_annotation(x=0.28, y=0.55, text="zero line", showarrow=False, font=dict(size=10, color="#9ca3af"))
    fig.update_layout(
        title=dict(text="F · Not evaluable: non-physical baseline", font=dict(size=14)),
        xaxis_title="Particle size (µm)",
        yaxis_title="Signal",
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=20, t=45, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
    )
    fig.update_xaxes(range=[0.2, 1.2])
    return fig


def _fig_shoulder_clear(rng: np.random.Generator) -> go.Figure:
    """A detectable shoulder: the cell region sits above the single-peak prediction."""
    x = np.linspace(0.2, 1.2, 400)
    ib = _peak(x, 1.0, 0.70, 0.08)
    cell = _peak(x, 0.32, 0.95, 0.07)  # shoulder riding on the IB right tail
    raw = _noisy(ib + cell, 0.012, rng)
    single = _peak(x, 1.0, 0.70, 0.08)  # what one peak (IB only) predicts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=single, name="Single-peak prediction", line=dict(color=_TIGHT_COLOR, width=2.5, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=x, y=cell, name="Shoulder (excess)", line=dict(color=_CELL_COLOR, width=2))
    )
    return _base_layout(fig, "G · A detectable shoulder")


def _fig_shoulder_hidden(rng: np.random.Generator) -> go.Figure:
    """No detectable shoulder: the right tail matches one peak within the noise."""
    x = np.linspace(0.2, 1.2, 400)
    ib = _peak(x, 1.0, 0.70, 0.11)  # broader single peak whose tail reaches the cell region
    raw = _noisy(ib, 0.012, rng)
    single = _peak(x, 1.0, 0.70, 0.11)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw data", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=single, name="Single-peak prediction", line=dict(color=_FIT_COLOR, width=2.5, dash="dash"))
    )
    return _base_layout(fig, "H · No detectable shoulder (below the limit)")


def _build_figures() -> Dict[str, str]:
    """Return ``{figure_id: plotly div+script snippet}`` for the guide."""
    rng = np.random.default_rng(20260729)
    figures = {
        "clean": _fig_clean_two_peak(rng),
        "overlap": _fig_overlap(rng),
        "single_ib": _fig_single_ib(rng),
        "broad": _fig_broad(rng),
        "lone_cell": _fig_lone_cell(rng),
        "bad": _fig_not_evaluable(rng),
        "shoulder_clear": _fig_shoulder_clear(rng),
        "shoulder_hidden": _fig_shoulder_hidden(rng),
    }
    snippets: Dict[str, str] = {}
    for fid, fig in figures.items():
        snippets[fid] = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f"guide_fig_{fid}",
            config={"responsive": True, "displaylogo": False},
        )
    return snippets


def build_guide_html() -> str:
    """Assemble the full standalone guide HTML (prose + figures + plotly.js)."""
    snippets = _build_figures()
    body = GUIDE_BODY
    for fid, snippet in snippets.items():
        body = body.replace("{{FIGURE:" + fid + "}}", snippet)

    plotly_js = _plotly_offline.get_plotlyjs()
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>LysoSense — User Guide</title>\n"
        "<style>\n" + _CSS + "\n</style>\n"
        "<script>" + plotly_js + "</script>\n"
        "</head>\n<body>\n<main>\n" + body + "\n</main>\n</body>\n</html>\n"
    )


_CSS = """
:root { --accent: #1f77b4; }
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  color: #1f2937; line-height: 1.6; margin: 0; padding: 0;
  background: #ffffff;
}
main { max-width: 940px; margin: 0 auto; padding: 2rem 1.25rem 6rem; }
h1 { font-size: 1.9rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.35rem; margin-top: 2.2rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.3rem; }
h3 { font-size: 1.1rem; margin-top: 1.4rem; color: #111827; }
p, li { font-size: 0.98rem; }
a { color: var(--accent); }
code { background: #f3f4f6; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.9em; }
pre { background: #1f2937; color: #e5e7eb; padding: 0.9rem 1rem; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; color: inherit; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }
th, td { border: 1px solid #e5e7eb; padding: 0.45rem 0.6rem; text-align: left; vertical-align: top; }
th { background: #f9fafb; }
.toc { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.75rem 1.25rem 0.85rem; }
.toc ul { margin: 0.3rem 0 0; padding-left: 1.1rem; }
.note { background: #eff6ff; border-left: 4px solid var(--accent); padding: 0.6rem 0.9rem; border-radius: 0 6px 6px 0; margin: 1rem 0; }
.warn { background: #fff7ed; border-left: 4px solid #f59e0b; padding: 0.6rem 0.9rem; border-radius: 0 6px 6px 0; margin: 1rem 0; }
.danger { background: #fef2f2; border-left: 4px solid #ef4444; padding: 0.6rem 0.9rem; border-radius: 0 6px 6px 0; margin: 1rem 0; }
figure { margin: 1.2rem 0; }
figcaption { text-align: center; color: #6b7280; font-size: 0.88rem; margin-top: 0.4rem; }
.js-plotly-plot { width: 100% !important; }
.tag { display:inline-block; background:#eef2ff; color:#3730a3; border-radius:999px; padding:0.05rem 0.55rem; font-size:0.8rem; margin: 0 0.1rem; }
"""


GUIDE_BODY = """
<h1>LysoSense — User Guide</h1>
<p><em>Interactive guide with example plots. The plots are live — hover for
values, <strong>click a trace name in the legend to isolate or hide it</strong>,
and use the toolbar (top-right of each plot) to zoom and pan. The same controls
work in the app.</em></p>

<div class="toc">
  <strong>Contents</strong>
  <ul>
    <li>What LysoSense measures</li>
    <li>About the DCS measurement itself</li>
    <li>Set the IB &amp; cell sizes (important)</li>
    <li>How the fit is decided</li>
    <li>Step-by-step workflow</li>
    <li>Sidebar reference</li>
    <li>Reading the results</li>
    <li>Why shoulder results are uncertain</li>
    <li>Worked examples</li>
    <li>Tips, gotchas &amp; FAQ</li>
    <li>Troubleshooting</li>
  </ul>
</div>

<h2 id="what">What LysoSense measures</h2>
<p>LysoSense analyses <strong>differential centrifugal sedimentation (DCS/CPS)</strong>
traces. For cell-disruption work the trace usually contains two populations:</p>
<ul>
  <li><strong>Inclusion bodies (IBs)</strong> — small particles released from
  broken cells (default target <code>~0.48 µm</code>).</li>
  <li><strong>Intact cells</strong> — larger (default target <code>~0.85 µm</code>).</li>
</ul>
<p>The headline output is <strong>lysis efficiency</strong> — the fraction of
total signal that is <em>not</em> in the intact-cell peak:</p>
<pre><code>lysis_efficiency = 1 − (area of the Cell peak ÷ total area)</code></pre>
<p>High lysis ⇒ most cells are disrupted and their contents (IBs) have been
released. The analysis method follows Klausser et al., 2025 (linked on the start
page), implemented here as an <strong>automated, algorithm-based</strong>
workflow.</p>

<h2 id="measurement">About the DCS measurement itself</h2>
<p>It helps to know what the instrument actually reports. In DCS, particles are
injected into a spinning sucrose gradient and sediment outward; the instrument
times when each size band passes a detector. That sedimentation time is converted
to a size via <strong>Stokes' law</strong> — settling speed depends on
diameter² and on (particle density − fluid density).</p>
<p>Two consequences worth remembering:</p>
<ul>
  <li>The x-axis is an <strong>equivalent spherical diameter</strong>. The
  calculation assumes spherical particles of a known density. Real cells and IBs
  are not perfect spheres and their density may differ from the assumed value, so
  <strong>absolute µm values are approximate</strong>, not exact.</li>
  <li>DCS is excellent for <strong>relative</strong> comparisons — peak
  <em>positions</em> and, crucially, peak <em>areas</em> (which lysis % is built
  on) are reliable within and across runs.</li>
</ul>
<p>So: trust the shapes and the area ratios; treat the absolute sizes as a guide.
This is also why the next point matters so much.</p>

<h2 id="sizes">Set the IB &amp; cell sizes — this is critical</h2>
<div class="danger">
<strong>The algorithm assumes peaks near the IB and cell target sizes you set.</strong>
If your instrument, organism or process produces peaks at <em>different</em> sizes
than the defaults (0.48 / 0.85 µm), you <strong>must</strong> update
<em>IB target size</em> and <em>Cell target size</em> in the sidebar. With wrong
targets the fit can miss a peak, attach it to the wrong component, or report a
nonsense lysis value (e.g. 0% or 100%). Always confirm the peaks sit where the
targets say before trusting the numbers.
</div>
<p>This is especially fragile for <strong>single-peak</strong> traces: the lone
peak is assigned to whichever target it is closest to.</p>
<ul>
  <li>Lone peak near the <strong>IB</strong> size → reported as <strong>~100% lysis</strong>.</li>
  <li>Lone peak near the <strong>cell</strong> size → reported as <strong>~0% lysis</strong> (i.e. 100% cells).</li>
</ul>
<p>So a single peak does <em>not</em> always mean 100% lysis — it means
"whatever population that peak was assigned to". If that assignment is wrong
because the target sizes are off, the lysis number is wrong. (See Examples C and E.)

<h2 id="pipeline">How the fit is decided</h2>
<p>For every uploaded trace LysoSense tries models in order and keeps the first
that is justified by the data:</p>
<ol>
  <li><strong>Fit one peak.</strong></li>
  <li><strong>Is there a real second peak?</strong> Statistical <em>gates</em>
  decide — the residual (signal left after the 1-peak fit) must contain a peak
  that is prominent enough, far enough from the first peak, and that improves the
  Bayesian Information Criterion (BIC).</li>
  <li><strong>If yes → two-peak fit.</strong> The peaks must also be locally
  dominant and sufficiently separated.</li>
  <li><strong>If the second peak is only a hidden shoulder</strong> (no separate
  maximum), <em>overlap deconvolution</em> can still split it — and reports an
  <em>area-robustness</em> tag (stable / moderate / uncertain) telling you how
  trustworthy that split is.</li>
</ol>
<p>The reported <code>fit_kind</code> is one of <span class="tag">one</span>
<span class="tag">two</span> <span class="tag">overlap</span>.</p>

<h2 id="workflow">Step-by-step workflow</h2>
<ol>
  <li><strong>Upload</strong> one or more <code>.dat</code> files in
  <em>Data Upload</em>. Each file is one trace; upload several to compare them.</li>
  <li><strong>Check the peaks sit at the right sizes</strong> and adjust
  <em>IB / Cell target size</em> if needed (see above).</li>
  <li><strong>(Optional) Preprocess</strong> — baseline subtraction, normalization,
  or restricting the particle-size window.</li>
  <li><strong>Pick a peak-detection mode.</strong></li>
  <li><strong>Fit</strong> runs automatically. Use the <em>Results Table</em> tab
  to read lysis % and R²; use <em>Individual Samples</em> to inspect each fit
  visually.</li>
  <li><strong>Download</strong> the summary or the full experimental data as XLSX.</li>
</ol>

<h2 id="sidebar">Sidebar reference</h2>

<h3>Data Upload</h3>
<p>Drop <code>.dat</code> exports. Use the <em>Traces to analyze</em> selector on
the main page to focus on a subset.</p>

<h3>Peaks &amp; Sample</h3>
<ul>
  <li><strong>Peak labels</strong> — rename the two components (e.g.
  <em>debris</em>, <em>aggregates</em>) for separation samples. Names appear in
  the plots, the results table and the downloads. Lysis % is always calculated
  for the <em>Cell</em> peak.</li>
  <li><strong>IB / Cell target size (µm)</strong> — expected peak centres
  (defaults 0.48 / 0.85). <strong>Adjust these to where your peaks actually
  sit.</strong> The fit may shift each peak within the <em>Allowed peak shift</em>
  window.</li>
  <li><strong>Limit particle-size range</strong> — restrict the fit to a size
  window (default 0.2–1.2 µm). Keep this on to ignore large debris outside the
  range of interest.</li>
</ul>

<h3>Data Preprocessing</h3>
<ul>
  <li><strong>Baseline subtraction</strong> — removes a constant/edge offset.
  Try only if the trace clearly does not return to zero (or goes negative).
  Methods: <em>minimum</em>, <em>percentile</em> (1st), <em>linear</em> (edge fit).</li>
  <li><strong>Normalize data</strong> — scales every trace to its own maximum.
  Use this to compare samples of different concentration. Units become
  <em>relative weight</em> rather than µg.</li>
</ul>

<h3>Fitting</h3>
<ul>
  <li><strong>Peak model</strong> — the shape each peak is fitted with.
  <em>autofit</em> (recommended) tries all four and keeps the best R²; or pick one
  for both peaks, or a different model per peak:
  <ul>
    <li><strong>gaussian</strong> — symmetric bell. Simplest; often too stiff for
    real (skewed) DCS peaks.</li>
    <li><strong>lognormal</strong> — asymmetric; the natural shape for particle-size
    distributions and a good single default.</li>
    <li><strong>splitgaussian</strong> — different width on each side of the peak;
    flexible, can absorb a shoulder.</li>
    <li><strong>gennormal</strong> — generalized normal with a tunable top (flat to
    sharp); the most flexible — and the most prone to hiding a real shoulder by
    flattening, so autofit uses it cautiously.</li>
  </ul>
  </li>
  <li><strong>Relax peak-width constraints</strong> — for genuinely broad peaks
  (see Example D). A tight fit is tried first; widths are relaxed only if R² is
  poor, so clean traces are unaffected.</li>
  <li><strong>Peak detection</strong>:
  <ul>
    <li><strong>Automatic</strong> (default): resolved peaks first, overlap
    deconvolution only if a shoulder is detected.</li>
    <li><strong>Resolved peaks only</strong>: stricter — needs a clear second
    maximum.</li>
    <li><strong>Allow overlapping peaks</strong>: forces overlap deconvolution on.</li>
    <li><strong>Single peak only</strong>: disables the two-peak fit entirely.</li>
  </ul>
  </li>
</ul>

<h3>Fitting — Advanced</h3>
<p><strong>Sensitivity</strong> presets (Low / Medium / High) set the 2-peak
gates together. <em>Custom</em> exposes them individually:</p>
<table>
  <tr><th>Gate</th><th>What it controls</th></tr>
  <tr><td>Residual prominence / distance / area</td><td>How strong a leftover signal must be to count as a second peak.</td></tr>
  <tr><td>BIC improvement threshold</td><td>How much the 2-peak model must beat the 1-peak model.</td></tr>
  <tr><td>Local dominance</td><td>The second peak must "own" some region of the curve.</td></tr>
  <tr><td>Min 2nd peak area</td><td>Smallest area fraction to keep the second peak.</td></tr>
  <tr><td>Min separation</td><td>Peaks must be far enough apart relative to their width.</td></tr>
  <tr><td>Max Cell peak FWHM / compactness / prominence</td><td>Quality constraints on the (usually smaller) second peak.</td></tr>
</table>
<p><strong>Fitting constraints:</strong> <em>Allowed peak shift</em>,
<em>Min 2nd peak fraction</em>, <em>Max peak width</em> (FWHM cap), and
<em>Peak-top weighting</em> (0 = ordinary least squares; higher gives high-signal
points more influence).</p>

<h3>Overlap deconvolution (Advanced)</h3>
<p>These only matter when a second peak is fit as a <em>shoulder</em> (no maximum of
its own) via overlap deconvolution. They constrain that split:</p>
<ul>
  <li><strong>Cell center shift (%)</strong> — how far the cell peak's centre may move
  around the <em>Cell target size</em> during the overlap fit.</li>
  <li><strong>Max overlap IB / cell FWHM (µm)</strong> — the widest each peak is allowed
  to be in the overlap fit (stops the deconvolution from spreading unrealistically).</li>
  <li><strong>Min overlap cell area (%)</strong> — the smallest cell area below which an
  overlap split is rejected as negligible.</li>
</ul>
<p>Because shoulder fits are inherently uncertain (see
<a href="#shoulder-uncertain">Why shoulder results are uncertain</a>), the
<code>area_robustness</code> tag tells you how much the cell‑area answer moves when
these settings vary.</p>

<h3>Visualization</h3>
<p><strong>View mode:</strong> Combined (raw + fit + components), Fit Overview
(components only), or Raw Data Only. Toggle the fit envelope, components, and a
logarithmic size axis (display only — the fit always runs in linear µm).</p>
<p><strong>Light / dark theme:</strong> switch via the menu (<span class="kbd">☰</span>,
top-right) → <em>Settings</em> → <em>Theme</em> (light, dark, or follow system).</p>

<h2 id="results">Reading the results</h2>
<p>The <strong>Results Table</strong> tab shows two tables: <strong>Results</strong>
(lysis efficiency and the peak positions, widths and areas it derives from) and
<strong>Diagnostics</strong> (how reliable each fit is). Lysis efficiency leads the
Results table.</p>
<table>
  <tr><th>Column</th><th>Meaning</th></tr>
  <tr><td><code>fit_kind</code></td><td>one / two / overlap — how many peaks were fitted.</td></tr>
  <tr><td><code>lysis_efficiency</code></td><td>1 − cell area ÷ total area. The headline number.</td></tr>
  <tr><td><code>intact_fraction</code></td><td>cell area ÷ total area (= 1 − lysis).</td></tr>
  <tr><td><code>area_cells</code> / <code>area_inclusion_bodies</code></td><td>Integrated area of each component.</td></tr>
  <tr><td><code>area_total</code></td><td>Sum of the two component areas.</td></tr>
  <tr><td><code>mean_cell_µm</code> / <code>mean_ib_µm</code></td><td>Mean particle size of each component.</td></tr>
  <tr><td><code>fwhm_cell_µm</code> / <code>fwhm_ib_µm</code></td><td>Full-width-at-half-maximum (peak width) of each component. A surprisingly narrow width can flag an over-tight fit — try <em>Relax peak-width constraints</em>.</td></tr>
  <tr><td><code>r_squared</code></td><td>Goodness of fit. 🟢 ≥0.95 · 🟡 ≥0.90 · 🟠 ≥0.80 · 🔴 &lt;0.80.</td></tr>
  <tr><td><code>area_robustness</code></td><td>Overlap fits only: <em>stable / moderate / uncertain</em> — how much the cell-area estimate moves when overlap settings vary.</td></tr>
  <tr><td><code>shoulder_verdict</code></td><td>Objective second-component check: <span class="tag">shoulder</span> (a second component is detected), <span class="tag">none</span> (clean single peak), <span class="tag">indeterminate</span> (near the detection limit), <span class="tag">n/a</span> (dominant peak at/after the cell target, or trace not evaluable).</td></tr>
  <tr><td><code>shoulder_excess_sigma</code></td><td>How <em>confidently</em> a shoulder is detected — the signal excess over a single-peak prediction, in noise σ. This is confidence, <strong>not</strong> shoulder size (see below).</td></tr>
</table>
<div class="note">
<strong>Reading the shoulder columns:</strong> <code>shoulder_excess_sigma</code> says how
<em>confidently</em> a second component is detected, <strong>not how big it is</strong>. The size of the
cell population is <code>intact_fraction</code> (or <code>area_cells</code>) — a small but clean shoulder
can score a higher σ than a large but noisy one. So judge <em>presence</em> by the verdict/σ and
<em>magnitude</em> by the area. And <code>shoulder_verdict = none</code> means <strong>below the detection
limit</strong>, not zero intact cells — the true residual could be anything from 0 to a few %.
</div>
<p>The four tabs: <strong>Overview</strong> (all traces together),
<strong>Individual Samples</strong> (one plot per sample),
<strong>Results Table</strong> (the metrics), and
<strong>Detailed Information</strong> (metadata + raw numbers).</p>

<h2 id="shoulder-uncertain">Why shoulder (overlap) results are uncertain</h2>
<p>A <strong>resolved</strong> second peak has its own maximum, so its position, height and width are
directly visible and the split is well constrained. A <strong>shoulder</strong> has no maximum of its own
— it shows up only as a distortion of the main peak's tail — so its area has to be inferred from the
leftover (residual) shape. That makes shoulder / overlap fits inherently less certain, which is why they
carry an <code>area_robustness</code> tag.</p>
<div class="warn">
<strong>The fundamental ambiguity.</strong> An asymmetric, non-Gaussian peak can be explained two ways,
and near the detection limit they are hard to tell apart:
<ul>
  <li>a <em>symmetric</em> main peak plus a small <strong>shoulder</strong> (a real second population), or</li>
  <li>one genuinely <strong>asymmetric</strong> peak — a single non-Gaussian population with no second component.</li>
</ul>
The shoulder check uses an asymmetric (lognormal) single-peak reference precisely to avoid calling every
skewed peak a shoulder — but it cannot always separate the two, so small / late-cycle shoulders stay
genuinely ambiguous.
</div>
<p>This gets <strong>harder when the IB and cell sizes overlap</strong> (i.e. are close together). The
closer the two populations, the more the cell shoulder sits right on the IB peak's tail, and the more a
"second population" looks identical to "one skewed IB peak". There the lysis % becomes ill-conditioned —
small changes in the model or the noise move the split substantially. When your IB and cell targets are
close (common for some strains), treat late-cycle / small-shoulder lysis numbers as approximate and report
the <em>trend</em> across cycles rather than a single precise value.</p>

<h3>What a shoulder looks like — and when it's below the limit</h3>
<p>The shoulder check separates two situations that are easy to confuse by eye:</p>
{{FIGURE:shoulder_clear}}
<figcaption>Fig. G — <strong>A detectable shoulder.</strong> In the cell region the raw
data (grey) rises clearly above the single-peak prediction (red dashed); that excess
(green) is the second population. Verdict: <span class="tag">shoulder</span>.</figcaption>
{{FIGURE:shoulder_hidden}}
<figcaption>Fig. H — <strong>No detectable shoulder.</strong> The right tail is fully
consistent with a single peak — the prediction tracks the data within the noise, so
there is no excess to flag. There <em>could</em> still be a small shoulder buried in
the noise; we simply can't tell, so the verdict is <span class="tag">none</span> (or
<span class="tag">indeterminate</span>) — meaning "below the detection limit", not
"definitely zero". This is exactly the late-cycle situation: report the trend across
samples, not a precise residual from a single trace.</figcaption>

<h2 id="examples">Worked examples</h2>
<div class="note">The traces below are <strong>synthetic illustrations</strong>
(generated from simple peak functions), not real measurements — they are designed
to show each situation clearly. The "not evaluable" one (F) is modelled on a real
file that showed a strongly negative, drifting baseline.</div>

<h3>Example A — Clean two-peak fit (trust the numbers)</h3>
<p>Two well-separated peaks. The gates accept the 2-peak model, R² is high and
the lysis % is reliable.</p>
{{FIGURE:clean}}
<figcaption>Fig. A — IB peak near 0.48 µm, cell peak near 0.86 µm. Raw (grey),
fit envelope (blue dashed), cells (green), IBs (orange dotted).</figcaption>

<h3>Example B — Overlapping peaks / shoulder (mind the robustness tag)</h3>
<p>The cell population appears only as a shoulder on the IB peak — no separate
maximum. <em>Overlap deconvolution</em> splits it, but the result is sensitive to
assumptions, so check <code>area_robustness</code>: treat
<span class="tag">uncertain</span> splits cautiously.</p>
{{FIGURE:overlap}}
<figcaption>Fig. B — the green cell component has no peak of its own in the raw
data; it was deconvolved from the shoulder.</figcaption>

<h3>Example C — Lone IB peak → lysis ≈ 100%</h3>
<p>No separate cell peak is detected and the lone peak sits near the IB size, so
it is assigned to IBs and lysis is reported as ~100%. Expected for a fully
disrupted sample — but see Example E for the opposite case.</p>
{{FIGURE:single_ib}}
<figcaption>Fig. C — one peak near the IB size → lysis ≈ 100%.</figcaption>

<h3>Example E — Lone cell peak → lysis ≈ 0% (i.e. 100% cells)</h3>
<p>The same logic, the other way: a lone peak near the <em>cell</em> size is
assigned to cells, so lysis is reported as ~0%. A single peak therefore does
<em>not</em> always mean 100% lysis — it depends on which target the peak is
closest to, which is why the target sizes must be right.</p>
{{FIGURE:lone_cell}}
<figcaption>Fig. E — one peak near the cell size → lysis ≈ 0%.</figcaption>

<h3>Example D — Broad peak: when to relax widths</h3>
<p>If the default (tight) width bounds are narrower than the real peak, the fit
underfits the wings and R² drops. Turning on <strong>Relax peak-width
constraints</strong> lets the peak widen to match the data.</p>
{{FIGURE:broad}}
<figcaption>Fig. D — tight fit (red) underfits a genuinely broad peak; relaxed
fit (blue) matches it. Typical symptom: the fit looks sharper than the data and
R² is poor.</figcaption>

<h3>Example F — Not evaluable: non-physical baseline</h3>
<p>Some traces cannot be fit meaningfully. The hallmark here is a <strong>strongly
negative, drifting baseline</strong> — mass signal should not be negative, so a
trace that sits well below zero over much of the range means the measurement (or
its reference subtraction) went wrong. Any IB/cell fit on top of that is
meaningless.</p>
<div class="warn"><strong>What to do:</strong> try <em>Baseline subtraction</em>;
check the sample preparation / dilution; if it stays like this, <strong>exclude
the sample</strong> — do not report a lysis number from it.</div>
{{FIGURE:bad}}
<figcaption>Fig. F — signal strongly negative with only small positive
excursions; the dotted line marks zero. Not evaluable.</figcaption>

<h2 id="tips">Tips, gotchas &amp; FAQ</h2>
<ul>
  <li><strong>Single peak is ambiguous.</strong> It is assigned by proximity to
  the IB or cell target: IB-side → ~100% lysis, cell-side → ~0% (see C &amp; E).
  If the target sizes are wrong, the assignment — and the lysis number — is
  wrong.</li>
  <li><strong>Always check target sizes first.</strong> Wrong IB/cell sizes are
  the most common cause of nonsense results.</li>
  <li><strong>Poor R²?</strong> Try (in order): enable <em>Relax peak-width
  constraints</em>; widen the <em>size range</em>; check <em>Sensitivity</em>;
  try a different <em>Peak model</em>.</li>
  <li><strong>Uncertain overlap?</strong> Don't trust the split — report the
  total area, or change the detection mode to compare.</li>
  <li><strong>Shoulder verdict says <em>none</em> but you expected a second peak?</strong>
  It's below the detection limit — the leftover is indistinguishable from noise or peak
  asymmetry. Treat the number as approximate and report the trend, or raise <em>Sensitivity</em>
  / use <em>Allow overlapping peaks</em> to force a split and see how much it moves.</li>
  <li><strong>Compare concentrations?</strong> Turn on <em>Normalize data</em> so
  traces are on the same scale.</li>
  <li><strong>Separation samples?</strong> Rename the peaks (Peak labels) to what
  they actually are.</li>
  <li><strong>Non-physical / negative trace?</strong> Not evaluable — see
  Example F.</li>
</ul>

<h2 id="trouble">Troubleshooting</h2>
<table>
  <tr><th>Symptom</th><th>Likely fix</th></tr>
  <tr><td>All traces failed to parse</td><td>Confirm the files are CPS/DCS <code>.dat</code> exports.</td></tr>
  <tr><td>Peaks not where the targets assume</td><td>Set <em>IB / Cell target size</em> to the real peak positions.</td></tr>
  <tr><td>Fit is much narrower than the data, low R²</td><td>Enable <em>Relax peak-width constraints</em>.</td></tr>
  <tr><td>Expected second peak is missed</td><td>Set <em>Sensitivity → High</em>, or <em>Allow overlapping peaks</em>.</td></tr>
  <tr><td>Spurious second peak appears</td><td>Set <em>Sensitivity → Low</em>, or tighten <em>Max peak width</em>.</td></tr>
  <tr><td>Shoulder verdict <em>none</em> but a second peak is expected</td><td>Near the detection limit — raise <em>Sensitivity</em> or <em>Allow overlapping peaks</em>; treat the number as approximate (see <em>Why shoulder results are uncertain</em>).</td></tr>
  <tr><td>Trace is negative or drifting</td><td>Not evaluable — try baseline subtraction, else exclude the sample.</td></tr>
</table>

<hr>
<p><small>LysoSense user guide — generated from the app. Re-open anytime with the
<em>📖 Guide</em> button in the sidebar.</small></p>
"""
