"""Build the standalone LysoSense user guide as a self-contained HTML page.

The guide is opened in its own browser window (see ``streamlit_app``) so it can
sit beside the running app. It bundles:

* rich HTML prose (``GUIDE_BODY``) describing the whole workflow, and
* interactive Plotly example figures (SYNTHETIC DCS traces) embedded inline.

The example traces are generated from simple peak functions — they are
illustrative, not real measurements. Every figure shares the same relative
axes: particle size relative to the cell target size (the cell peak sits near
1) and signal as a fraction of each trace's maximum. Plotly.js is inlined once
via ``plotly.offline.get_plotlyjs`` so the page works fully offline. No new
runtime dependencies: only numpy and plotly (both already used by the app).
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

# All example figures are plotted on relative axes (matching the plateau
# example): x is particle size relative to the cell target size — the cell
# peak sits near 1 — and y is each trace normalized to its own maximum. The
# synthetic figures are designed at the app-default targets (IB 0.48 /
# cell 0.85 µm) and converted with this constant.
_CELL_TARGET_UM = 0.85


def _peak(x: np.ndarray, height: float, mu: float, sigma: float) -> np.ndarray:
    """Symmetric peak of given ``height`` (max value) at ``mu`` with width ``sigma``."""
    return height * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _peak_um(x: np.ndarray, height: float, mu: float, sigma: float) -> np.ndarray:
    """Peak designed at µm positions, drawn on the relative x-axis."""
    return _peak(x, height, mu / _CELL_TARGET_UM, sigma / _CELL_TARGET_UM)


def _r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Relative particle size",
        yaxis_title="Signal (normalized)",
        template="plotly_white",
        height=330,
        margin=dict(l=45, r=20, t=45, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
    )
    fig.update_xaxes(range=[0.2 / _CELL_TARGET_UM, 1.2 / _CELL_TARGET_UM])
    return fig


def _noisy(total: np.ndarray, amp: float, rng: np.random.Generator) -> np.ndarray:
    return total + rng.normal(0.0, amp, size=total.shape)


def _fig_clean_two_peak(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    ib = _peak_um(x, 1.0, 0.48, 0.06)
    cell = _peak_um(x, 0.32, 0.86, 0.09)
    raw = _noisy(ib + cell, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib + cell, name="Fit", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=cell, name="Cells", line=dict(color=_CELL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib, name="IBs", line=dict(color=_IB_COLOR, width=2, dash="dot")))
    return _base_layout(fig, "A · Clean two-peak fit")


def _fig_overlap(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    ib = _peak_um(x, 1.0, 0.50, 0.09)
    cell = _peak_um(x, 0.22, 0.70, 0.07)  # sits as a shoulder on the IB slope
    raw = _noisy(ib + cell, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib + cell, name="Fit", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=cell, name="Cells (shoulder)", line=dict(color=_CELL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib, name="IBs", line=dict(color=_IB_COLOR, width=2, dash="dot")))
    return _base_layout(fig, "B · Overlapping peaks (shoulder)")


def _fig_single_ib(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    ib = _peak_um(x, 1.0, 0.48, 0.07)
    raw = _noisy(ib, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=ib, name="Fit (IB only)", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    return _base_layout(fig, "C · Lone IB peak → lysis ≈ 100%")


def _fig_lone_cell(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    cell = _peak_um(x, 1.0, 0.85, 0.08)
    raw = _noisy(cell, 0.012, rng)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=x, y=cell, name="Fit (cells only)", line=dict(color=_FIT_COLOR, width=2.5, dash="dash")))
    return _base_layout(fig, "E · Lone cell peak → lysis ≈ 0%")


def _fig_broad(rng: np.random.Generator) -> go.Figure:
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    raw = _noisy(_peak_um(x, 1.0, 0.55, 0.16), 0.012, rng)
    tight = _peak_um(x, 1.0, 0.55, 0.075)  # default tight bounds underfit a broad peak
    relaxed = _peak_um(x, 1.0, 0.55, 0.16)
    r2_tight = _r_squared(raw, tight)
    r2_relaxed = _r_squared(raw, relaxed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
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
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    drift = -3.0 + 3.2 * _CELL_TARGET_UM * (x - 0.2 / _CELL_TARGET_UM)  # ~-3 at small sizes → ~+0.2
    bumps = _peak_um(x, 0.6, 0.30, 0.04) + _peak_um(x, 0.4, 0.95, 0.06)
    raw = drift + bumps + rng.normal(0.0, 0.04, size=x.shape)
    raw = raw / float(np.max(raw))  # y as a fraction of the trace maximum, like every guide figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig = _base_layout(fig, "F · Not evaluable: non-physical baseline")
    fig.add_hline(y=0.0, line=dict(color="#9ca3af", width=1, dash="dot"))
    fig.add_annotation(x=0.33, y=0.55, text="zero line", showarrow=False, font=dict(size=10, color="#9ca3af"))
    return fig


def _fig_shoulder_clear(rng: np.random.Generator) -> go.Figure:
    """A detectable shoulder: the cell region sits above the single-peak prediction."""
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    ib = _peak_um(x, 1.0, 0.70, 0.08)
    cell = _peak_um(x, 0.32, 0.95, 0.07)  # shoulder riding on the IB right tail
    raw = _noisy(ib + cell, 0.012, rng)
    single = _peak_um(x, 1.0, 0.70, 0.08)  # what one peak (IB only) predicts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=single, name="Single-peak prediction", line=dict(color=_TIGHT_COLOR, width=2.5, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=x, y=cell, name="Shoulder (excess)", line=dict(color=_CELL_COLOR, width=2))
    )
    return _base_layout(fig, "G · A detectable shoulder")


def _fig_shoulder_hidden(rng: np.random.Generator) -> go.Figure:
    """No detectable shoulder: the right tail matches one peak within the noise."""
    x = np.linspace(0.2, 1.2, 400) / _CELL_TARGET_UM
    ib = _peak_um(x, 1.0, 0.70, 0.11)  # broader single peak whose tail reaches the cell region
    raw = _noisy(ib, 0.012, rng)
    single = _peak_um(x, 1.0, 0.70, 0.11)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw, name="Raw (normalized)", line=dict(color=_RAW_COLOR, width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=single, name="Single-peak prediction", line=dict(color=_FIT_COLOR, width=2.5, dash="dash"))
    )
    return _base_layout(fig, "H · No detectable shoulder (below the limit)")


# Plateau-example traces for Example G (see GUIDE_BODY): inspired by a
# similar real case and adapted for the guide. Both axes are normalized
# — x in relative size units, y as a fraction of each trace's maximum —
# and the fits (gaussian + gaussian, autofit) are stored as component
# parameters (height at mode / mode / sigma). Reported metrics are
# scale-invariant.
_PLATEAU_TRACES = {
    "cyc2": {
        "x": [0.2107, 0.2116, 0.2125, 0.2134, 0.2144, 0.2153, 0.2161, 0.2170, 0.2179,
        0.2188, 0.2197, 0.2207, 0.2216, 0.2226, 0.2235, 0.2245, 0.2255, 0.2265, 0.2275,
        0.2284, 0.2294, 0.2303, 0.2312, 0.2322, 0.2332, 0.2342, 0.2352, 0.2362, 0.2372,
        0.2382, 0.2393, 0.2404, 0.2414, 0.2425, 0.2436, 0.2446, 0.2456, 0.2466, 0.2476,
        0.2486, 0.2497, 0.2507, 0.2518, 0.2529, 0.2540, 0.2551, 0.2562, 0.2574, 0.2585,
        0.2597, 0.2609, 0.2621, 0.2633, 0.2643, 0.2654, 0.2665, 0.2676, 0.2687, 0.2699,
        0.2710, 0.2722, 0.2734, 0.2745, 0.2758, 0.2770, 0.2782, 0.2795, 0.2807, 0.2820,
        0.2833, 0.2847, 0.2860, 0.2874, 0.2886, 0.2898, 0.2910, 0.2922, 0.2934, 0.2946,
        0.2959, 0.2971, 0.2984, 0.2997, 0.3010, 0.3023, 0.3037, 0.3050, 0.3064, 0.3078,
        0.3092, 0.3107, 0.3121, 0.3136, 0.3151, 0.3166, 0.3182, 0.3197, 0.3213, 0.3227,
        0.3240, 0.3254, 0.3267, 0.3281, 0.3294, 0.3308, 0.3322, 0.3336, 0.3351, 0.3366,
        0.3380, 0.3395, 0.3411, 0.3426, 0.3442, 0.3458, 0.3474, 0.3490, 0.3507, 0.3523,
        0.3540, 0.3558, 0.3575, 0.3593, 0.3611, 0.3629, 0.3648, 0.3667, 0.3686, 0.3705,
        0.3723, 0.3738, 0.3754, 0.3769, 0.3785, 0.3801, 0.3817, 0.3833, 0.3849, 0.3866,
        0.3883, 0.3900, 0.3917, 0.3935, 0.3952, 0.3970, 0.3989, 0.4007, 0.4026, 0.4045,
        0.4064, 0.4084, 0.4104, 0.4124, 0.4144, 0.4165, 0.4186, 0.4208, 0.4229, 0.4252,
        0.4274, 0.4297, 0.4320, 0.4344, 0.4367, 0.4392, 0.4417, 0.4442, 0.4467, 0.4493,
        0.4520, 0.4547, 0.4566, 0.4585, 0.4604, 0.4623, 0.4642, 0.4661, 0.4681, 0.4701,
        0.4721, 0.4741, 0.4762, 0.4783, 0.4805, 0.4826, 0.4848, 0.4870, 0.4893, 0.4915,
        0.4938, 0.4962, 0.4986, 0.5010, 0.5034, 0.5059, 0.5084, 0.5110, 0.5136, 0.5162,
        0.5189, 0.5216, 0.5244, 0.5272, 0.5300, 0.5329, 0.5359, 0.5389, 0.5419, 0.5450,
        0.5482, 0.5514, 0.5547, 0.5580, 0.5614, 0.5648, 0.5684, 0.5719, 0.5756, 0.5793,
        0.5831, 0.5870, 0.5909, 0.5949, 0.5990, 0.6032, 0.6075, 0.6119, 0.6164, 0.6210,
        0.6256, 0.6304, 0.6353, 0.6403, 0.6445, 0.6471, 0.6497, 0.6524, 0.6551, 0.6578,
        0.6606, 0.6634, 0.6662, 0.6691, 0.6720, 0.6750, 0.6779, 0.6810, 0.6841, 0.6872,
        0.6903, 0.6935, 0.6968, 0.7001, 0.7034, 0.7068, 0.7102, 0.7137, 0.7172, 0.7208,
        0.7245, 0.7282, 0.7319, 0.7357, 0.7396, 0.7436, 0.7476, 0.7516, 0.7558, 0.7600,
        0.7643, 0.7686, 0.7730, 0.7775, 0.7821, 0.7868, 0.7915, 0.7963, 0.8013, 0.8063,
        0.8114, 0.8166, 0.8219, 0.8273, 0.8329, 0.8385, 0.8443, 0.8501, 0.8561, 0.8622,
        0.8685, 0.8749, 0.8814, 0.8881, 0.8950, 0.9020, 0.9092, 0.9165, 0.9240, 0.9318,
        0.9397, 0.9478, 0.9561, 0.9647, 0.9735, 0.9825, 0.9918, 1.0013, 1.0112, 1.0213,
        1.0318, 1.0426, 1.0537, 1.0652, 1.0770, 1.0893, 1.1020, 1.1152, 1.1288, 1.1430,
        1.1577, 1.1729, 1.1889, 1.2054, 1.2227, 1.2408, 1.2596],
        "y": [-0.0000, -0.0013, -0.0008, 0.0024, 0.0008, 0.0053, 0.0050, 0.0085, 0.0101,
        0.0098, 0.0127, 0.0135, 0.0141, 0.0176, 0.0178, 0.0196, 0.0240, 0.0228, 0.0259,
        0.0268, 0.0297, 0.0295, 0.0290, 0.0301, 0.0332, 0.0329, 0.0339, 0.0343, 0.0341,
        0.0351, 0.0370, 0.0387, 0.0406, 0.0437, 0.0442, 0.0439, 0.0448, 0.0430, 0.0426,
        0.0421, 0.0449, 0.0457, 0.0465, 0.0486, 0.0482, 0.0472, 0.0483, 0.0487, 0.0490,
        0.0509, 0.0511, 0.0498, 0.0506, 0.0504, 0.0498, 0.0493, 0.0496, 0.0489, 0.0476,
        0.0478, 0.0483, 0.0485, 0.0495, 0.0493, 0.0504, 0.0480, 0.0468, 0.0472, 0.0453,
        0.0446, 0.0451, 0.0441, 0.0445, 0.0455, 0.0436, 0.0434, 0.0438, 0.0428, 0.0433,
        0.0434, 0.0440, 0.0436, 0.0409, 0.0401, 0.0409, 0.0415, 0.0402, 0.0399, 0.0389,
        0.0381, 0.0369, 0.0367, 0.0350, 0.0345, 0.0344, 0.0344, 0.0346, 0.0344, 0.0340,
        0.0317, 0.0313, 0.0306, 0.0309, 0.0301, 0.0294, 0.0291, 0.0284, 0.0275, 0.0272,
        0.0266, 0.0270, 0.0266, 0.0262, 0.0252, 0.0236, 0.0224, 0.0231, 0.0235, 0.0235,
        0.0231, 0.0238, 0.0240, 0.0244, 0.0231, 0.0215, 0.0216, 0.0216, 0.0225, 0.0232,
        0.0228, 0.0228, 0.0220, 0.0211, 0.0218, 0.0223, 0.0227, 0.0237, 0.0243, 0.0246,
        0.0257, 0.0265, 0.0264, 0.0264, 0.0274, 0.0298, 0.0301, 0.0306, 0.0320, 0.0324,
        0.0333, 0.0347, 0.0369, 0.0381, 0.0392, 0.0419, 0.0439, 0.0459, 0.0478, 0.0494,
        0.0517, 0.0547, 0.0578, 0.0604, 0.0626, 0.0652, 0.0685, 0.0714, 0.0745, 0.0775,
        0.0812, 0.0838, 0.0859, 0.0893, 0.0932, 0.0963, 0.0998, 0.1037, 0.1071, 0.1106,
        0.1138, 0.1175, 0.1213, 0.1258, 0.1303, 0.1345, 0.1390, 0.1436, 0.1482, 0.1534,
        0.1579, 0.1634, 0.1694, 0.1752, 0.1807, 0.1877, 0.1945, 0.2014, 0.2082, 0.2163,
        0.2237, 0.2320, 0.2405, 0.2492, 0.2586, 0.2674, 0.2773, 0.2884, 0.2995, 0.3108,
        0.3227, 0.3348, 0.3474, 0.3606, 0.3738, 0.3881, 0.4028, 0.4177, 0.4345, 0.4515,
        0.4687, 0.4865, 0.5047, 0.5243, 0.5444, 0.5645, 0.5857, 0.6072, 0.6297, 0.6525,
        0.6760, 0.6994, 0.7230, 0.7472, 0.7671, 0.7795, 0.7918, 0.8037, 0.8155, 0.8275,
        0.8398, 0.8520, 0.8634, 0.8747, 0.8866, 0.8983, 0.9097, 0.9202, 0.9305, 0.9400,
        0.9488, 0.9574, 0.9653, 0.9728, 0.9792, 0.9851, 0.9902, 0.9945, 0.9977, 0.9995,
        1.0000, 0.9990, 0.9965, 0.9925, 0.9876, 0.9816, 0.9740, 0.9649, 0.9544, 0.9423,
        0.9287, 0.9142, 0.8991, 0.8832, 0.8667, 0.8493, 0.8307, 0.8119, 0.7927, 0.7740,
        0.7549, 0.7354, 0.7158, 0.6960, 0.6764, 0.6563, 0.6360, 0.6156, 0.5955, 0.5753,
        0.5549, 0.5348, 0.5149, 0.4954, 0.4766, 0.4587, 0.4419, 0.4259, 0.4111, 0.3972,
        0.3843, 0.3714, 0.3587, 0.3455, 0.3309, 0.3152, 0.2980, 0.2791, 0.2579, 0.2350,
        0.2117, 0.1874, 0.1635, 0.1409, 0.1196, 0.0996, 0.0816, 0.0659, 0.0522, 0.0407,
        0.0314, 0.0236, 0.0172, 0.0126, 0.0089, 0.0062, 0.0040],
        "ib": {"h": 0.98067, "mu": 0.7249, "s": 0.11856},
        "cell": {"h": 0.22527, "mu": 0.9816, "s": 0.07682},
        "metrics": {
            "lysis": 0.8705,
            "intact": 0.1295,
            "r2": 0.9947,
            "excess_sigma": 3.65,
            "verdict": "shoulder",
            "robustness": "stable",
        },
    },
    "cyc3": {
        "x": [0.2106, 0.2115, 0.2124, 0.2133, 0.2143, 0.2152, 0.2162, 0.2170, 0.2179,
        0.2188, 0.2197, 0.2206, 0.2215, 0.2225, 0.2234, 0.2244, 0.2254, 0.2264, 0.2274,
        0.2284, 0.2294, 0.2303, 0.2312, 0.2322, 0.2332, 0.2341, 0.2351, 0.2361, 0.2371,
        0.2381, 0.2392, 0.2402, 0.2413, 0.2424, 0.2435, 0.2446, 0.2456, 0.2466, 0.2476,
        0.2486, 0.2496, 0.2507, 0.2517, 0.2528, 0.2539, 0.2550, 0.2561, 0.2572, 0.2584,
        0.2595, 0.2607, 0.2619, 0.2631, 0.2643, 0.2655, 0.2665, 0.2676, 0.2687, 0.2698,
        0.2710, 0.2721, 0.2733, 0.2744, 0.2756, 0.2768, 0.2780, 0.2793, 0.2805, 0.2818,
        0.2831, 0.2844, 0.2857, 0.2871, 0.2884, 0.2898, 0.2910, 0.2922, 0.2934, 0.2946,
        0.2958, 0.2971, 0.2983, 0.2996, 0.3009, 0.3022, 0.3035, 0.3048, 0.3062, 0.3076,
        0.3090, 0.3104, 0.3118, 0.3133, 0.3147, 0.3162, 0.3177, 0.3193, 0.3208, 0.3224,
        0.3240, 0.3254, 0.3267, 0.3280, 0.3294, 0.3308, 0.3321, 0.3336, 0.3350, 0.3364,
        0.3379, 0.3394, 0.3409, 0.3424, 0.3439, 0.3455, 0.3470, 0.3487, 0.3503, 0.3519,
        0.3536, 0.3553, 0.3570, 0.3588, 0.3605, 0.3623, 0.3642, 0.3660, 0.3679, 0.3698,
        0.3717, 0.3737, 0.3754, 0.3769, 0.3785, 0.3800, 0.3816, 0.3832, 0.3848, 0.3865,
        0.3881, 0.3898, 0.3915, 0.3932, 0.3950, 0.3967, 0.3985, 0.4004, 0.4022, 0.4041,
        0.4060, 0.4079, 0.4098, 0.4118, 0.4138, 0.4159, 0.4179, 0.4200, 0.4222, 0.4243,
        0.4265, 0.4288, 0.4311, 0.4334, 0.4357, 0.4381, 0.4405, 0.4430, 0.4455, 0.4480,
        0.4506, 0.4532, 0.4559, 0.4585, 0.4604, 0.4623, 0.4642, 0.4661, 0.4680, 0.4700,
        0.4720, 0.4740, 0.4760, 0.4781, 0.4802, 0.4823, 0.4845, 0.4866, 0.4888, 0.4911,
        0.4934, 0.4957, 0.4980, 0.5004, 0.5028, 0.5052, 0.5077, 0.5102, 0.5127, 0.5153,
        0.5179, 0.5206, 0.5233, 0.5261, 0.5289, 0.5317, 0.5346, 0.5375, 0.5405, 0.5435,
        0.5466, 0.5498, 0.5530, 0.5562, 0.5595, 0.5629, 0.5663, 0.5698, 0.5733, 0.5770,
        0.5807, 0.5844, 0.5883, 0.5922, 0.5962, 0.6002, 0.6044, 0.6087, 0.6130, 0.6174,
        0.6220, 0.6266, 0.6313, 0.6362, 0.6411, 0.6462, 0.6498, 0.6524, 0.6551, 0.6578,
        0.6605, 0.6633, 0.6660, 0.6689, 0.6717, 0.6746, 0.6776, 0.6806, 0.6836, 0.6867,
        0.6898, 0.6929, 0.6961, 0.6993, 0.7026, 0.7059, 0.7093, 0.7127, 0.7162, 0.7197,
        0.7233, 0.7269, 0.7306, 0.7344, 0.7382, 0.7420, 0.7459, 0.7499, 0.7540, 0.7581,
        0.7623, 0.7665, 0.7708, 0.7752, 0.7797, 0.7843, 0.7889, 0.7936, 0.7984, 0.8033,
        0.8083, 0.8134, 0.8186, 0.8238, 0.8292, 0.8347, 0.8403, 0.8460, 0.8518, 0.8578,
        0.8638, 0.8700, 0.8764, 0.8828, 0.8895, 0.8962, 0.9032, 0.9103, 0.9175, 0.9250,
        0.9326, 0.9404, 0.9485, 0.9567, 0.9651, 0.9738, 0.9827, 0.9919, 1.0013, 1.0110,
        1.0210, 1.0313, 1.0419, 1.0528, 1.0641, 1.0758, 1.0878, 1.1003, 1.1132, 1.1266,
        1.1405, 1.1548, 1.1698, 1.1853, 1.2015, 1.2184, 1.2360, 1.2544],
        "y": [0.0000, 0.0003, 0.0028, 0.0009, 0.0021, 0.0035, 0.0068, 0.0070, 0.0067,
        0.0086, 0.0095, 0.0091, 0.0082, 0.0131, 0.0168, 0.0174, 0.0170, 0.0190, 0.0208,
        0.0218, 0.0221, 0.0208, 0.0234, 0.0271, 0.0262, 0.0270, 0.0290, 0.0305, 0.0310,
        0.0330, 0.0327, 0.0320, 0.0314, 0.0336, 0.0370, 0.0369, 0.0371, 0.0379, 0.0381,
        0.0383, 0.0390, 0.0402, 0.0374, 0.0378, 0.0404, 0.0403, 0.0411, 0.0404, 0.0387,
        0.0397, 0.0400, 0.0384, 0.0405, 0.0408, 0.0378, 0.0394, 0.0389, 0.0377, 0.0377,
        0.0376, 0.0373, 0.0377, 0.0373, 0.0377, 0.0369, 0.0351, 0.0341, 0.0346, 0.0339,
        0.0331, 0.0330, 0.0326, 0.0329, 0.0305, 0.0293, 0.0295, 0.0314, 0.0328, 0.0323,
        0.0328, 0.0308, 0.0299, 0.0319, 0.0319, 0.0304, 0.0305, 0.0289, 0.0279, 0.0275,
        0.0268, 0.0269, 0.0262, 0.0247, 0.0237, 0.0206, 0.0214, 0.0227, 0.0221, 0.0211,
        0.0206, 0.0209, 0.0201, 0.0197, 0.0205, 0.0203, 0.0200, 0.0191, 0.0183, 0.0191,
        0.0202, 0.0191, 0.0185, 0.0194, 0.0175, 0.0168, 0.0168, 0.0168, 0.0169, 0.0178,
        0.0166, 0.0161, 0.0172, 0.0172, 0.0178, 0.0184, 0.0178, 0.0167, 0.0165, 0.0169,
        0.0184, 0.0182, 0.0172, 0.0167, 0.0182, 0.0191, 0.0189, 0.0190, 0.0196, 0.0207,
        0.0210, 0.0216, 0.0217, 0.0219, 0.0237, 0.0257, 0.0259, 0.0268, 0.0279, 0.0283,
        0.0296, 0.0307, 0.0320, 0.0346, 0.0362, 0.0363, 0.0375, 0.0402, 0.0426, 0.0440,
        0.0466, 0.0486, 0.0520, 0.0548, 0.0561, 0.0580, 0.0607, 0.0634, 0.0671, 0.0702,
        0.0733, 0.0764, 0.0794, 0.0828, 0.0860, 0.0890, 0.0919, 0.0943, 0.0964, 0.0993,
        0.1027, 0.1064, 0.1099, 0.1142, 0.1183, 0.1224, 0.1262, 0.1303, 0.1344, 0.1396,
        0.1444, 0.1494, 0.1551, 0.1602, 0.1654, 0.1706, 0.1763, 0.1829, 0.1889, 0.1946,
        0.2017, 0.2090, 0.2164, 0.2246, 0.2326, 0.2406, 0.2494, 0.2583, 0.2682, 0.2782,
        0.2883, 0.2986, 0.3096, 0.3210, 0.3333, 0.3465, 0.3594, 0.3729, 0.3873, 0.4020,
        0.4176, 0.4332, 0.4495, 0.4667, 0.4845, 0.5028, 0.5226, 0.5418, 0.5610, 0.5819,
        0.6034, 0.6255, 0.6480, 0.6710, 0.6946, 0.7189, 0.7358, 0.7486, 0.7610, 0.7731,
        0.7858, 0.7982, 0.8110, 0.8236, 0.8356, 0.8480, 0.8605, 0.8724, 0.8838, 0.8952,
        0.9064, 0.9181, 0.9294, 0.9394, 0.9488, 0.9574, 0.9652, 0.9726, 0.9793, 0.9849,
        0.9899, 0.9942, 0.9970, 0.9989, 0.9999, 0.9997, 0.9981, 0.9951, 0.9911, 0.9854,
        0.9785, 0.9713, 0.9631, 0.9540, 0.9437, 0.9327, 0.9218, 0.9101, 0.8979, 0.8857,
        0.8736, 0.8611, 0.8478, 0.8339, 0.8200, 0.8056, 0.7905, 0.7749, 0.7587, 0.7416,
        0.7232, 0.7040, 0.6842, 0.6635, 0.6416, 0.6195, 0.5966, 0.5725, 0.5474, 0.5226,
        0.4982, 0.4733, 0.4479, 0.4221, 0.3965, 0.3707, 0.3448, 0.3190, 0.2928, 0.2665,
        0.2401, 0.2145, 0.1893, 0.1654, 0.1434, 0.1231, 0.1047, 0.0879, 0.0729, 0.0600,
        0.0488, 0.0389, 0.0308, 0.0239, 0.0183, 0.0135, 0.0100, 0.0072],
        "ib": {"h": 0.98024, "mu": 0.742, "s": 0.12484},
        "cell": {"h": 0.20572, "mu": 0.9639, "s": 0.08606},
        "metrics": {
            "lysis": 0.8736,
            "intact": 0.1264,
            "r2": 0.997,
            "excess_sigma": 3.24,
            "verdict": "shoulder",
            "robustness": "stable",
        },
    },
}


def _plateau_gauss(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Normalized gaussian component from (height at mode, mode, sigma)."""
    return p["h"] * np.exp(-0.5 * ((x - p["mu"]) / p["s"]) ** 2)


def _fig_plateau_fits() -> go.Figure:
    """Plateau example, cycle 2 vs cycle 3: same lysis %, visibly different form.

    Axes are normalized (x in relative size units, y as fraction of the trace
    maximum) — see the data-block comment above and Example G in GUIDE_BODY.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=(
            "Cycle 2 — lysis 87.1%",
            "Cycle 3 — lysis 87.4%",
        ),
    )
    for col, cyc in enumerate(("cyc2", "cyc3"), start=1):
        d = _PLATEAU_TRACES[cyc]
        x = np.asarray(d["x"], dtype=float)
        ib = _plateau_gauss(x, d["ib"])
        cell = _plateau_gauss(x, d["cell"])
        show = col == 1
        fig.add_trace(
            go.Scatter(
                x=x, y=np.asarray(d["y"], dtype=float), name="Raw (normalized)",
                line=dict(color=_RAW_COLOR, width=2), showlegend=show,
            ), row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=ib + cell, name="Fit", line=dict(color=_FIT_COLOR, width=2.5, dash="dash"),
                showlegend=show,
            ), row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=cell, name="Cells (shoulder)", line=dict(color=_CELL_COLOR, width=2),
                showlegend=show,
            ), row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=ib, name="IBs", line=dict(color=_IB_COLOR, width=2, dash="dot"),
                showlegend=show,
            ), row=1, col=col,
        )
    fig.update_layout(
        title=dict(text="I · Plateau: identical lysis %, different distribution", font=dict(size=14)),
        xaxis_title="Relative particle size",
        yaxis_title="Signal (normalized)",
        template="plotly_white",
        height=340,
        margin=dict(l=45, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0),
    )
    fig.update_xaxes(range=[0.2, 1.3])
    return fig


def _fig_plateau_overlay() -> go.Figure:
    """The two normalized raw traces superimposed: the sample kept changing."""
    fig = go.Figure()
    for cyc, color, name in (
        ("cyc2", "#1f77b4", "Cycle 2"),
        ("cyc3", "#d62728", "Cycle 3 (one further pass)"),
    ):
        d = _PLATEAU_TRACES[cyc]
        fig.add_trace(
            go.Scatter(
                x=np.asarray(d["x"], dtype=float), y=np.asarray(d["y"], dtype=float),
                name=name, line=dict(color=color, width=2),
            )
        )
    fig = _base_layout(fig, "J · The distributions are not identical — but lysis % is")
    fig.update_xaxes(range=[0.2, 1.3])  # plateau data starts slightly below the shared window
    return fig


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
        "plateau_fits": _fig_plateau_fits(),
        "plateau_overlay": _fig_plateau_overlay(),
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
  <tr><td><code>model</code></td><td>The peak model(s) actually fitted: <code>A + B</code> means model A for the IB peak and model B for the cell peak; a single name means a one-peak fit of that model.</td></tr>
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
A high σ with verdict <span class="tag">indeterminate</span> means the tail bulges but the shape test
did not confirm it — that can also be a main peak whose shape is not lognormal, so do not call it a
shoulder yet.
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
<div class="note">All example plots use <strong>relative axes</strong>, like the
analyzer's <em>Normalize data</em> option: size relative to the cell target size
(the cell peak sits near 1) and signal as a fraction of each trace's maximum.
Examples A–F are <strong>synthetic illustrations</strong> (generated from simple
peak functions), not real measurements — they are designed to show each situation
clearly. The "not evaluable" one (F) is modelled on a real file that showed a
strongly negative, drifting baseline. Example G is <strong>inspired by a similar
real case</strong> and adapted slightly for this guide.</div>

<h3>Example A — Clean two-peak fit (trust the numbers)</h3>
<p>Two well-separated peaks. The gates accept the 2-peak model, R² is high and
the lysis % is reliable.</p>
{{FIGURE:clean}}
<figcaption>Fig. A — IB peak near 0.56, cell peak near 1.0 (relative size).
Raw (grey), fit envelope (blue dashed), cells (green), IBs (orange dotted).</figcaption>

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

<h3>Example G — The plateau: same lysis %, different sample</h3>
<p>This case is <strong>inspired by a similar one from a multi-pass
homogenization campaign</strong> and was adapted slightly for this guide: cycle 2
and cycle 3 of the same material, with <strong>one further homogenization cycle
at significant pressure in between</strong>. As throughout this guide, the axes
are relative (size relative to the cell target, signal normalized to each trace's
maximum); the analysis follows the method publication (workflow and target
settings as described there). Lysis %, R² and the shoulder excess-σ do not depend
on the axis scaling.</p>
{{FIGURE:plateau_fits}}
<figcaption>Fig. I — <strong>Cycle 2 vs cycle 3.</strong> Both are overlap fits
(gaussian + gaussian, R² 0.995 / 0.997, robustness <em>stable</em>). In cycle 2
the shoulder is visually trustworthy: the cell component (green) forms a clear
bulge on the IB tail, and the objective check agrees (verdict
<span class="tag">shoulder</span>, excess 3.7σ). Cycle 3 still passes the check
(3.2σ) and the fit still splits off ~12.6% cells — but by eye the second
component is barely there. That is the <strong>resolution limit of the
method</strong>: detected, yet close to what the fit cannot distinguish from one
slightly asymmetric peak.</figcaption>
<div class="warn"><strong>The punchline:</strong> lysis % is identical across the
extra high-pressure cycle — <strong>87.1%</strong> (cycle 2) vs
<strong>87.4%</strong> (cycle 3). The homogenizer did do something, and the
distribution shows it: the main peak shifted up-size by ~2% and broadened by ~5%,
and the shoulder became less distinct (excess 3.7σ → 3.2σ; the valley between the
populations rose from ~33% to ~43% of the peak height). But the cell-area split —
and with it lysis % — has <strong>saturated</strong>. Near the plateau, a constant
lysis % does <em>not</em> mean "nothing changed".</div>
{{FIGURE:plateau_overlay}}
<figcaption>Fig. J — the same two traces as raw curves: clearly not identical
distributions — cycle 3's main peak sits larger and further up-size — yet the
two-component split assigns both the same lysis %.</figcaption>
<p><strong>How to work at the plateau:</strong> report the <em>trend</em> across
cycles and read the distribution's form — peak position and width, valley depth,
shoulder excess-σ — instead of single point values of lysis %. When late cycles
drop to <span class="tag">indeterminate</span> or <span class="tag">none</span>,
lysis % sits at its floor (≈100%) and carries no further information; what still
moves is the IB peak itself.</p>

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
  <li><strong>One challenging sample in the batch?</strong> Analyze it on its own.
  Sidebar settings apply to <em>all</em> uploaded traces at once, so an adjustment
  that helps a difficult sample (different targets, sensitivity, or width
  relaxation) would also change the fit of every other trace in the process. The
  Analyzer page has its own URL — open it in several browser tabs or windows side
  by side; each tab keeps its own independent uploads and settings. That also
  makes it easy to compare two settings on the same trace live.</li>
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
  <tr><td>Lysis % stops moving across passes while the peak still changes</td><td>Plateau at the resolution limit — read the trend and the peak form (position, width, valley), not the point lysis % (see Example G).</td></tr>
  <tr><td>Trace is negative or drifting</td><td>Not evaluable — try baseline subtraction, else exclude the sample.</td></tr>
</table>

<hr>
<p><small>LysoSense user guide — generated from the app. Re-open anytime with the
<em>📖 Guide</em> button in the sidebar.</small></p>
"""
