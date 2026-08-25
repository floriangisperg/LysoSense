# LysoSense User Guide

This is the written reference for LysoSense. The app also ships an interactive
**📖 Guide** (sidebar) with the same content **plus live example plots** — open
it in a separate tab and keep it beside the analyzer while you work. When the
two disagree, the in-app guide is the more recent one.

LysoSense fits differential centrifugal sedimentation (DCS/CPS) traces to
separate **intact cells** from **inclusion bodies (IBs)** and reports their
relative abundance and **lysis efficiency**. The method follows
Klausser et al., 2025, implemented here as an automated, algorithm-based
workflow.

## Contents

- What LysoSense measures
- About the DCS measurement itself
- Set the IB & cell sizes (important)
- How the fit is decided
- Step-by-step workflow
- Sidebar reference
- Reading the results
- Worked examples
- Tips, gotchas & FAQ
- Troubleshooting

## What LysoSense measures

For cell-disruption work the trace usually contains two populations:

- **Inclusion bodies (IBs)** — small particles released from broken cells
  (default target ~0.48 µm).
- **Intact cells** — larger (default target ~0.85 µm).

The headline output is **lysis efficiency** — the fraction of total signal that
is *not* in the intact-cell peak:

```
lysis_efficiency = 1 − (area of the Cell peak ÷ total area)
```

High lysis ⇒ most cells are disrupted and their contents (IBs) have been
released.

## About the DCS measurement itself

It helps to know what the instrument actually reports. In DCS, particles are
injected into a spinning sucrose gradient and sediment outward; the instrument
times when each size band passes a detector. That sedimentation time is
converted to a size via **Stokes' law** — settling speed depends on diameter²
and on (particle density − fluid density).

Two consequences worth remembering:

- The x-axis is an **equivalent spherical diameter**. The calculation assumes
  spherical particles of a known density. Real cells and IBs are not perfect
  spheres and their density may differ from the assumed value, so **absolute µm
  values are approximate**, not exact.
- DCS is excellent for **relative** comparisons — peak *positions* and, crucially,
  peak *areas* (which lysis % is built on) are reliable within and across runs.

So: trust the shapes and the area ratios; treat the absolute sizes as a guide.

## Set the IB & cell sizes — this is critical

> **The algorithm assumes peaks near the IB and cell target sizes you set.** If
> your instrument, organism or process produces peaks at *different* sizes than
> the defaults (0.48 / 0.85 µm), you **must** update *IB target size* and
> *Cell target size* in the sidebar. With wrong targets the fit can miss a peak,
> attach it to the wrong component, or report a nonsense lysis value (e.g. 0% or
> 100%). Always confirm the peaks sit where the targets say before trusting the
> numbers.

This is especially fragile for **single-peak** traces: the lone peak is assigned
to whichever target it is closest to.

- Lone peak near the **IB** size → reported as **~100% lysis**.
- Lone peak near the **cell** size → reported as **~0% lysis** (i.e. 100% cells).

So a single peak does *not* always mean 100% lysis — it means "whatever
population that peak was assigned to". If that assignment is wrong because the
target sizes are off, the lysis number is wrong.

## How the fit is decided

For every uploaded trace LysoSense tries models in order and keeps the first that
is justified by the data:

1. **Fit one peak.**
2. **Is there a real second peak?** Statistical *gates* decide — the residual
   (signal left after the 1-peak fit) must contain a peak that is prominent
   enough, far enough from the first peak, and that improves the Bayesian
   Information Criterion (BIC).
3. **If yes → two-peak fit.** The peaks must also be locally dominant and
   sufficiently separated.
4. **If the second peak is only a hidden shoulder** (no separate maximum),
   *overlap deconvolution* can still split it — and reports an *area-robustness*
   tag (stable / moderate / uncertain) telling you how trustworthy that split is.

The reported `fit_kind` is one of `one`, `two`, `overlap`.

## Step-by-step workflow

1. **Upload** one or more `.dat` files in *Data Upload*. Each file is one trace;
   upload several to compare them.
2. **Check the peaks sit at the right sizes** and adjust *IB / Cell target size*
   if needed (see above).
3. **(Optional) Preprocess** — baseline subtraction, normalization, or
   restricting the particle-size window.
4. **Pick a peak-detection mode.**
5. **Fit** runs automatically. Use the *Results Table* tab to read lysis % and
   R²; use *Individual Samples* to inspect each fit visually.
6. **Download** the summary or the full experimental data as XLSX.

## Sidebar reference

### Data Upload
Drop `.dat` exports. Use the *Traces to analyze* selector to focus on a subset.

### Peaks & Sample
- **Peak labels** — rename the two components (e.g. *debris*, *aggregates*) for
  separation samples. Names appear in the plots, the results table and the
  downloads. Lysis % is always calculated for the *Cell* peak.
- **IB / Cell target size (µm)** — expected peak centres (defaults 0.48 / 0.85).
  **Adjust these to where your peaks actually sit.** The fit may shift each peak
  within the *Allowed peak shift* window.
- **Limit particle-size range** — restrict the fit to a size window
  (default 0.2–1.2 µm). Keep this on to ignore large debris outside the range of
  interest.

### Data Preprocessing
- **Baseline subtraction** — removes a constant/edge offset. Try only if the
  trace clearly does not return to zero (or goes negative). Methods: *minimum*,
  *percentile* (1st), *linear* (edge fit).
- **Normalize data** — scales every trace to its own maximum, so samples of
  different concentration can be compared. Units become *relative weight*.

### Fitting
- **Peak model** — *autofit* (recommended) tries gaussian / lognormal /
  splitgaussian / gennormal and keeps the best; or pick one, or a different model
  per peak.
- **Relax peak-width constraints** — for genuinely broad peaks (see Example D).
  A tight fit is tried first; widths are relaxed only if R² is poor, so clean
  traces are unaffected.
- **Peak detection**:
  - **Automatic** (default): resolved peaks first, overlap deconvolution only if a
    shoulder is detected.
  - **Resolved peaks only**: stricter — needs a clear second maximum.
  - **Allow overlapping peaks**: forces overlap deconvolution on.
  - **Single peak only**: disables the two-peak fit entirely.

### Advanced
**Sensitivity** presets (Low / Medium / High) set the 2-peak gates together.
*Custom* exposes them individually:

| Gate | What it controls |
| --- | --- |
| Residual prominence / distance / area | How strong a leftover signal must be to count as a second peak. |
| BIC improvement threshold | How much the 2-peak model must beat the 1-peak model. |
| Local dominance | The second peak must "own" some region of the curve. |
| Min 2nd peak area | Smallest area fraction to keep the second peak. |
| Min separation | Peaks must be far enough apart relative to their width. |
| Max Cell peak FWHM / compactness / prominence | Quality constraints on the (usually smaller) second peak. |

**Fitting constraints:** *Allowed peak shift*, *Min 2nd peak fraction*,
*Max peak width* (FWHM cap), and *Peak-top weighting* (0 = ordinary least
squares; higher gives high-signal points more influence).

### Visualization
**View mode:** Combined (raw + fit + components), Fit Overview (components only),
or Raw Data Only. Toggle the fit envelope, components, and a logarithmic size
axis (display only — the fit always runs in linear µm).

**Light / dark theme:** switch via the menu (☰, top-right) → *Settings* →
*Theme* (light, dark, or follow system).

## Reading the results

The **Results Table** tab shows two tables: **Results** (lysis efficiency and the
peak positions, widths and areas it derives from) and **Diagnostics** (how
reliable each fit is). Lysis efficiency leads the Results table.

### Results columns
| Column | Meaning |
| --- | --- |
| `lysis_efficiency` | 1 − cell area ÷ total area. The headline number. |
| `intact_fraction` | cell area ÷ total area (= 1 − lysis). |
| `fit_kind` | one / two / overlap — how many peaks were fitted (a one-peak fit forces lysis to 0% or 100%). |
| `mean_cell_µm` / `mean_ib_µm` | Mean particle size of each component. |
| `fwhm_cell_µm` / `fwhm_ib_µm` | Full-width-at-half-maximum (peak width) of each component. A surprisingly narrow width can flag an over-tight fit — try *Relax peak-width constraints*. |
| `area_cells` / `area_inclusion_bodies` | Integrated area of each component. |
| `area_total` | Sum of the two component areas. |

### Diagnostics columns
| Column | Meaning |
| --- | --- |
| `r_squared` | Goodness of fit. 🟢 ≥0.95 · 🟡 ≥0.90 · 🟠 ≥0.80 · 🔴 <0.80. |
| `fit_quality` | Verbal label for the R² bands (Excellent / Good / Fair / Poor). |
| `model` | Peak model(s) actually fitted: `A + B` = model A for the IB peak and model B for the cell peak; a single name = one-peak fit of that model. |
| `shoulder_verdict` | *shoulder / none / indeterminate / n/a* — an objective check for a second component, independent of the fit. |
| `shoulder_excess_sigma` | How strongly the right tail rises above the best single-peak prediction, in units of the measurement noise (σ). Bigger = more confident a second component is present. This is confidence, **not** shoulder size — the amount of cells is the area / intact fraction. A high σ with verdict *indeterminate* means the tail bulges but the shape test did not confirm it. |
| `area_robustness` | Overlap fits only: *stable / moderate / uncertain* — how much the cell-area estimate moves when overlap settings vary. |

The four tabs: **Overview** (all traces together), **Individual Samples** (one
plot per sample), **Results Table** (the metrics), and **Detailed Information**
(metadata + raw numbers).

## Worked examples

The in-app guide shows these as **live, interactive plots**. The situations:

- **A — Clean two-peak fit (trust the numbers).** Two well-separated peaks; the
  gates accept the 2-peak model, R² is high, lysis % is reliable.
- **B — Overlapping peaks / shoulder (mind the robustness tag).** The cell
  population appears only as a shoulder on the IB peak; overlap deconvolution
  splits it, but the result is sensitive to assumptions — check
  `area_robustness` and treat *uncertain* splits cautiously.
- **C — Lone IB peak → lysis ≈ 100%.** No separate cell peak; the lone peak sits
  near the IB size, so lysis is reported as ~100%. Expected for a fully
  disrupted sample — but see Example E for the opposite case.
- **E — Lone cell peak → lysis ≈ 0% (i.e. 100% cells).** A lone peak near the
  *cell* size is assigned to cells, so lysis is ~0%. A single peak therefore does
  *not* always mean 100% lysis — it depends on which target the peak is closest
  to, which is why the target sizes must be right.
- **D — Broad peak: when to relax widths.** If the default (tight) width bounds
  are narrower than the real peak, the fit underfits the wings and R² drops.
  Turning on *Relax peak-width constraints* lets the peak widen to match the
  data.
- **F — Not evaluable: non-physical baseline.** A strongly negative, drifting
  baseline (mass signal should not be negative) means the measurement or its
  reference subtraction went wrong; any IB/cell fit on top of that is
  meaningless. Try baseline subtraction, check sample prep/dilution, and if it
  stays like this, exclude the sample.
- **G — The plateau: same lysis %, different sample.** Inspired by a similar
  case from a multi-pass homogenization campaign and adapted slightly for this
  guide: two cycles of the same material with one further homogenization cycle
  at significant pressure in between (axes in normalized units; workflow and
  target settings as in the method publication). Cycle 2 shows a visually
  trustworthy shoulder (verdict *shoulder*, 3.7σ); cycle 3 still passes the
  check (3.2σ) but the second component is barely visible — the method's
  resolution limit. The model reports the same lysis (87.1% vs 87.4%) even
  though the distribution clearly changed (main peak ~2% up-size and ~5%
  broader, shoulder excess 3.7σ → 3.2σ). Near the plateau a constant lysis %
  does **not** mean "nothing changed" — report the trend across cycles and the
  peak form (position, width, valley depth), not point values. The in-app guide
  shows the plots of this pair.

## Tips, gotchas & FAQ

- **Single peak is ambiguous.** It is assigned by proximity to the IB or cell
  target: IB-side → ~100% lysis, cell-side → ~0%. If the target sizes are wrong,
  the assignment — and the lysis number — is wrong.
- **Always check target sizes first.** Wrong IB/cell sizes are the most common
  cause of nonsense results.
- **Poor R²?** Try (in order): enable *Relax peak-width constraints*; widen the
  *size range*; check *Sensitivity*; try a different *Peak model*.
- **Uncertain overlap?** Don't trust the split — report the total area, or change
  the detection mode to compare.
- **Compare concentrations?** Turn on *Normalize data* so traces are on the same
  scale.
- **Separation samples?** Rename the peaks (Peak labels) to what they actually
  are.
- **One challenging sample in the batch?** Analyze it on its own. Sidebar
  settings apply to *all* uploaded traces at once, so an adjustment that helps a
  difficult sample (different targets, sensitivity, or width relaxation) would
  also change the fit of every other trace in the process. The Analyzer page has
  its own URL — open it in several browser tabs or windows side by side; each
  tab keeps its own independent uploads and settings. Also handy for comparing
  two settings on the same trace live.
- **Non-physical / negative trace?** Not evaluable — see Example F.

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| All traces failed to parse | Confirm the files are CPS/DCS `.dat` exports. |
| Peaks not where the targets assume | Set *IB / Cell target size* to the real peak positions. |
| Fit is much narrower than the data, low R² | Enable *Relax peak-width constraints*. |
| Expected second peak is missed | Set *Sensitivity → High*, or *Allow overlapping peaks*. |
| Spurious second peak appears | Set *Sensitivity → Low*, or tighten *Max peak width*. |
| Lysis % stops moving across passes while the peak still changes | Plateau at the resolution limit — read the trend and the peak form (position, width, valley), not the point lysis % (see Example G). |
| Trace is negative or drifting | Not evaluable — try baseline subtraction, else exclude the sample. |
