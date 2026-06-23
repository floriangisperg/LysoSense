# LysoSense User Guide

This guide describes the typical workflow for analyzing CPS/DCS `.dat` files with LysoSense.

## 1. Open The App

Open the Streamlit app in your browser:

- local
- hosted: `https://lysosense.streamlit.app/`

## 2. Upload Data

1. Open `Data Upload` in the sidebar.
2. Upload one or more CPS/DCS `.dat` files.
3. The measurements are analyzed automatically after upload.
4. Use `Measurements to show` to choose which samples are displayed in plots and tables.

## 3. Set Preprocessing Options

These settings are in the sidebar under `Data Preprocessing`.

**Baseline subtraction**

Enable this if the measurement has a visible signal offset or sloped baseline. Available methods:

- `minimum`: subtracts the minimum signal value.
- `percentile`: subtracts a robust low signal value.
- `linear`: fits a linear baseline to the edge regions.

Usually, this is not necessary

**Normalize data**

Enable this when comparing samples with different absolute signal heights. The data are scaled to the maximum signal intensity. In normalized plots, the Y-axis is labeled `Rel Weight`.

**Limit particle-size range for fitting**

By default, only a selected particle-size window is fitted. For homogenisation datasets, a range around the expected IB and cell peaks is usually appropriate, for example `0.2` to `1.2 µm`. The range should include both relevant peaks completely while excluding unnecessary edge regions.

## 4. Choose The Model

The main model options are under `Model Settings`.

**Peak model**

Recommended setting: `autofit`.

`autofit` tests multiple peak shapes and selects the best plausible solution. Manual models are mainly useful for testing or special cases:

- `gaussian`: symmetric peak.
- `lognormal`: asymmetric peak with a longer right tail.
- `splitgaussian`: different width on the left and right side of the peak maximum.
- `gennormal`: peak shape that can be rounder or sharper than a normal Gaussian.

**Peak Parameters**

- `IB target size`: expected position of the inclusion-body peak.
- `Cell target size`: expected position of the intact-cell peak.

These values do not need to be exact. They provide initial guidance for the fit.

**Fitting Constraints**

- `Allowed peak shift`: allows peaks to move around the target size. Higher values are more flexible but can make implausible assignments more likely.
- `Min 2nd peak fraction`: minimum area fraction required for a second peak. Higher values suppress very small secondary peaks.
- `Limit max peak width`: prevents overly broad fits. Recommended: enabled.
- `Peak-top weighting`: gives high-signal regions more influence during fitting. The default is a useful compromise when the peak top is more important than small tail deviations.

**2-Peak Detection**

Recommended setting: keep `Use gated 2-peak detection` enabled with sensitivity `Medium (default)`.

- `Low (strict)`: fewer false positives; small secondary peaks are more likely to be rejected.
- `Medium (default)`: recommended default.
- `High (sensitive)`: detects smaller secondary peaks but may be more likely to interpret noise or shoulders as peaks.
- `Custom`: enables manual tuning of the detection criteria.

## 5. Interpret Plots

The app displays these traces:

- `Raw`: measured CPS/DCS distribution.
- `Fit`: sum of the fitted components.
- `Cells`: fitted intact-cell component.
- `IBs`: fitted inclusion-body component.

A good fit follows the raw signal in the relevant peak region without creating obvious peaks that are not present in the data. Small deviations at peak tails or in the valley between two peaks are normal because the model is a simplified description of the real peak shape.

## 6. Results Tabs

**Overview**

Combined view of all selected measurements.

**Individual Samples**

Single-sample plots. This is usually the best view for identifying poor fits.

**Results Table**

Summary of the main metrics:

- `area_cells`: area of the cell peak.
- `area_inclusion_bodies`: area of the IB peak.
- `area_total`: total fitted area.
- `intact_fraction`: fraction of intact cells in the total area.
- `lysis_efficiency`: calculated lysis efficiency.
- `mean_cell_µm`: mean size of the cell peak.
- `mean_ib_µm`: mean size of the IB peak.
- `r_squared`: fit quality. Values close to `1` indicate better agreement.

**Details**

Shows metadata and the first data points of the analyzed traces.

## 7. Export Results

At the bottom of the app, two downloads are available:

- `Download summary (XLSX)`: metrics table for each sample.
- `Download experimental data (XLSX)`: one sheet per sample with raw signal, fit, and components.

## 8. Practical Recommendations

For most homogenisation datasets:

1. Upload the data.
2. Restrict the particle-size range to the relevant region.
3. Use `Peak model = autofit`.
4. Keep `Use gated 2-peak detection` enabled.
5. Start with `Medium (default)`.
6. If a small cell peak is missed, try `High (sensitive)`.
7. If artificial mini-peaks appear, use `Low (strict)` or increase the minimum second-peak fraction.
8. Always inspect the result visually in `Individual Samples`.

## 9. Limitations

LysoSense provides reproducible peak modeling, not a perfect reconstruction of every measured trace. Real CPS/DCS peaks can be asymmetric, overlapping, or slightly deformed. Metrics should therefore be interpreted together with the fitted plots.

For trend analysis and downstream modeling, it is important to use consistent settings across comparable samples.
