# LysoSense CPS Analyzer

LysoSense provides a reproducible workflow for analyzing differential centrifugal sedimentation (DCS/CPS) traces collected along E. coli homogenisation campaigns. It parses instrument `.dat` exports, fits bi-peak Gaussian/lognormal models to quantify intact cells and inclusion bodies, and serves interactive overlays via Streamlit. The data-processing strategy is adapted from the method described in [Klausser et al., 2025](https://www.sciencedirect.com/science/article/pii/S0168165625002706).

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .
set PYTHONPATH=src
streamlit run app\streamlit_app.py
```
Open the printed localhost URL, upload one or more CPS `.dat` files, and review the particle-size overlays, peak areas, and downloadable XLSX summary.

## Project Layout
```
+- app/               # Streamlit entry point (uses plotly for overlays)
+- data/              # Local CPS exports (ignored; keep private)
+- notebooks/         # Exploratory notebooks (e.g., cps_analyzer.ipynb)
+- src/lysosense/     # Installable package: io.py (parsing), analysis.py (fitting)
+- pyproject.toml     # Dependency + metadata definition
+- AGENTS.md          # Contributor guide
+- README.md          # You are here
```

## Methodology Highlights
- Each `.dat` file is parsed into `particle_size_um` vs. `mass_signal_ug` samples with metadata preserved.
- Analysis fits two constrained peaks (IB vs. intact cells). If the second peak is negligible the model falls back to a single IB peak.
- Key metrics reported: area per component, intact fraction, lysis efficiency, and component mean sizes.
- Plotly overlays display raw traces, fitted envelopes, and component contributions; results can be exported as `lysosense_summary.xlsx`.

## Development Notes
- Follow PEP 8 with 4-space indentation and snake_case names; keep files ASCII-only.
- Add regression tests under `tests/` (pytest recommended) to cover new parsing or fitting logic.
- Before opening a PR, document validation steps such as `streamlit run app\streamlit_app.py` and relevant unit tests.

Questions or contributions? Open an issue or submit a pull request with screenshots/GIFs for UI-facing changes.
