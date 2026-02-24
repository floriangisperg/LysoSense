## LysoSense CPS Analyzer

LysoSense provides a reproducible workflow for analyzing differential centrifugal sedimentation (DCS/CPS) traces collected along E. coli homogenisation campaigns. It parses instrument `.dat` exports, fits bi-peak Gaussian/lognormal models to quantify intact cells and inclusion bodies, and serves interactive overlays via Streamlit. The data-processing strategy is adapted from the method described in [Klausser et al., 2025](https://www.sciencedirect.com/science/article/pii/S0168165625002706).

### Use the Web App
- Open the hosted app: [lysosense.streamlit.app](https://lysosense.streamlit.app/)
- Upload one or more CPS `.dat` files, inspect overlays and component fits, and download the XLSX summary.

### Features
- Parse CPS/DCS `.dat` exports into `particle_size_um` vs `mass_signal_ug`
- Constrained bi-peak fitting (intact cells vs inclusion bodies), with single-peak fallback
- Metrics: component areas, intact fraction, lysis efficiency, mean sizes
- Interactive Plotly overlays and downloadable results table

### Run Locally (for development)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .
set PYTHONPATH=src
streamlit run app\streamlit_app.py
```

### Project Layout
```
+- app/               # Streamlit entry point (uses Plotly for overlays)
+- data/              # Local CPS exports (ignored; keep private)
+- notebooks/         # Exploratory notebooks (e.g., cps_analyzer.ipynb)
+- src/lysosense/     # Installable package: io.py (parsing), analysis.py (fitting)
+- pyproject.toml     # Dependency + metadata definition
+- AGENTS.md          # Contributor notes
+- README.md          # You are here
```


### Citation
If this tool supports your work, please cite:
- Klausser et al., 2025. “Increased purity and refolding yield of bacterial inclusion bodies by recursive high pressure homogenization” [Link](https://www.sciencedirect.com/science/article/pii/S0168165625002706).

Questions or feedback? Open an issue or submit a pull request.
