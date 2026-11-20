# LysoSense CPS Analyzer

🌐 **Hosted app:** [lysosense.streamlit.app](https://lysosense.streamlit.app/)  ·  📖 **User guide:** [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

LysoSense is a reproducible workflow for analyzing differential centrifugal
sedimentation (DCS/CPS) traces from *E. coli* homogenisation campaigns. It
parses instrument `.dat` exports, fits bi-peak Gaussian/lognormal models to
quantify **intact cells** and **inclusion bodies**, and serves interactive
overlays, metrics, and downloadable summaries through a Streamlit app.

The data-processing strategy is adapted from the method described in
[Klausser et al., 2025](https://www.sciencedirect.com/science/article/pii/S0168165625002706).

## Use the web app

- Open the hosted app: [lysosense.streamlit.app](https://lysosense.streamlit.app/)
- Upload one or more CPS `.dat` files, inspect overlays and component fits, and
  download the XLSX summary.
- First time? Open the in-app **📖 Guide** in the sidebar for a walkthrough with
  example plots (the written reference is [docs/USER_GUIDE.md](docs/USER_GUIDE.md)).

## Features

- Parse CPS/DCS `.dat` exports into `particle_size_um` vs `mass_signal_ug`
- Constrained bi-peak fitting (intact cells vs inclusion bodies) with single-peak
  fallback, overlap deconvolution, and area-robustness tagging
- Metrics: component areas, intact fraction, **lysis efficiency**, mean sizes, R²
- Custom peak names, multiple peak models, and adjustable detection sensitivity
- Interactive Plotly overlays, a results table, and XLSX exports (summary + experimental data)
- A fully offline-capable in-app guide with example plots

## Run locally (for development)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

The app entry point adds `src` to `sys.path` at import time, so no editable
install or `PYTHONPATH` setup is required.

## Project layout

```
app/                  Streamlit entry point, in-app guide, and logo assets
src/lysosense/        Package: io.py (parsing), analysis.py (fitting), transforms.py
tests/                pytest suite (golden tests use local data when present, else skip)
docs/                 User guide, dataset provenance (DATASETS.md), structure, notes/
data/                 Measurement data — local only (gitignored), canonical for the HMS campaign
scripts/              Local-only experiment scripts and outputs (gitignored)
.github/workflows/    CI (ruff, mypy, pyright on Python 3.13)
requirements.txt      Pinned runtime dependencies
```

## Data (local only)

The proprietary measurement data lives in `data/` (gitignored,
OneDrive-backed). This repo is the **canonical home** of the executed HMS
campaign data (`data/HMS_data/`); the sibling *Homogenisation* repo keeps a
hash-verified working copy (a subset) and consumes the per-experiment
`lysosense_summary.xlsx` exports from it — so nothing inside `HMS_data/` may
be renamed or moved casually. Full provenance ledger, hash-verification
status, operator-note translations and downstream-file contract:
[docs/DATASETS.md](docs/DATASETS.md) · folder maps: [data/README.md](data/README.md).

| Folder | Contents |
|---|---|
| `HMS_data/` | Executed HMS campaign: Small EXP 1/2/3, Daten Pilot (V302/V303), Small_700, Bench DEVOPS (V224 fresh/frozen) — canonical |
| `CPS-FACS/` | FACS campaign CPS + FACS raw (Demo/Eng1/Eng2); `CPS/Demo` backs the golden tests |
| `cps_archive_2024_2025/` | Raw CPS exports from 2024–2025 campaigns (unreferenced provenance archive) |
| `rollout-demo/` | Curated demo bundle (byte-identical subset of the rows above) |

## Citation

If this tool supports your work, please cite **both** the software and the
underlying method:

- **LysoSense** (this repository) — use the **"Cite this repository"** button on
  the GitHub page (generated from [CITATION.cff](CITATION.cff)).
- Klausser et al., 2025. "Increased purity and refolding yield of bacterial
  inclusion bodies by recursive high pressure homogenization."
  [Link](https://www.sciencedirect.com/science/article/pii/S0168165625002706).

## Acknowledgements

Developed in the [CD Laboratory for Inclusion Body Processing 4.0](https://www.tuwien.at/en/cdl/ibp4),
[IBD Group — Integrated Bioprocess Development](https://www.tuwien.at/en/tch/icebe/ibdgroup),
[TU Wien](https://www.tuwien.at/en/). Funded by the
[Christian Doppler Gesellschaft](https://www.cdg.ac.at) and
[Boehringer Ingelheim](https://www.boehringer-ingelheim.com/).

## License

Proprietary — © 2025–2026 Florian Gisperg and TU Wien. All rights reserved. See
[LICENSE](LICENSE). No part of this repository may be copied, modified,
distributed, or used without prior written permission of the copyright holders.

---

Questions or feedback? Open an issue or submit a pull request.
