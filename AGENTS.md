# Repository Guidelines

## Project Structure & Module Organization
- `src/lysosense/`: installable package for CPS/DCS parsing (`io.py`) and bi-peak analysis (`analysis.py`).
- `app/streamlit_app.py`: Streamlit UI; run from repo root with `PYTHONPATH=src` to access the package.
- `data/`: local `.dat` inputs; never commit confidential datasets.
- `notebooks/`: exploratory analyses such as `cps_analyzer.ipynb` used to derive fitting defaults.
- `.venv/` (optional) plus standard config files (`pyproject.toml`) live at the root.

## Build, Test, and Development Commands
```powershell
python -m venv .venv ; .\.venv\Scripts\activate
python -m pip install -e .        # install app + deps in editable mode
set PYTHONPATH=src
streamlit run app\streamlit_app.py  # launch CPS analyzer UI
```
Add regression tests under `tests/` as the library grows (pytest is recommended but not yet configured).

## Coding Style & Naming Conventions
- Follow PEP 8 (4-space indents, snake_case for modules/functions, PascalCase for dataclasses).
- Keep files ASCII-only unless the dataset mandates symbols; prefer explicit units (`um`, `ug`).
- Type hints are required for new public APIs; document intent with short comments where logic is non-obvious.
- Use f-strings, pathlib, and pandas/numpy idioms already established in `lysosense`.

## Testing Guidelines
- Mirror notebook scenarios by loading `.dat` fixtures and asserting metrics (lysis efficiency, peak assignments).
- Name tests `test_<module>_<behavior>()` and place shared fixtures in `tests/conftest.py`.
- Run `pytest` (once added) before submitting changes; target coverage on analysis helpers.

## Commit & Pull Request Guidelines
- Write imperative, present-tense commit subjects (e.g., `Add lognormal peak fitting fallback`).
- Describe the “why” plus notable side-effects in the body; reference issue IDs when relevant.
- PRs should summarize scope, list validation steps (`streamlit run …`, `pytest`), and include screenshots/GIFs for UI tweaks.
- Keep PRs focused (parsing vs. UI vs. docs) to simplify review.

## Security & Configuration Tips
- Never commit raw production datasets; scrub identifiers before sharing.
- Streamlit secrets or external credentials should stay outside the repo; use environment variables when integrating services.
