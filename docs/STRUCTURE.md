# Structure

How this repository is organized and where new work should land. The dataset
provenance ledger lives in [DATASETS.md](DATASETS.md); the narrative of the app
and its method is in the [README](../README.md) and
[USER_GUIDE.md](USER_GUIDE.md).

## Organizing principle

Two zones, by **audience**:

- **Tracked (the product):** the Streamlit app, the `lysosense` package, its
  tests, and the documentation. A fresh clone without `data/` is a fully
  working product; golden tests skip cleanly when the data is absent.
- **Untracked / local (the material):** proprietary measurement data
  (`data/`) and experiment scripts with their outputs (`scripts/`),
  OneDrive-backed, never committed — documented here and in the
  `data/README.md` maps.

Within `data/`, one folder per **dataset/purpose** (campaign or archive);
within `scripts/`, generated outputs go only to `scripts/out/` and parked old
iterations to `scripts/archive_*/`.

## Map

| Path | Tracked | Contents |
|---|---|---|
| `src/lysosense/` | yes | Package: `io.py` (`.dat` parsing), `analysis.py` (peak fitting), `transforms.py`, `_version.py` (version + changelog, single source of truth) |
| `src/lysosense/facs/` | no | Local flow-cytometry subpackage — **not** part of the app, not wired in |
| `app/` | yes | Streamlit entry point, in-app guide, logo assets |
| `tests/` | yes | pytest suite; golden-master tests use `data/CPS-FACS/CPS/Demo` when present, else skip |
| `docs/` | yes | `USER_GUIDE.md`, `DATASETS.md` (data provenance), `STRUCTURE.md` (this file), `notes/` (analysis notes) |
| `data/` | no | Measurement data — see [data/README.md](../data/README.md) and [DATASETS.md](DATASETS.md) |
| `scripts/` | no | Local experiment/diagnostic scripts; `scripts/out/` = generated outputs; `scripts/archive_2025/` = parked 2025 iterations |

## Where new work lands

- **App/package change:** `src/lysosense/` + `tests/` (plus `app/` for UI), and
  bump `src/lysosense/_version.py` (version **and** CHANGELOG) for anything
  user-visible; also `CITATION.cff` on releases (not wired to `_version.py`).
- **New dataset:** `data/<name>/` with a short `README.md`, plus an entry in
  `DATASETS.md`. Never rename raw measurement files — filenames are parsed
  downstream.
- **Exploration/diagnostics:** a script in `scripts/` writing to
  `scripts/out/`; promote lasting conclusions into `docs/notes/`.

## Hygiene

- Generated outputs belong in `scripts/out/` only (regenerable, safe to
  delete); no scratch files elsewhere in the tree.
- Raw instrument data (`.dat`, `.fcs`) is provenance — never delete or rename.
- German-language operator notes in `data/` are provenance: kept verbatim,
  translated in `DATASETS.md`.
- `_version.py` feeds the app's version label, the "What's new?" dialog and
  the XLSX export filenames; `CITATION.cff` must be bumped alongside it.
