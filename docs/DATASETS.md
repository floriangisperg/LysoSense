# Datasets

`data/` is gitignored (proprietary measurement data, OneDrive-backed). This
file is the provenance ledger; per-folder quick maps live in the `data/README.md`
files. **Raw instrument files (`.dat`, `.fcs`) are provenance: never rename or
delete them.** Filenames inside `HMS_data/` encode executed conditions and are
parsed by downstream loaders.

All cross-repo claims below were **hash-verified on 2026-11-20**.

## HMS_data — executed HMS campaign (canonical)

Canonical home of the executed high-pressure homogenization campaign
(78 files, ~38 MB):

| Folder | Contents |
|---|---|
| `Small EXP 1/2/3` | Lab scale, 5 passes each; the executed pressure per pass is encoded in the filename (`EXP_<exp>_<pass>_<pressure>_1100.dat`) |
| `Daten Pilot` | Pilot runs V302 (300 bar) and V303 (450 bar), passes C1–C5 plus RBM baseline (`I3_D26I30V30x_PS…_HOMO_C<n>_S_01_1100.dat`) |
| `Small_700` | V224 fresh-biomass subset at 700 bar (2 replicates × pre-homo + cyc1–3) |
| `Bench DEVOPS` | V224 fresh vs frozen biomass: `Fresh/{CPS,FCS}` and `Frozen/{CPS,FACS}` (`.fcs` = raw flow-cytometry events) |
| `HPH_experimental_plan.xlsx` | The accepted experimental plan |

### Operator notes (German, provenance — kept verbatim)

- `Small EXP 3/ACHTUNG_EXP_3_3_4_5 eventuell vertauscht im Probenahmen.txt` —
  "WARNING: EXP 3 samples 3/4/5 possibly swapped in the sampling schedule"
  (empty body; the filename is the note). Operator review concluded passes
  2↔3 are swapped; the Homogenisation loader applies that swap by default
  (`NEW_CELL_BAYES_SMALL3_SWAP_23=1`), which makes the trajectory monotonic.
- `Bench DEVOPS/{Fresh,Frozen}/Immer 700 bar.txt` — "Always 700 bar" (empty
  body; records that the whole block ran at 700 bar).

### Generated summaries inside this tree are processed data

The `lysosense_summary*.xlsx` files (Small EXP 1/2/3, Daten Pilot, Bench
DEVOPS `Fresh/CPS`, Small_700) were produced by the LysoSense app but are
**consumed downstream** and must stay at their exact paths:

- `Homogenisation/src/homogenisation/analysis/newcell/load_hms_lysis.py` reads
  `Small EXP {1,2,3}/lysosense_summary.xlsx` and
  `Daten Pilot/lysosense_summary.xlsx` (sheet `summary`) and parses the
  `measurement` column with the filename patterns above.
- The executed-pressure decoding (e.g. Small EXP 2 pass 2 ran at 200 bar, not
  the planned 300) relies on those filenames.

A local-only script (`scripts/repro_hms.py`, gitignored) re-runs the app
pipeline over this tree and compares against the shipped summaries.

### Cross-repo sync

`Homogenisation/data/HMS/HMS_data/` is a working copy (kept in sync as of
2026-07; **this repo is canonical**). Verified 2026-11-20: all **43 shared
files byte-identical (SHA256)**; their copy deliberately omits `Bench
DEVOPS/` and all `.fcs` (their loader doesn't read them). To re-verify:
compare the SHA256 of every file by relative path under both trees. Anything
added under `Small EXP */`, `Daten Pilot/` or `Small_700/` here requires a
re-sync there and an update to their `docs/DATASETS.md` sync note.

## CPS-FACS — FACS campaign (Demo / Eng1 / Eng2)

~371 MB: `CPS/` (27 `.dat` across `Demo/`, `Eng1/`, `Eng2/`), `FACS/`
(38 `.fcs` raw events files, same three subfolders) and
`FACS/Zusammenfassung FACS Demo_Eng1_Eng2.pptx` (operator summary deck,
"Summary FACS Demo/Eng1/Eng2").

`CPS/Demo/` is **load-bearing**: `tests/test_golden.py` locks end-to-end
parse+fit metrics against three Demo files (golden-master values) and skips
cleanly when `data/` is absent (fresh CI clones).

## cps_archive_2024_2025 (renamed from `test-data/` on 2026-11-20)

89 raw CPS `.dat` exports plus one instrument control file (`export.ctr`)
from 2024–2025 campaigns (G7 series, V301–V307 pilots, D23–D25 series).
**Referenced by no code** (checked across `src/`, `app/`, `tests/` and
`scripts/` on 2026-11-20); kept purely as measurement provenance.

## rollout-demo — curated demo bundle

8 `.dat` files: `bl21/` duplicates 4 files of `CPS-FACS/CPS/Eng1`, `hms/`
duplicates 4 files of `HMS_data/Bench DEVOPS/Fresh/CPS` (V224 FRESH_3). All
8 verified byte-identical to their sources on 2026-11-20 — regenerable from
those folders, safe to delete and re-copy if space is needed.
