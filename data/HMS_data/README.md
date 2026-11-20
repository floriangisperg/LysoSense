# HMS_data — executed HMS campaign (canonical)

Raw + processed DCS/LysoSense data for the executed high-pressure
homogenization campaign. Full provenance, hashes and downstream contract:
[../../docs/DATASETS.md](../../docs/DATASETS.md).

- `Small EXP 1/2/3/` — lab scale, 5 passes each; executed pressures encoded in filenames
- `Daten Pilot/` — pilot runs V302 (300 bar) / V303 (450 bar), C1–C5 + RBM baseline
- `Small_700/` — V224 fresh subset at 700 bar
- `Bench DEVOPS/` — V224 fresh vs frozen: `Fresh/{CPS,FCS}`, `Frozen/{CPS,FACS}`
- `HPH_experimental_plan.xlsx` — accepted plan

Filenames are parsed by downstream loaders, and the `lysosense_summary.xlsx`
files are consumed by the sibling Homogenisation repo — never rename anything
here without announcing a re-sync of that copy.
