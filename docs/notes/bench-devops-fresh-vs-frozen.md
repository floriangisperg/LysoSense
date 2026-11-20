---
name: bench-devops-fresh-vs-frozen
description: Bench DEVOPS (V224 fresh vs frozen) CPS analysis — correct targets, and the fresh-climbs/frozen-stalls finding
metadata:
  type: project
---

Dataset: `data/HMS_data/Bench DEVOPS/{Fresh,Frozen}/CPS/` — V224, same pressure, 2 reps × (pre-homo + cyc1/2/3), fresh vs frozen biomass. FACS (SYTO9) sits alongside in `Fresh/FCS` and `Frozen/FACS`.

**Correct analysis targets for this instrument: ib 0.65 / cell 0.95** (NOT the code defaults 0.48/0.85 — those give wrong lysis, because the real peaks sit at ~0.65/0.95). Extends [[hms-data-repro-settings]]. The target SIZE is the dominant lever; `relax_peak_widths` is secondary (only needed to fit broad peaks at wrong targets).

**Finding (CPS, two methods agree):** same pressure, but frozen biomass behaves differently:
- Frozen peaks are broader (FWHM ~0.42 vs ~0.29 at cyc2/3; ~10% broader even at pre-homo).
- Frozen lysis stalls ~70–73% while fresh climbs to ~90–94% (fresh 0→75→88→94, frozen 0→83→78→73). Corroborated model-independently by cell-band area (frozen's cell-sized material persists/grows cyc1→cyc3; fresh's shrinks).
- pre-homo: frozen has a small (~2–5%) low-size (IB-band) signal fresh lacks — a modest buried shoulder from freeze-thaw, below detection, so 0% lysis at pre-homo is defensible.
- At the ~70%+ plateau lysis% is ill-conditioned; read cell-band area / D50 there, not the headline number.

**Open:** identity of frozen's retained cell-band material (intact cells vs aggregates vs damaged-bright) — FACS can't settle without an IB anchor; see [[flowlysis-classifier-needs-ib-anchor]]. Investigation scripts (gitignored) in `scripts/analyze_*{fresh_vs_frozen,facs,settings,targets,lysis}*` + dashboards in `scripts/out/dashboard_*.html`.
