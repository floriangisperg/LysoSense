---
name: flowlysis-classifier-needs-ib-anchor
description: The FlowLysis reference-anchored FACS classifier needs a clean IB-only anchor; do not apply it to datasets that lack one
metadata:
  type: feedback
---

The FlowLysis reference-anchored 3-state Gaussian classifier (Suppl. S1 of the Homogenization paper; impl in `../05_Projects/FlowLysis/src/flowlysis/h23*_anchored_three_state*.py`) needs **three** anchors built from reference material: intact (RBM/pre-homo), fully-disrupted (IB-only fraction), and damaged (intermediate homogenate). Lysis = 1 − mean p(intact) over events; features = log10(FL1/FSC/SSC +1); optional morphology support from the RBM FSC/SSC 99.5% envelope.

**Why:** Without a true IB-only anchor, substituting a late homogenate (e.g. cyc3) makes the "disrupted" Gaussian contaminated with intact cells → the three Gaussians overlap → the softmax spills intact events onto disrupted/damaged. On the Bench DEVOPS set this produced a biologically impossible ~35% lysis at pre-homo and compressed the fresh/frozen gap.

**How to apply:** Before running it on any FCS dataset, confirm an IB-only fraction exists in that campaign. The morphology filter is NOT the cause of frozen-bias here (fresh-RBM vs pooled vs none all agreed to <1%); the calibration failure is purely the missing IB anchor. Datasets with only pre-homo + cyc1–3 (no IB fraction) cannot use this method. Related: [[bench-devops-fresh-vs-frozen]].
