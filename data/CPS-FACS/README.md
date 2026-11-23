# CPS-FACS — CPS measurements of the G7/CMC2 FACS campaign

CPS `.dat` measurements (Demo / Eng1 / Eng2) from the campaign whose paired
FACS raw data (38 `.fcs` + summary deck) lives in the sibling **FlowLysis**
repo (`data/dataset_eng/FACS/`) — the duplicated copy here was removed on
2026-11-23 after byte-identity verification.

`CPS/Demo/` is load-bearing: `tests/test_golden.py` (golden-master metrics)
reads three files from it and skips cleanly when this data is absent.
Provenance details: [../../docs/DATASETS.md](../../docs/DATASETS.md).
