# CPS-FACS — FACS campaign (Demo / Eng1 / Eng2)

Paired CPS (`.dat`) and FACS (`.fcs`) measurements for the G7/CMC2 FACS
campaign, one subfolder per study arm; `FACS/Zusammenfassung FACS
Demo_Eng1_Eng2.pptx` is the operator summary deck.

`CPS/Demo/` is load-bearing: `tests/test_golden.py` (golden-master metrics)
reads three files from it and skips cleanly when this data is absent.
Provenance details: [../../docs/DATASETS.md](../../docs/DATASETS.md).
