"""Single source of truth for the LysoSense app version and changelog.

Ship a user-visible change by bumping ``__version__`` and prepending an entry
to ``CHANGELOG`` (newest first). The Streamlit app reads both, so the page
version label, the "What's new?" dialog and the exported filenames stay in
sync automatically — there is nothing else to update.

``date`` is intentionally a free-form string so editing this file never needs
to import ``datetime`` or touch the clock at import time.
"""

from __future__ import annotations

__version__ = "1.4.0"

# (version, date, [human-readable change bullets]) — newest release first.
CHANGELOG: list[tuple[str, str, list[str]]] = [
    (
        "1.4.0",
        "2026-07-30",
        [
            "Results table split into a Results table (lysis efficiency, peak positions, widths and areas) and a Diagnostics table (R², fit quality, shoulder verdict, robustness). Lysis efficiency now leads the Results table instead of sitting at the far right.",
            "Peak widths exposed: each fit now reports the full-width-at-half-maximum (FWHM) of the inclusion-body and cell peaks in the Results table and XLSX exports.",
            "Sidebar reorganized: a new 'Peaks & Sample' group puts peak labels, target sizes and the size window up front (the inputs that most affect lysis); model and detection controls are regrouped under 'Fitting'; rarely-used preprocessing is collapsed by default.",
        ],
    ),
    (
        "1.3.0",
        "2026-07-30",
        [
            "Objective shoulder-detection diagnostic: each fit now reports a shoulder verdict (shoulder / none / indeterminate / n/a) and an excess-σ confidence in the run summary, results table, and XLSX exports.",
            "User guide expanded: how to read the shoulder columns, why shoulder/overlap fits are uncertain, and documentation of every sidebar control (peak-model shapes and overlap-deconvolution settings).",
        ],
    ),
    (
        "1.2.0",
        "2026-07-29",
        [
            "Custom peak names: rename the two fitted components in plots, the results table, and XLSX exports.",
            "In-app user guide with example plots, opened in a separate browser window (📖 Guide button).",
        ],
    ),
    (
        "1.1.0",
        "2026-07-27",
        [
            "Optional logarithmic particle-size axis (display only; fit and lysis% unaffected).",
        ],
    ),
    (
        "1.0.0",
        "2026-07-22",
        [
            "First versioned release of the LysoSense CPS/DCS analyzer.",
            "Peak-width relaxation option for fitting broad or overlapping peaks.",
            "Overlap-area robustness metrics added to the results table.",
            "Revamped peak-detection and fitting options in the sidebar.",
        ],
    ),
]
