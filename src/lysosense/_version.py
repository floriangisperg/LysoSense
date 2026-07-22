"""Single source of truth for the LysoSense app version and changelog.

Ship a user-visible change by bumping ``__version__`` and prepending an entry
to ``CHANGELOG`` (newest first). The Streamlit app reads both, so the page
version label, the "What's new?" dialog and the exported filenames stay in
sync automatically — there is nothing else to update.

``date`` is intentionally a free-form string so editing this file never needs
to import ``datetime`` or touch the clock at import time.
"""

from __future__ import annotations

__version__ = "1.0.0"

# (version, date, [human-readable change bullets]) — newest release first.
CHANGELOG: list[tuple[str, str, list[str]]] = [
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
