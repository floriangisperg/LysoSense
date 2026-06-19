"""Shared pytest configuration for the LysoSense test suite.

Inserts the ``src`` layout directory onto ``sys.path`` so that
``import lysosense`` resolves without an editable install, keeping the
tests runnable directly from a fresh checkout in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
