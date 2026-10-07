"""Pytest path setup for the station tests.

Puts the repository root (for ``station.*`` imports) and ``robot/wheels_app``
(for ``reachy_wheels_app.*`` — the wheels client and depth decoder) on
``sys.path`` so ``pytest station`` works without exporting PYTHONPATH.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "robot" / "wheels_app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
