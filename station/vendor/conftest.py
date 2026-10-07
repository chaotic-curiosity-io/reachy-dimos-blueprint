"""Put ``station/vendor`` on ``sys.path`` so the tests import the vendored ``xr_nav``."""

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
