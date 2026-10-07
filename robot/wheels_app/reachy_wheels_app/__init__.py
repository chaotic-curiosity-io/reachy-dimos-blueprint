"""Reachy Mini app for driving the Wheels mecanum chassis.

Import safety contract: importing this package must not pull in the
``reachy_mini`` SDK — only ``main`` does, and only when the daemon loads the
app. Everything else (client, API wiring, config) is testable offline.
"""

__all__ = ["__version__"]
__version__ = "0.5.0"
