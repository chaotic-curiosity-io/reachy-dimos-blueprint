"""dimos_scanner — Reachy Mini app that streams camera + accepts motion commands.

The ``DimosScannerApp`` class lives in ``dimos_scanner.app`` (loaded directly
by the Reachy Mini app launcher via the ``reachy_mini_apps`` entry point in
``pyproject.toml``). We deliberately do NOT import it here at module load
time, because it pulls in the heavy ``reachy_mini`` SDK — code that runs the
core helpers (tests, examples/stub_run.py, the CLI ``info``/``check``
commands) shouldn't need that SDK installed.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["DimosScanner"]


def __getattr__(name: str):
    # Lazy re-export: `from dimos_scanner import DimosScanner` works, but
    # only when reachy_mini is actually installed.
    if name == "DimosScanner":
        from .main import DimosScanner

        return DimosScanner
    raise AttributeError(name)
