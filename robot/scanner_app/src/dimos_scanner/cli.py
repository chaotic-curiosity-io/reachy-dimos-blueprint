"""Command-line entry point for the scanner app.

Three subcommands, in the usual Reachy Mini app CLI shape:

    dimos-scanner info               Print the resolved config and exit.
    dimos-scanner check              Validate the config + ping the bridge.
    dimos-scanner run [--no-robot]   Start the app. ``--no-robot`` skips the
                                     Reachy Mini SDK so you can rehearse the
                                     bridge handshake without a robot.

Most users never call this directly — the Reachy Mini app launcher invokes
``DimosScannerApp().wrapped_run()`` via the entry-point declared in
``pyproject.toml``. The CLI is for local development and CI smoke checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from .config import BridgeConfig


def _cmd_info(args: argparse.Namespace) -> int:
    cfg = BridgeConfig.from_env()
    print(json.dumps(cfg.__dict__, indent=2))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    cfg = BridgeConfig.from_env()
    if not cfg.host:
        print("error: DIMOS_SCANNER_BRIDGE_HOST not set", file=sys.stderr)
        return 2
    print(f"config: host={cfg.host} port={cfg.port} hz={cfg.frame_hz} q={cfg.jpeg_quality}")

    import websockets

    from .io.protocol import hello, parse_text

    async def probe() -> tuple[bool, str]:
        try:
            async with websockets.connect(cfg.ws_url(), open_timeout=3.0) as ws:
                await ws.send(hello("controller", name="dimos-scanner-check"))
                return True, "connected"
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    ok, msg = asyncio.run(probe())
    print(f"bridge: {'OK' if ok else 'FAIL'} — {msg}")
    return 0 if ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    if args.no_robot:
        print("error: --no-robot is a placeholder for the mocked-SDK path; see examples/stub_run.py", file=sys.stderr)
        return 2
    # Defer the import so `dimos-scanner info` works without reachy_mini installed.
    try:
        from .main import DimosScanner
    except ImportError as e:
        print(f"error: cannot import app — install the [robot] extra: {e}", file=sys.stderr)
        return 2
    app = DimosScanner()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="dimos-scanner", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="print resolved config").set_defaults(func=_cmd_info)
    sub.add_parser("check", help="ping the bridge").set_defaults(func=_cmd_check)
    run_p = sub.add_parser("run", help="launch the Reachy Mini app")
    run_p.add_argument("--no-robot", action="store_true", help="(placeholder — see examples/)")
    run_p.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
