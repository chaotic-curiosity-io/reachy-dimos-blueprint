"""Reachy Mini -> dimOS station server (single CLI).

One Python process, shaped like dimOS's ``run_camera_pipeline.py`` invocation::

    python -m station.dimos_bridge.server --dimos-dir /path/to/dimos \
        --depth depthpro --pose external --display-width 768 --max-fps 2.0

What it does:

  1. Starts the WebSocket broker on ``--ws-port`` and waits for the on-robot
     ``dimos_scanner`` app (``robot/scanner_app``) to connect and stream JPEG
     frames + head-pose (forward kinematics) samples.
  2. Reads arrow keys / WASD from this terminal and forwards them to the robot
     as head motion commands.
  3. Once a frame arrives, runs the dimOS fork's spatial-perception pipeline
     (``mac_iphone_spatial_foxglove.py``: depth -> pose -> ObjectDB ->
     SpatialMemory) against a live ``NetworkVideoSource``. The fork's script is
     loaded as an importable module and its ``VideoSource`` is swapped for the
     network source — **no edits to the dimOS checkout are needed**.

The dimOS checkout is located by ``--dimos-dir`` or ``$DIMOS_DIR`` (required).
Visualisation is whatever the pipeline is told via ``--extra``: ``--viz rerun``
for the Rerun viewer, ``--viz foxglove`` (the fork's default) for Foxglove on
``ws://localhost:8765``, or ``--viz both``. See ``station/README.md`` for the dependency story.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

# Allow running either as ``python -m station.dimos_bridge.server`` (from the
# repo root) or as ``python station/dimos_bridge/server.py``.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# The xr_nav modules the dimOS pipeline imports (voxel map, ICP, map IO,
# keyframes, relocalization), vendored so the fork's submodule isn't needed.
_VENDOR = _REPO / "station" / "vendor"

from station.dimos_bridge.bridge import Bridge  # noqa: E402
from station.dimos_bridge.frames import T_HEAD_CAM as _T_HEAD_CAM  # noqa: E402
from station.dimos_bridge.imu_foxglove import ImuFoxgloveServer  # noqa: E402
from station.dimos_bridge.protocol import (  # noqa: E402
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
)

logger = logging.getLogger("dimos_bridge.server")


# ---------------------------------------------------------------------- #
# Terminal arrow-key reader                                               #
# ---------------------------------------------------------------------- #


KEY_TO_ACTION = {
    "a": "yaw_left",   "d": "yaw_right",
    "w": "pitch_up",   "s": "pitch_down",
    "r": "reset",
}
# Arrow keys arrive as 3-byte escape sequences "\x1b[A" etc.
ARROW_TO_ACTION = {"A": "pitch_up", "B": "pitch_down", "C": "yaw_right", "D": "yaw_left"}


def _read_keys_loop(bridge: Bridge, stop_event: threading.Event, step_deg: float) -> None:
    """Blocking thread that puts stdin in cbreak mode and pumps arrow keys
    into bridge.send_control."""
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        logger.warning("stdin is not a tty — keyboard control disabled")
        return
    try:
        tty.setcbreak(fd)
        print("[server] keyboard ready — arrows / WASD pan+tilt, r=reset, +/-=step, q=quit")
        while not stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if len(seq) == 2 and seq[0] == "[" and seq[1] in ARROW_TO_ACTION:
                    bridge.send_control(ARROW_TO_ACTION[seq[1]], step_deg=step_deg)
                    continue
                stop_event.set()
                break
            if ch in ("q", "\x03", "\x04"):
                stop_event.set()
                break
            if ch in KEY_TO_ACTION:
                bridge.send_control(KEY_TO_ACTION[ch], step_deg=step_deg)
            elif ch == "+":
                step_deg = min(30.0, step_deg + 1.0)
                print(f"[server] step = {step_deg:.1f}°")
            elif ch == "-":
                step_deg = max(0.5, step_deg - 1.0)
                print(f"[server] step = {step_deg:.1f}°")
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------- #
# Dimos pipeline runner (import the fork's script + swap VideoSource)     #
# ---------------------------------------------------------------------- #


def _resolve_depth_model(
    cli_depth: str | None,
    cli_da3_model: str | None,
    robot_pref: str | None,
) -> tuple[str, str]:
    """Pick the dimos ``--depth`` + ``--da3-model`` pair.

    Precedence: explicit CLI flags > robot's settings-page preference > default
    ``da3-small``. Returns ``(depth, da3_model)`` where ``depth`` is
    ``"depthpro"`` or ``"da3"`` and ``da3_model`` is the size variant when
    applicable (ignored by dimos when depth=depthpro).
    """
    # Robot sends one combined string, e.g. "da3-small", "da3metric-large", "depthpro".
    if cli_depth is not None:
        depth = cli_depth
        da3 = cli_da3_model or "da3-small"
    elif robot_pref == "depthpro":
        depth, da3 = "depthpro", "da3-small"
    elif robot_pref and robot_pref.startswith("da3"):
        depth, da3 = "da3", robot_pref
    else:
        depth, da3 = "da3", "da3-small"
    return depth, da3


def _watch_robot_model_change(
    bridge,  # type: ignore[no-untyped-def]
    current_model: str,
    stop_event: threading.Event,
) -> None:
    """Log a clear message whenever the robot reconnects with a different
    depth_model. The pipeline can't be hot-swapped — operator needs to
    restart the server to actually apply the change.
    """
    seen = current_model
    while not stop_event.is_set():
        # Block until a fresh robot hello arrives.
        bridge._robot_config_event.clear()  # noqa: SLF001
        if not bridge._robot_config_event.wait(timeout=1.0):  # noqa: SLF001
            continue
        cfg = dict(bridge._robot_config)  # noqa: SLF001
        new_model = cfg.get("depth_model")
        if new_model and new_model != seen:
            print(
                f"\n[server] !! robot now requests depth_model={new_model!r} "
                f"(currently running with {seen!r}). "
                f"Restart the server to apply.\n",
                flush=True,
            )
            seen = new_model


def _load_dimos_module(dimos_dir: Path, script_name: str):
    script = dimos_dir / script_name
    if not script.exists():
        sys.exit(f"dimos script not found: {script}")
    # The iphone script does ``import dimos`` from inside ``main()``. When the
    # script is run normally (``python run_camera_pipeline.py``), this works
    # because ``cwd`` is set to the dimos root and Python's implicit ``''`` on
    # ``sys.path`` finds the package. We bypass that subprocess hop, so we add
    # the dimos root to ``sys.path`` ourselves and chdir into it for any code
    # downstream that resolves relative paths.
    if str(dimos_dir) not in sys.path:
        sys.path.insert(0, str(dimos_dir))
    os.chdir(dimos_dir)
    spec = importlib.util.spec_from_file_location("dimos_iphone", script)
    if spec is None or spec.loader is None:
        sys.exit(f"could not build import spec for {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dimos_iphone"] = mod
    spec.loader.exec_module(mod)
    # Loading the script put the fork's ``xr-nav/src`` submodule at the front of
    # ``sys.path``, but its ``xr_nav`` imports only run later, inside ``main()``.
    # Re-prepend the vendored copy so it wins whether the submodule is empty (a
    # plain clone of the fork) or populated — every run uses the code that ships.
    sys.path.insert(0, str(_VENDOR))
    return mod, script


# Head -> camera extrinsic (``_T_HEAD_CAM``, imported from ``frames.py`` above),
# mirroring the SDK's ReachyMini.T_head_cam. Rotation: optical (RDF: X-right,
# Y-down, Z-fwd) -> head body (FLU: X-fwd, Y-left, Z-up). The streamed head pose
# is in the body convention, but the dimOS pipeline back-projects depth in the
# OpenCV optical convention — without the rotation a head yaw (about body Z) is
# applied as a ROLL about the optical view axis and the cloud fans into
# overlapping copies. Translation: the camera sits 43.7 mm forward / 51.2 mm
# above the head frame origin; without it every head/body rotation swings the
# real camera on an arc the math ignores and nearby objects smear in the map.


def _make_network_video_source(bridge: Bridge, source_frame_cls,
                               pose_time_offset: float = 0.0):
    """Build a class that conforms to dimos's ``VideoSource`` interface but
    pulls frames from the in-process bridge instead of a video file."""
    # Escape hatches for A/B comparison in Foxglove:
    #   REACHY_POSE_RAW=1       raw head pose, no extrinsic at all (oldest)
    #   REACHY_POSE_NO_LEVER=1  rotation-only extrinsic, lever arm dropped
    #   REACHY_POSE_LATEST=1    pair frames with the latest pose instead of
    #                           interpolating to capture time (pre-fix pairing)
    apply_extrinsic = os.environ.get("REACHY_POSE_RAW", "").lower() not in ("1", "true", "yes")
    use_latest_pose = os.environ.get("REACHY_POSE_LATEST", "").lower() in ("1", "true", "yes")
    t_head_cam = _T_HEAD_CAM.copy()
    if os.environ.get("REACHY_POSE_NO_LEVER", "").lower() in ("1", "true", "yes"):
        t_head_cam[:3, 3] = 0.0

    class NetworkVideoSource:
        def __init__(self, video_path=None, fps_cap: float = 10.0, loop: bool = True):
            self._period = 1.0 / max(fps_cap, 0.1)
            print("[server] waiting for first frame from Reachy ...")
            if not bridge.wait_for_first_frame(timeout=300.0):
                raise RuntimeError(
                    "no frame received within 5 minutes — is the dimos_scanner "
                    "app running on the robot and pointed at this station's IP?"
                )
            probe = bridge.latest_frame()
            if probe is None:
                raise RuntimeError("bridge reported first frame but latest_frame() is None")
            self._H, self._W = probe.shape[:2]
            self._idx = 0
            # Anchor for the streamed head pose: c2w = inv(pose0) @ pose, so the
            # dimos world frame starts at identity (matches VO + the rest of the
            # pipeline). Captured on the first pose seen. Mirrors the replay
            # script's HeadPoseInterpolator.
            self._pose_inv0 = None
            print(
                f"[server] streaming — {self._W}x{self._H} target fps={fps_cap:.1f}"
            )

        @property
        def frame_size(self):
            return self._W, self._H

        def _c2w_from_pose(self, frame_ts: float = 0.0):
            # Interpolate the pose to the frame's capture instant (both stamped
            # on the robot clock). Frames are consumed slower than poses arrive,
            # so "latest pose" belongs to a later instant than the frame —
            # during a pan that misassigns rotation and smears the map.
            # ``pose_time_offset`` absorbs constant camera-pipeline latency
            # (get_frame() returns a buffered frame that is slightly older than
            # its stamp says): positive shifts the query earlier in time.
            pose = None
            if frame_ts > 0.0 and not use_latest_pose:
                pose = bridge.pose_at(frame_ts - pose_time_offset)
            if pose is None:
                pose = bridge.latest_pose()
            if pose is None:
                return None
            # Compose the head->camera extrinsic (optical rotation + lever
            # arm), then anchor to the first sample so the world starts at
            # identity.
            c2w = pose @ t_head_cam if apply_extrinsic else pose
            if self._pose_inv0 is None:
                try:
                    self._pose_inv0 = np.linalg.inv(c2w)
                except np.linalg.LinAlgError:
                    return None
            return self._pose_inv0 @ c2w

        def __iter__(self):
            last_seq = 0
            while True:
                bgr, frame_ts, last_seq = bridge.wait_for_next_frame_ts(
                    last_seq=last_seq, timeout=2.0
                )
                if bgr is None:
                    bgr = bridge.latest_frame()
                    if bgr is None:
                        continue
                yield source_frame_cls(color_bgr=bgr,
                                       ts=frame_ts if frame_ts > 0.0 else time.time(),
                                       frame_idx=self._idx,
                                       c2w=self._c2w_from_pose(frame_ts))
                self._idx += 1
                time.sleep(self._period)

        def close(self) -> None:
            pass

    return NetworkVideoSource


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #


def _print_lan_ips() -> None:
    """Print this station's LAN IPs so the operator knows what to type in the
    dimos_scanner settings page on the robot."""
    import subprocess

    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        # Linux without net-tools (e.g. a DGX Spark): fall back to hostname -I.
        try:
            ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        except FileNotFoundError:
            return
        for ip in ips:
            print(f"           {ip}")
        return
    iface = ""
    for line in out.splitlines():
        if line and not line.startswith((" ", "\t")):
            iface = line.split(":")[0]
        ls = line.strip()
        if ls.startswith("inet ") and "127.0.0.1" not in ls:
            ip = ls.split()[1]
            print(f"           {ip:<18s} (iface {iface})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dimos-style flags (shape matches run_camera_pipeline.py). `--depth` and
    # `--da3-model` default to ``None`` so we can tell whether the operator
    # explicitly passed them on the CLI (in which case they win) or whether
    # we should defer to the robot's settings-page preference.
    parser.add_argument("--depth", default=None, choices=["depthpro", "da3"])
    parser.add_argument(
        "--da3-model", default=None,
        choices=["da3-small", "da3-base", "da3-large", "da3metric-large"],
    )
    parser.add_argument(
        "--pose", default=None, choices=["vo", "identity", "external"],
        help="Pose source. Default (unset): 'external' if the robot streams its "
             "head pose (kinematics — correct under in-place rotation), else 'vo'. "
             "Pass explicitly to force a mode.",
    )
    parser.add_argument("--display-width", type=int, default=768)
    parser.add_argument("--max-fps", type=float, default=2.0)
    parser.add_argument(
        "--device", default=os.environ.get("DIMOS_DEVICE", "mps"),
        choices=["mps", "cuda", "cpu"],
        help="Torch device for the depth model: 'mps' on Apple silicon (default), "
             "'cuda' on an NVIDIA box such as a DGX Spark, 'cpu' as a slow fallback. "
             "Env: DIMOS_DEVICE.",
    )
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument(
        "--mv-window", type=int, default=0,
        help="DA3-only multi-view windowed refinement: frames per window "
             "(0 = off, 6-8 typical). A background worker builds a "
             "cross-frame-consistent /accumulated_cloud_refined map.",
    )
    parser.add_argument(
        "--mv-replace-live", action="store_true",
        help="With --mv-window, save the refined multi-view map as the primary map.",
    )
    parser.add_argument(
        "--pose-time-offset", type=float,
        default=float(os.environ.get("REACHY_POSE_TIME_OFFSET", "0.0")),
        help="Seconds to shift pose lookups earlier than each frame's capture "
             "timestamp, absorbing constant camera-pipeline latency (the SDK's "
             "get_frame() returns a slightly stale buffered frame). Tune by "
             "sweeping on a replay and minimizing wall-plane RMS.",
    )
    parser.add_argument(
        "--no-scan-retain",
        action="store_true",
        help="Disable scan-retention mode. By default (Reachy is a fixed-base "
             "scanner) the map accumulates instead of being carved/pruned, the "
             "voxel grid is densified, and a raw /accumulated_cloud is published "
             "so the 3D scan fills in over time. Pass this to fall back to the "
             "stock moving-robot SLAM behaviour.",
    )
    parser.add_argument(
        "--no-registration",
        action="store_true",
        help="Disable frame-to-map ICP registration. By default the scanner "
             "refines each keyframe's kinematic pose against the accumulated map "
             "to remove residual backlash/servo-lag drift. Pass this to trust the "
             "streamed kinematic pose verbatim.",
    )

    # Station-side knobs.
    parser.add_argument(
        "--ws-host",
        default=os.environ.get("DIMOS_BRIDGE_HOST", DEFAULT_BRIDGE_HOST),
        help="WS bind address (default 0.0.0.0 — accepts robot connections from the LAN).",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=int(os.environ.get("DIMOS_BRIDGE_PORT", DEFAULT_BRIDGE_PORT)),
        help="WS port the robot dials (default %(default)s; scripts/scan.sh uses "
             "9879 because Rerun's gRPC port also defaults to 9876). Env: DIMOS_BRIDGE_PORT.",
    )
    parser.add_argument(
        "--step-deg",
        type=float,
        default=5.0,
        help="Degrees per arrow-key press (live-tunable with +/-).",
    )
    parser.add_argument(
        "--dimos-dir",
        type=Path,
        default=Path(os.environ["DIMOS_DIR"]) if os.environ.get("DIMOS_DIR") else None,
        help="Path to the dimOS fork checkout (required; or set DIMOS_DIR).",
    )
    parser.add_argument(
        "--imu-foxglove-port",
        type=int,
        default=int(os.environ.get("DIMOS_IMU_FOXGLOVE_PORT", 8766)),
        help="Port for the auxiliary Foxglove server that publishes IMU samples. "
             "Set to 0 to disable. Needs the optional foxglove-websocket package.",
    )
    parser.add_argument(
        "--dimos-script",
        default=os.environ.get("DIMOS_SCRIPT", "mac_iphone_spatial_foxglove.py"),
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Pass-through args appended to the dimos argv (e.g. --viz rerun "
             "--save-map out.pkl). Must be the last flag.",
    )

    args = parser.parse_args()
    if args.dimos_dir is None:
        parser.error("--dimos-dir (or DIMOS_DIR) is required: the path to your "
                     "dimOS fork checkout (see station/README.md)")
    args.dimos_dir = args.dimos_dir.expanduser().resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 1. Bridge
    bridge = Bridge(host=args.ws_host, port=args.ws_port)
    bridge.start()

    # 1b. Optional auxiliary Foxglove server for IMU. Runs on its own port so
    # it doesn't collide with dimos's primary one (port 8765 by default).
    # Subscribers fire from the bridge's WS thread → publisher hops onto the
    # Foxglove server's own loop, so the bridge stays responsive.
    imu_fox: ImuFoxgloveServer | None = None
    if args.imu_foxglove_port > 0:
        imu_fox = ImuFoxgloveServer(host="0.0.0.0", port=args.imu_foxglove_port)
        if imu_fox.start():
            bridge.subscribe_imu(imu_fox.publish)
        else:
            imu_fox = None

    print("==[dimOS station server]==========================================")
    print(f"bridge listening at  ws://{args.ws_host}:{args.ws_port}")
    if imu_fox is not None:
        print(f"IMU foxglove        ws://0.0.0.0:{args.imu_foxglove_port}  (topics: /reachy/imu, /reachy/imu/temp)")
    print(f"display_w={args.display_width}  max_fps={args.max_fps}  (pose source resolved below)")
    print("In the dimos_scanner app settings page (http://<robot>:8042/), set the")
    print("bridge host to one of this station's IPs:")
    _print_lan_ips()
    print("==================================================================")

    # 2. Keyboard reader (background thread).
    stop_event = threading.Event()
    kb_thread = threading.Thread(
        target=_read_keys_loop,
        args=(bridge, stop_event, args.step_deg),
        daemon=True,
        name="dimos-bridge-keys",
    )
    kb_thread.start()

    # 3. Wait briefly for the robot to connect so we can honour its
    #    settings-page depth-model preference and learn whether it streams head
    #    pose (CLI flags override if passed).
    need_depth_cfg = args.depth is None or (args.depth == "da3" and args.da3_model is None)
    if need_depth_cfg:
        print("[server] waiting up to 60 s for robot hello to learn depth-model preference ...")
        robot_cfg = bridge.wait_for_robot_config(timeout=60.0)
    elif args.pose is None:
        # Depth is pinned but pose isn't — a short wait is enough to read the
        # robot's pose_stream advertisement and pick external vs vo.
        robot_cfg = bridge.wait_for_robot_config(timeout=5.0)
    else:
        robot_cfg = {}
    depth, da3_model = _resolve_depth_model(
        cli_depth=args.depth,
        cli_da3_model=args.da3_model,
        robot_pref=robot_cfg.get("depth_model"),
    )
    print(
        f"[server] depth model: {depth}"
        + (f"  ({da3_model})" if depth == "da3" else "")
        + ("  (from robot settings)" if not args.depth and robot_cfg.get("depth_model") else "")
    )

    # Resolve pose source. Explicit CLI flag wins; otherwise prefer kinematic
    # pose ('external') when the robot advertised it streams head pose, else VO.
    if args.pose is not None:
        pose_mode = args.pose
    elif robot_cfg.get("pose_stream"):
        pose_mode = "external"
    else:
        pose_mode = "vo"
    print(
        f"[server] pose source: {pose_mode}"
        + ("  (robot streams head pose — kinematic, no VO drift)"
           if pose_mode == "external" else "")
        + ("  (robot did not advertise pose_stream)"
           if pose_mode == "vo" and args.pose is None else "")
    )

    # 4. Load the fork's pipeline script as a module (no main() run yet)
    mod, script_path = _load_dimos_module(args.dimos_dir, args.dimos_script)
    mod.VideoSource = _make_network_video_source(
        bridge, mod.SourceFrame, pose_time_offset=args.pose_time_offset
    )
    # Forward the pipeline's live tracked-object list (name + distance-from-robot
    # + detection count) back to the robot so its web app can render it. The
    # hook is a no-op for standalone dimos runs (sink stays None).
    if hasattr(mod, "OBJECT_LIST_SINK"):
        def _forward_objects(objects: list, ts: float) -> None:
            try:
                bridge.send_objects(objects, ts)
            except Exception:  # noqa: BLE001 — a UI sink must not kill the pipeline
                pass

        mod.OBJECT_LIST_SINK = _forward_objects

    # 5. Build dimos argv. NetworkVideoSource ignores --video so we just pass
    #    /dev/null to satisfy argparse's type=Path.
    dimos_argv = [
        str(script_path),
        "--video", "/dev/null",
        "--depth", depth,
        "--pose", pose_mode,
        "--display-width", str(args.display_width),
        "--max-fps", str(args.max_fps),
        "--device", args.device,
        "--no-loop",
    ]
    if depth == "da3":
        dimos_argv += ["--da3-model", da3_model]
    if args.no_detect:
        dimos_argv.append("--no-detect")
    # Real camera intrinsics advertised in the robot's hello beat the pipeline's
    # calibration-file guess (which may not match the streamed resolution/crop).
    cam_k = robot_cfg.get("camera_K")
    cam_wh = robot_cfg.get("camera_wh")
    if cam_k and len(cam_k) == 9 and cam_wh and len(cam_wh) == 2:
        fx, fy, cx, cy = float(cam_k[0]), float(cam_k[4]), float(cam_k[2]), float(cam_k[5])
        dimos_argv += ["--camera-k", f"{fx},{fy},{cx},{cy},{int(cam_wh[0])},{int(cam_wh[1])}"]
        print(f"[server] camera intrinsics from robot: fx={fx:.1f} fy={fy:.1f} "
              f"@ {int(cam_wh[0])}x{int(cam_wh[1])}")

    # Scan-retention defaults for the fixed-base Reachy scanner: accumulate the
    # map (no carve/prune), densify the voxel grid, and publish a raw
    # /accumulated_cloud so the 3D scan fills in as the robot pans. These go
    # before args.extra so an explicit --extra override still wins (argparse
    # last-value-wins). Disable the whole block with --no-scan-retain.
    if not args.no_scan_retain:
        dimos_argv += [
            "--raycast-every-n", "0",
            "--no-prune",
            "--voxel-size", "0.03",
            "--voxel-min-observations", "1",
            "--accumulate-cloud",
        ]
        print("[server] scan-retain ON: map accumulates + /accumulated_cloud published "
              "(disable with --no-scan-retain)")

    # Frame-to-map ICP registration — on by default for the scanner to absorb
    # residual kinematic pose error; goes before args.extra so an explicit
    # override still wins. Disable with --no-registration.
    if not args.no_registration:
        if "--registration" not in args.extra:
            dimos_argv += ["--registration", "icp"]
            print("[server] frame-to-map ICP registration ON (disable with --no-registration)")

    # Multi-view windowed refinement — opt-in, DA3 only (slower; builds a
    # cross-frame-consistent refined map in a background worker).
    if args.mv_window > 0 and "--mv-window" not in args.extra:
        dimos_argv += ["--mv-window", str(args.mv_window)]
        if args.mv_replace_live:
            dimos_argv += ["--mv-replace-live"]
        print(f"[server] multi-view refinement ON: window={args.mv_window} "
              f"(refined map on /accumulated_cloud_refined"
              f"{', saved as primary' if args.mv_replace_live else ''})")

    dimos_argv += args.extra

    sys.argv = dimos_argv
    print(f"[server] dimos argv: {' '.join(dimos_argv)}\n")

    # Watch for the robot changing its depth_model mid-run.
    running_model = robot_cfg.get("depth_model") or (
        da3_model if depth == "da3" else "depthpro"
    )
    watcher_thread = threading.Thread(
        target=_watch_robot_model_change,
        args=(bridge, running_model, stop_event),
        daemon=True,
        name="dimos-bridge-model-watcher",
    )
    watcher_thread.start()

    # 6. Run dimos's main()
    try:
        mod.main()
    except KeyboardInterrupt:
        print("[server] interrupted")
    finally:
        stop_event.set()
        bridge.stop()
        if imu_fox is not None:
            imu_fox.stop()


if __name__ == "__main__":
    main()
