"""The follow control law — pure, so all of it is testable without a robot.

These tests pin the things that are expensive to discover on hardware: the
sign conventions (a robot that turns the wrong way is not a subtle bug), the
deadbands that stop it oscillating, and the hold → search → give-up ladder
that decides whether a lost target leaves the base rolling.
"""

from __future__ import annotations

import pytest

from reachy_wheels_app.tracking.follow import (
    ARRIVED,
    FOLLOWING,
    HOLDING,
    LOST,
    SEARCHING,
    TIMEOUT,
    FollowConfig,
    FollowController,
    apparent_size,
    estimate_distance_m,
)
from reachy_wheels_app.tracking.types import Detection, FrameInfo

FRAME = FrameInfo(width=640, height=480, hfov_deg=70.0, vfov_deg=55.0)


def box_at(cx, cy, w=120, h=200, name="person", track_id=1):
    return Detection(bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                     score=0.9, class_name=name, track_id=track_id)


def controller(**overrides):
    return FollowController(FollowConfig(**overrides), started_at=0.0)


# --- geometry -------------------------------------------------------------

def test_bearing_sign_is_positive_to_the_right():
    assert FRAME.bearing_deg(640) == pytest.approx(35.0)
    assert FRAME.bearing_deg(0) == pytest.approx(-35.0)
    assert FRAME.bearing_deg(320) == pytest.approx(0.0)


def test_elevation_sign_is_positive_above_centre():
    assert FRAME.elevation_deg(0) == pytest.approx(27.5)
    assert FRAME.elevation_deg(480) == pytest.approx(-27.5)


def test_apparent_size_uses_the_larger_axis():
    tall = box_at(320, 240, w=64, h=240)     # 240/480 = 0.5 beats 64/640
    assert apparent_size(tall, FRAME) == pytest.approx(0.5)


def test_distance_estimate_shrinks_as_the_box_grows():
    near = estimate_distance_m(box_at(320, 240, h=400), FRAME)
    far = estimate_distance_m(box_at(320, 240, h=100), FRAME)
    assert near is not None and far is not None
    assert far > near * 3


def test_distance_estimate_declines_to_guess_without_a_height_prior():
    assert estimate_distance_m(box_at(320, 240, name="frisbee"), FRAME) is None


# --- gaze -----------------------------------------------------------------

def test_head_turns_right_for_a_target_on_the_right():
    # Image bearing is right-positive; robot yaw is left-positive, so a
    # target on the right must produce a NEGATIVE head yaw.
    cmd = controller().step(box_at(600, 240), FRAME, now=1.0)
    assert cmd.head_yaw is not None and cmd.head_yaw < 0
    assert cmd.bearing_deg > 0


def test_head_turns_left_for_a_target_on_the_left():
    cmd = controller().step(box_at(40, 240), FRAME, now=1.0)
    assert cmd.head_yaw is not None and cmd.head_yaw > 0


def test_head_pitches_up_for_a_target_above_centre():
    cmd = controller().step(box_at(320, 60), FRAME, now=1.0)
    assert cmd.head_pitch is not None and cmd.head_pitch > 0


def test_gaze_deadband_leaves_the_head_alone_when_centred():
    cmd = controller().step(box_at(322, 241), FRAME, now=1.0)
    assert cmd.head_yaw is None and cmd.head_pitch is None


def test_head_yaw_is_clamped_to_the_neck_limit():
    cmd = controller(head_yaw_limit=40.0, body_assist_enabled=False).step(
        box_at(60, 240), FRAME, head_yaw=30.0, now=1.0)
    assert cmd.head_yaw == pytest.approx(40.0)


def test_a_head_pinned_at_its_limit_hands_the_turn_to_the_wheels():
    # The neck is out of travel, so the only way to keep facing the target
    # is the base — this is the escalation the tier hierarchy promises.
    cmd = controller(head_yaw_limit=40.0, body_assist_enabled=False).step(
        box_at(60, 240), FRAME, head_yaw=39.5, now=1.0)
    assert cmd.head_yaw is None
    assert cmd.omega > 0


# --- base rotation --------------------------------------------------------

def test_base_rotates_ccw_when_the_head_is_already_looking_left():
    # Head is 30 deg left, target centred in that view: the target is 30 deg
    # left of wheel-forward, so omega must be positive (CCW / left).
    cmd = controller(body_assist_enabled=False).step(box_at(320, 240), FRAME, head_yaw=30.0, now=1.0)
    assert cmd.yaw_error_deg == pytest.approx(30.0)
    assert cmd.omega > 0


def test_base_rotates_cw_for_a_target_off_to_the_right():
    cmd = controller().step(box_at(620, 240), FRAME, now=1.0)
    assert cmd.omega < 0


def test_body_shell_angle_counts_toward_the_wheel_frame_error():
    cmd = controller().step(box_at(320, 240), FRAME, head_yaw=10.0,
                            body_yaw=20.0, now=1.0)
    assert cmd.yaw_error_deg == pytest.approx(30.0)


def test_yaw_deadband_stops_the_base_hunting():
    cmd = controller(yaw_deadband_deg=6.0).step(
        box_at(320, 240), FRAME, head_yaw=3.0, now=1.0)
    assert cmd.omega == 0.0


def test_omega_is_clamped():
    cmd = controller(max_omega=0.5, yaw_gain=1.0).step(
        box_at(0, 240), FRAME, now=1.0)
    assert abs(cmd.omega) == pytest.approx(0.5)


# --- approach -------------------------------------------------------------

def test_drives_forward_when_the_target_is_small_and_on_axis():
    cmd = controller().step(box_at(320, 240, h=100), FRAME, now=1.0)
    assert cmd.vx > 0
    assert cmd.phase == FOLLOWING


def test_does_not_drive_while_badly_off_axis():
    # Way off to the side: turn first, roll second — otherwise the robot
    # drives past the thing it is trying to reach.
    cmd = controller(approach_gate_deg=25.0, body_assist_enabled=False).step(
        box_at(320, 240, h=100), FRAME, head_yaw=39.0, now=1.0)
    assert cmd.vx == 0.0
    assert cmd.omega != 0.0
    assert "turning to face" in cmd.note


def test_arrived_when_aligned_and_at_the_follow_distance():
    cfg = FollowConfig()
    height = cfg.target_size * FRAME.height
    cmd = controller().step(box_at(320, 240, h=height), FRAME, now=1.0)
    assert cmd.phase == ARRIVED
    assert cmd.vx == 0.0 and cmd.omega == 0.0
    assert not cmd.moving


def test_backs_off_when_the_target_crowds_us():
    cmd = controller(allow_reverse=True).step(
        box_at(320, 240, h=470), FRAME, now=1.0)
    assert cmd.vx < 0
    assert "easing back" in cmd.note


def test_reverse_can_be_disabled():
    cmd = controller(allow_reverse=False).step(
        box_at(320, 240, h=470), FRAME, now=1.0)
    assert cmd.vx == 0.0


def test_forward_speed_is_clamped_and_gentle():
    cfg = FollowConfig()
    cmd = controller().step(box_at(320, 240, w=8, h=8), FRAME, now=1.0)
    assert 0 < cmd.vx <= cfg.max_vx
    assert cmd.speed == cfg.drive_speed


def test_rotation_only_moves_use_the_higher_torque_speed():
    cfg = FollowConfig()
    height = cfg.target_size * FRAME.height
    cmd = controller().step(box_at(40, 240, h=height), FRAME, now=1.0)
    assert cmd.vx == 0.0 and cmd.omega != 0.0
    assert cmd.speed == cfg.rotate_speed


def test_distance_cm_maps_onto_an_apparent_size():
    cfg = FollowConfig()
    near = cfg.size_for_distance("person", 1.0, FRAME)
    far = cfg.size_for_distance("person", 3.0, FRAME)
    assert near is not None and far is not None and near > far
    assert cfg.size_for_distance("frisbee", 1.0, FRAME) is None


# --- losing the target ----------------------------------------------------

def test_a_dropped_frame_holds_rather_than_panicking():
    ctrl = controller(hold_seconds=1.0)
    ctrl.step(box_at(320, 240), FRAME, now=1.0)
    cmd = ctrl.step(None, FRAME, now=1.5)
    assert cmd.phase == HOLDING
    assert not cmd.moving and not cmd.stop and not cmd.done


def test_a_longer_gap_starts_a_search_sweep():
    ctrl = controller(hold_seconds=1.0, search_seconds=30.0)
    ctrl.step(box_at(40, 240), FRAME, now=1.0)   # last seen to the LEFT
    cmd = ctrl.step(None, FRAME, now=3.0)
    assert cmd.phase == SEARCHING
    # The requested scan order begins vertically: up, down, left, right.
    assert cmd.head_pitch is not None and cmd.head_pitch > 0
    assert cmd.omega == 0.0
    assert not cmd.done


def test_search_turns_toward_the_side_the_target_left_on():
    # …once the ordered head and shell tiers have completed.
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0)
    ctrl.step(box_at(620, 240), FRAME, now=1.0)  # last seen to the RIGHT
    commands, _ = advance_search(ctrl, 1.25, 30.0)
    assert next(c.omega for c in commands if c.omega) < 0

    other = controller(hold_seconds=0.1, search_seconds=60.0)
    other.step(box_at(40, 240), FRAME, now=1.0)  # last seen to the LEFT
    commands, _ = advance_search(other, 1.25, 30.0)
    assert next(c.omega for c in commands if c.omega) > 0


def test_giving_up_stops_and_ends_the_session():
    ctrl = controller(hold_seconds=1.0, search_seconds=2.0)
    ctrl.step(box_at(320, 240), FRAME, now=1.0)
    cmd = ctrl.step(None, FRAME, now=20.0)
    assert cmd.phase == LOST
    assert cmd.stop and cmd.done


def test_never_seeing_the_target_also_terminates():
    ctrl = controller(hold_seconds=0.5, search_seconds=1.0, acquire_seconds=1.0)
    cmd = ctrl.step(None, FRAME, now=5.0)
    assert cmd.phase == LOST and cmd.done
    assert "never saw" in cmd.note


def test_acquisition_gets_a_longer_budget_than_reacquisition():
    """Finding the target the first time may need a full head sweep, and
    on-robot detection runs near 1 Hz — the short reacquire window gave up
    after about five frames."""
    cfg = dict(hold_seconds=1.0, search_seconds=2.0, acquire_seconds=20.0)

    never_seen = controller(**cfg)
    assert never_seen.step(None, FRAME, now=10.0).phase == SEARCHING

    had_it = controller(**cfg)
    had_it.step(box_at(320, 240), FRAME, now=0.5)
    assert had_it.step(None, FRAME, now=10.0).phase == LOST


def test_the_search_note_says_which_situation_it_is_in():
    ctrl = controller(hold_seconds=0.5, acquire_seconds=20.0)
    assert "looking for" in ctrl.step(None, FRAME, now=2.0).note
    ctrl.step(box_at(320, 240), FRAME, now=3.0)
    assert "reacquir" in ctrl.step(None, FRAME, now=5.0).note


def test_session_time_limit_stops_the_robot():
    ctrl = controller(max_session_seconds=10.0)
    cmd = ctrl.step(box_at(320, 240, h=50), FRAME, now=11.0)
    assert cmd.phase == TIMEOUT
    assert cmd.stop and cmd.done and not cmd.moving


def test_every_moving_command_carries_a_deadman_duration():
    cmd = controller().step(box_at(320, 240, h=100), FRAME, now=1.0)
    assert cmd.moving
    assert 0 < cmd.duration <= 2.0


# --- projection model -----------------------------------------------------

def test_bearing_uses_real_projection_not_the_linear_shortcut():
    """x = f·tan(θ), not (cx/W − 0.5)·HFOV.

    The two agree only at the frame edge. On this robot's ~93° lens the
    linear form understates a centred target's bearing by ~30%, so the
    controller would systematically under-turn in the region a follow loop
    actually lives in.
    """
    frame = FrameInfo(width=1280, height=720, hfov_deg=93.4, vfov_deg=61.7)
    quarter = frame.bearing_deg(960)          # a quarter-frame off centre
    linear = (960 / 1280 - 0.5) * 93.4
    assert quarter == pytest.approx(27.95, abs=0.1)
    assert linear == pytest.approx(23.35, abs=0.1)
    assert quarter > linear * 1.15


def test_bearing_still_equals_half_the_fov_at_the_frame_edge():
    frame = FrameInfo(width=1280, height=720, hfov_deg=93.4)
    assert frame.bearing_deg(1280) == pytest.approx(46.7, abs=0.05)
    assert frame.bearing_deg(0) == pytest.approx(-46.7, abs=0.05)


def test_focal_length_is_consistent_across_both_axes():
    # One lens, one focal length: the vertical FOV must follow from it.
    import math
    frame = FrameInfo(width=1280, height=720, hfov_deg=93.4, vfov_deg=61.7)
    f_from_h = frame.focal_px
    f_from_v = 720 / (2 * math.tan(math.radians(61.7) / 2))
    assert f_from_h == pytest.approx(f_from_v, rel=0.01)


def test_bearing_is_monotonic_and_centred():
    frame = FrameInfo(width=1280, height=720, hfov_deg=93.4)
    xs = [0, 200, 400, 640, 900, 1100, 1280]
    bearings = [frame.bearing_deg(x) for x in xs]
    assert bearings == sorted(bearings)
    assert frame.bearing_deg(640) == pytest.approx(0.0, abs=1e-9)


# --- how the robot is bolted to the chassis -------------------------------

def test_mount_offset_of_zero_is_a_plain_forward_drive():
    from reachy_wheels_app.tracking.types import robot_to_chassis
    vx, vy = robot_to_chassis(1.0, 0.0, 0.0)
    assert (vx, vy) == pytest.approx((1.0, 0.0))


def test_a_robot_facing_the_chassis_right_drives_by_strafing_right():
    """The robot is mounted 90° round, so 'go where I am looking' has to
    leave as a right strafe (chassis vy negative), not a forward drive."""
    from reachy_wheels_app.tracking.types import robot_to_chassis
    vx, vy = robot_to_chassis(1.0, 0.0, -90.0)
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(-1.0)


def test_backing_off_reverses_through_the_same_mounting():
    from reachy_wheels_app.tracking.types import robot_to_chassis
    vx, vy = robot_to_chassis(-0.35, 0.0, -90.0)
    assert vy == pytest.approx(0.35)


def test_mount_offset_preserves_speed():
    import math as _m
    from reachy_wheels_app.tracking.types import robot_to_chassis
    for yaw in (0, -90, 90, 180, 37):
        vx, vy = robot_to_chassis(0.8, 0.0, yaw)
        assert _m.hypot(vx, vy) == pytest.approx(0.8)


# --- tier 2: the body shell ----------------------------------------------

def test_the_shell_takes_over_once_the_head_is_carrying_the_offset():
    """Otherwise the robot tracks a person out of the corner of its eye
    while its body points somewhere else entirely."""
    cmd = controller(body_assist_enabled=True, body_assist_deg=12.0).step(
        box_at(320, 240), FRAME, head_yaw=25.0, now=1.0)
    assert cmd.body_yaw_delta is not None
    assert cmd.body_yaw_delta > 0, "shell must turn toward where the head looks"


def test_a_small_head_offset_leaves_the_shell_alone():
    cmd = controller(body_assist_deg=12.0).step(
        box_at(320, 240), FRAME, head_yaw=5.0, now=1.0)
    assert cmd.body_yaw_delta is None


def test_one_shell_command_is_rate_limited():
    cmd = controller(body_assist_enabled=True, body_assist_deg=5.0, body_step_limit=20.0).step(
        box_at(320, 240), FRAME, head_yaw=39.0, now=1.0)
    assert abs(cmd.body_yaw_delta) <= 20.0


def test_the_shell_never_exceeds_its_cable_limit():
    cmd = controller(body_assist_enabled=True, body_assist_deg=5.0, body_yaw_limit=120.0).step(
        box_at(320, 240), FRAME, head_yaw=30.0, body_yaw=118.0, now=1.0)
    assert cmd.body_yaw_delta is None or cmd.body_yaw_delta <= 2.0


# --- searching for a lost target -----------------------------------------

def search_at(ctrl, t, body_yaw=0.0, head_yaw=0.0, head_pitch=0.0):
    return ctrl.step(None, FRAME, head_yaw=head_yaw, head_pitch=head_pitch,
                     body_yaw=body_yaw, now=t)


def advance_search(ctrl, start, end, *, dt=0.25, posture=None):
    """Run the pure search loop with ideal measured actuator feedback."""
    posture = dict(posture or {"head_yaw": 0.0, "head_pitch": 0.0,
                               "body_yaw": 0.0})
    commands = []
    t = start
    while t <= end + 1e-9:
        cmd = ctrl.step(None, FRAME, now=t, **posture)
        commands.append(cmd)
        if cmd.head_yaw is not None:
            posture["head_yaw"] = cmd.head_yaw
        if cmd.head_pitch is not None:
            posture["head_pitch"] = cmd.head_pitch
        if cmd.body_yaw_delta is not None:
            posture["body_yaw"] += cmd.body_yaw_delta
        t += dt
    return commands, posture


def test_the_search_sweeps_the_head_first():
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    commands, _ = advance_search(ctrl, 0.25, 4.5)
    notes = [c.note for c in commands]
    order = [next(i for i, note in enumerate(notes) if f"head scan {name}" in note)
             for name in ("up", "down", "left", "right")]
    assert order == sorted(order)
    assert all(c.body_yaw_delta is None and c.omega == 0.0 for c in commands)


def test_a_complete_head_scan_precedes_the_shell_reposition():
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    commands, _ = advance_search(ctrl, 0.25, 9.0)
    shell = next(i for i, c in enumerate(commands) if c.body_yaw_delta)
    for name in ("up", "down", "left", "right"):
        assert any(f"head scan {name}" in c.note for c in commands[:shell])
    assert not any(c.omega for c in commands[:shell + 1])


def test_the_wheels_are_the_last_resort_in_a_search():
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    commands, _ = advance_search(ctrl, 0.25, 20.0)
    wheel = next(i for i, c in enumerate(commands) if c.omega)
    shell = next(i for i, c in enumerate(commands) if c.body_yaw_delta)
    second_scan = next(i for i, c in enumerate(commands)
                       if i > shell and "head scan up" in c.note)
    assert shell < second_scan < wheel
    assert not any(c.omega for c in commands[:wheel])


def test_wheel_search_stops_to_scan_the_head_between_sectors():
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0,
                      search_wheel_step_deg=20.0,
                      search_full_turn_seconds=10.0)
    commands, _ = advance_search(ctrl, 0.25, 30.0)
    moving = next(i for i, c in enumerate(commands) if c.omega)
    stopped = next(i for i, c in enumerate(commands[moving + 1:], moving + 1)
                   if c.stop and "sector complete" in c.note)
    assert any("head scan up" in c.note for c in commands[stopped + 1:])


def test_the_search_runs_for_the_configured_thirty_seconds():
    ctrl = controller(hold_seconds=1.0, search_seconds=30.0)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    assert search_at(ctrl, 25.0).phase == SEARCHING
    give_up = search_at(ctrl, 40.0)
    assert give_up.phase == LOST and give_up.done and give_up.stop


def test_the_search_duration_is_adjustable():
    ctrl = controller(hold_seconds=1.0, search_seconds=5.0)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    assert search_at(ctrl, 4.0).phase == SEARCHING
    assert search_at(ctrl, 10.0).phase == LOST


def test_the_default_follow_distance_is_two_metres():
    assert FollowConfig().follow_distance_m == pytest.approx(2.0)


def test_the_search_eventually_sweeps_a_full_circle():
    """Head and shell together only reach ±160°, leaving a blind wedge
    behind the robot. Only the wheels can cover that, so a search that
    never turns the base can never actually look everywhere."""
    cfg = dict(hold_seconds=0.1, search_seconds=90.0,
               search_full_turn_seconds=14.0)
    ctrl = controller(**cfg)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    commands, _ = advance_search(ctrl, 0.25, 70.0)
    assert max(c.search_covered_deg or 0 for c in commands) == pytest.approx(360.0)
    assert sum(bool(c.omega) for c in commands) > 4


def test_the_full_turn_budget_sets_the_spin_rate():
    fast = controller(hold_seconds=0.1,
                      search_full_turn_seconds=6.0, rotate_deg_per_s=80.0)
    slow = controller(hold_seconds=0.1,
                      search_full_turn_seconds=60.0, rotate_deg_per_s=80.0)
    for c in (fast, slow):
        c.step(box_at(320, 240), FRAME, now=0.0)
    fast_commands, _ = advance_search(fast, 0.25, 25.0)
    slow_commands, _ = advance_search(slow, 0.25, 25.0)
    fast_omega = next(abs(c.omega) for c in fast_commands if c.omega)
    slow_omega = next(abs(c.omega) for c in slow_commands if c.omega)
    assert fast_omega > slow_omega


def test_the_search_spin_is_never_faster_than_the_omega_cap():
    ctrl = controller(hold_seconds=0.1,
                      search_full_turn_seconds=1.0, max_omega=0.9)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    commands, _ = advance_search(ctrl, 0.25, 25.0)
    assert max(abs(c.omega) for c in commands) <= 0.18


def test_camera_posture_is_held_while_the_base_searches():
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0)
    ctrl.step(box_at(320, 240), FRAME, now=0.0)
    commands, _ = advance_search(ctrl, 0.25, 25.0)
    frames = [c for c in commands if c.omega]
    assert frames
    assert all(f.head_yaw is None and f.head_pitch is None for f in frames)
    assert all(f.body_yaw_delta is None for f in frames)


def test_legacy_search_speed_cannot_defeat_slow_search_cap():
    ctrl = controller(hold_seconds=0.1, search_seconds=60.0,
                      search_omega=0.4, search_full_turn_seconds=14.0)
    commands, _ = advance_search(ctrl, 0.25, 25.0)
    omega = next(abs(c.omega) for c in commands if c.omega)
    assert 0 < omega <= 0.18


def test_reached_search_pose_is_not_commanded_again():
    ctrl = controller(hold_seconds=0.1)
    first = search_at(ctrl, 0.25)
    reached = search_at(ctrl, 0.5, head_pitch=first.head_pitch)
    assert reached.head_yaw is None and reached.head_pitch is None
    at_limit = search_at(ctrl, 0.75, head_pitch=20.0)
    assert at_limit.head_yaw is None and at_limit.head_pitch is None


def test_unreachable_head_waypoint_times_out_into_the_next_pose():
    ctrl = controller(hold_seconds=0.1, search_waypoint_seconds=0.5)
    commands = [search_at(ctrl, t, head_pitch=5.0)
                for t in (0.25, 0.5, 0.75, 1.0)]
    assert any("head scan down" in c.note for c in commands)


def test_visible_target_does_not_rotate_shell_and_chassis_together_by_default():
    cmd = controller().step(box_at(320, 240), FRAME, head_yaw=30, now=1)
    assert cmd.omega == 0 and cmd.vx == 0
    assert cmd.body_yaw_delta == 6


def test_default_hold_survives_a_slow_detector_gap_without_searching():
    ctrl = controller()
    ctrl.step(box_at(320, 240), FRAME, now=1)
    cmd = ctrl.step(None, FRAME, now=3)
    assert cmd.phase == HOLDING and not cmd.moving


def test_search_head_steps_are_bounded_before_base_rotation():
    ctrl = controller(search_wheels_after=100)
    cmd = ctrl.step(None, FRAME, head_yaw=-30, now=3)
    assert abs(cmd.head_yaw + 30) <= ctrl.config.search_step_limit
