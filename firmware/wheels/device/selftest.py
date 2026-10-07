"""Bench routine to verify the wheel map in pins.py against physical reality.

Which driver channel reaches which corner, and which way each motor is wired,
cannot be known from the schematic alone -- a motor with its two leads swapped
looks identical in software. Run this with the chassis PROPPED UP so the wheels
spin free, watch each wheel, and fix pins.py accordingly.

    >>> import selftest
    >>> selftest.run()
"""

import time
import pins
from motors import MecanumDrive


def run(speed=0.6, on=1.2, off=1.0):
    """Spin one wheel at a time, forward then backward."""
    bot = MecanumDrive()
    print("Prop the chassis up so the wheels spin free. Starting in 3s.")
    time.sleep(3)

    try:
        for name in pins.ORDER:
            motor = bot.wheels[name]
            in_a, in_b = pins.WHEELS[name]["pins"]
            print("\n%s  (GPIO %d / %d)" % (name, in_a, in_b))

            for label, value in (("  forward (+)", speed), ("  reverse (-)", -speed)):
                print(label)
                motor.drive(value)
                time.sleep(on)
                motor.coast()
                time.sleep(off)

            print("  -> did the %s wheel move, and did '+' drive it FORWARD?" % name)
            print("     wrong corner  : swap the pin pairs in pins.WHEELS")
            print("     right corner, backwards : set invert=True for %s" % name)
    finally:
        bot.stop()

    print("\nPer-wheel pass done.")


def primitives(speed=0.6, on=1.5, off=1.2):
    """Run each movement primitive once, on the ground, to confirm kinematics."""
    bot = MecanumDrive()
    moves = ("forward", "reverse", "strafe_left", "strafe_right",
             "rotate_ccw", "rotate_cw", "diagonal_fl", "diagonal_rr")

    print("Put the chassis on the floor with room to move. Starting in 3s.")
    time.sleep(3)
    try:
        for name in moves:
            print("%-14s ->" % name, end=" ")
            bot.timed(name, on, speed=speed)
            print(bot.state())
            time.sleep(off)
    finally:
        bot.stop()
    print("Primitive pass done.")


def demo(speed=None, lead_in=5):
    """Rotate left, rotate right, then translate around. Floor, with clearance.

    Each move announces itself before it runs, so you can watch the chassis
    instead of the terminal and still know what it was asked to do.

        >>> import selftest; selftest.demo()
    """
    speed = pins.DEFAULT_SPEED if speed is None else speed
    bot = MecanumDrive()
    moves = (
        ("ROTATE LEFT   (ccw, in place)", "rotate_ccw",   3.0, 2.0),
        ("ROTATE RIGHT  (cw, in place)",  "rotate_cw",    3.0, 2.0),
        ("FORWARD",                       "forward",      1.5, 1.5),
        ("REVERSE",                       "reverse",      1.5, 2.0),
        ("STRAFE LEFT   (sideways)",      "strafe_left",  2.0, 1.5),
        ("STRAFE RIGHT  (sideways)",      "strafe_right", 2.0, 1.5),
    )
    print("Floor, ~1m clearance. Speed %.2f. Starting in %ds." % (speed, lead_in))
    time.sleep(lead_in)
    try:
        for label, name, run_for, rest in moves:
            print("  %-32s" % label, end=" ")
            bot.timed(name, run_for, speed=speed)
            print("done")
            time.sleep(rest)
    finally:
        bot.stop()
    print("Demo complete.")
