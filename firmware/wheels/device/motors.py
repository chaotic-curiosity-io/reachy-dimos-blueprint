"""Motor and mecanum kinematics layer.

Speeds are floats in [-1.0, 1.0]. Positive is forward for that wheel once its
`invert` flag in pins.py is correct.
"""

from machine import Pin, PWM
import time
import pins

_FULL = 65535


class Motor:
    """One L298N channel, driven sign-magnitude.

    Both IN pins are PWM objects. Direction comes from which pin carries the
    duty cycle; magnitude comes from the duty itself. This gives speed control
    without touching ENA/ENB, so the driver boards keep their enable jumpers on.
    """

    def __init__(self, in_a, in_b, invert=False, trim=1.0, freq=pins.PWM_FREQ):
        # Seat both pins LOW as plain outputs before handing them to the LEDC
        # peripheral, so there is no window where they float.
        Pin(in_a, Pin.OUT, value=0)
        Pin(in_b, Pin.OUT, value=0)
        self._a = PWM(Pin(in_a), freq=freq, duty_u16=0)
        self._b = PWM(Pin(in_b), freq=freq, duty_u16=0)
        self.invert = invert
        # Per-wheel output scale. TT motors vary enough that two on the same
        # duty cycle turn at visibly different rates, which shows up as a
        # rotate-in-place that drifts instead of pivoting. Trim evens them out.
        self.trim = trim
        self.speed = 0.0

    def drive(self, speed):
        speed = max(-1.0, min(1.0, float(speed)))
        self.speed = speed          # what was commanded, before trim
        if self.invert:
            speed = -speed
        speed = max(-1.0, min(1.0, speed * self.trim))

        mag = abs(speed)
        if mag < 0.01:
            self.coast()
            return
        # Rescale so the caller's 0..1 maps onto the band where the motor
        # actually turns, instead of a dead zone at the bottom.
        duty = int((pins.MIN_DUTY + (1.0 - pins.MIN_DUTY) * mag) * _FULL)

        if speed > 0:
            self._b.duty_u16(0)
            self._a.duty_u16(duty)
        else:
            self._a.duty_u16(0)
            self._b.duty_u16(duty)

    def coast(self):
        """Both inputs low: the H-bridge floats and the wheel free-spins."""
        self.speed = 0.0
        self._a.duty_u16(0)
        self._b.duty_u16(0)

    def brake(self):
        """Both inputs high: the motor is shorted and stops hard."""
        self.speed = 0.0
        self._a.duty_u16(_FULL)
        self._b.duty_u16(_FULL)


class MecanumDrive:
    """Omnidirectional drive over four mecanum wheels in the standard X layout.

    The body frame is: +vx forward, +vy left, +omega counter-clockwise.
    """

    def __init__(self):
        self.wheels = {}
        for name, cfg in pins.WHEELS.items():
            in_a, in_b = cfg["pins"]
            self.wheels[name] = Motor(in_a, in_b, invert=cfg["invert"],
                                      trim=cfg.get("trim", 1.0))

    # --- core -----------------------------------------------------------
    def move(self, vx=0.0, vy=0.0, omega=0.0, speed=1.0):
        """Inverse kinematics for X-configuration mecanum wheels.

        Any combination of translation and rotation; the result is normalised
        so the fastest wheel sits at `speed` and the others stay proportional.
        """
        fl = vx - vy - omega
        fr = vx + vy + omega
        rl = vx + vy - omega
        rr = vx - vy + omega

        peak = max(abs(fl), abs(fr), abs(rl), abs(rr))
        if peak > 1.0:
            fl, fr, rl, rr = fl / peak, fr / peak, rl / peak, rr / peak

        speed = max(0.0, min(1.0, speed))
        self.wheels["front_left"].drive(fl * speed)
        self.wheels["front_right"].drive(fr * speed)
        self.wheels["rear_left"].drive(rl * speed)
        self.wheels["rear_right"].drive(rr * speed)

    def set_wheels(self, speeds, speed=1.0):
        """Drive each wheel from an explicit mix; unnamed wheels coast.

        `move()` deliberately hides the individual wheels behind the
        kinematics, which is right for driving and wrong for tuning:
        rotate-in-place is the one manoeuvre where each corner's
        contribution matters (every roller scrubs sideways, so a corner
        carrying less load slips and the pivot walks). This is the hook the
        wheel lab drives to find a mix that actually pivots.

        `speeds` maps wheel name -> mix in [-1, 1]; the result is scaled by
        `speed` and NOT renormalised, so what you ask for is what is sent.
        """
        speed = max(0.0, min(1.0, float(speed)))
        for name, motor in self.wheels.items():
            mix = float(speeds.get(name, 0.0))
            motor.drive(max(-1.0, min(1.0, mix)) * speed)

    # --- movement primitives --------------------------------------------
    def forward(self, speed=1.0):      self.move(vx=1.0, speed=speed)
    def reverse(self, speed=1.0):      self.move(vx=-1.0, speed=speed)
    def strafe_left(self, speed=1.0):  self.move(vy=1.0, speed=speed)
    def strafe_right(self, speed=1.0): self.move(vy=-1.0, speed=speed)
    def rotate_ccw(self, speed=1.0):   self.move(omega=1.0, speed=speed)
    def rotate_cw(self, speed=1.0):    self.move(omega=-1.0, speed=speed)

    # Mecanum's party trick: 45-degree travel with two wheels idle.
    def diagonal_fl(self, speed=1.0):  self.move(vx=1.0, vy=1.0, speed=speed)
    def diagonal_fr(self, speed=1.0):  self.move(vx=1.0, vy=-1.0, speed=speed)
    def diagonal_rl(self, speed=1.0):  self.move(vx=-1.0, vy=1.0, speed=speed)
    def diagonal_rr(self, speed=1.0):  self.move(vx=-1.0, vy=-1.0, speed=speed)

    # --- stopping --------------------------------------------------------
    def stop(self):
        for m in self.wheels.values():
            m.coast()

    def brake(self):
        for m in self.wheels.values():
            m.brake()

    def timed(self, action, duration, *args, **kwargs):
        """Run a primitive for `duration` seconds, then always stop.

        The finally-clause is the point: an exception mid-move must not leave
        the chassis driving off the bench.
        """
        try:
            getattr(self, action)(*args, **kwargs)
            time.sleep(duration)
        finally:
            self.stop()

    def state(self):
        return {n: m.speed for n, m in self.wheels.items()}

    def tuning(self):
        """Per-wheel invert/trim as the board currently holds them.

        /tune changes these at runtime; without a readback a UI cannot tell
        a saved trim from one that was only ever typed into a slider.
        """
        return {n: {"invert": m.invert, "trim": m.trim}
                for n, m in self.wheels.items()}
