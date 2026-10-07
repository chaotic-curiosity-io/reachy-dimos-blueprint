"""Bounded continuous command lease; velocity planning is owned by the caller.

Commands refresh before the onboard deadman expires, without intentional stops
between healthy updates. All failures latch stopped until a new instance is made.
Normalized wheel commands are NOT metres/second or radians/second.
"""
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class DriveObservation:
    source_time: float
    segment: str
    tracking: bool
    clearance_m: float
    path_clear: bool


class ContinuousDrive:
    def __init__(self, board, segment, *, seconds=20, speed=.15,
                 clock=time.monotonic, wall_clock=time.time):
        if not math.isfinite(seconds) or not 0 < seconds <= 20:
            raise ValueError('trial must be at most 20 seconds')
        if not math.isfinite(speed) or not 0 < speed <= .15:
            raise ValueError('continuous trial speed must be at most .15')
        self.board, self.segment = board, segment
        self.clock, self.wall_clock = clock, wall_clock
        self.deadline = clock() + seconds
        self.speed = speed
        self.stopped = False
        self.reason = None
        self.commands = 0
        self.next_refresh = 0
        self.last_sent = None

    def stop(self, reason):
        # Latch before I/O, so a failed STOP can never rearm this trial.
        if self.stopped:
            return
        self.stopped, self.reason = True, reason
        try:
            self.board.stop()
        except Exception as exc:
            self.reason += '; STOP response failed: ' + str(exc)

    def tick(self, observation, *, vx=1.0, omega=0.0):
        if self.stopped:
            return False
        now = self.clock()
        age = self.wall_clock() - observation.source_time
        reason = None
        if now >= self.deadline: reason = 'trial time limit'
        elif self.last_sent is not None and now-self.last_sent > .65:
            reason = 'command refresh deadline missed'
        elif not observation.tracking: reason = 'localization lost'
        elif observation.segment != self.segment: reason = 'map segment changed'
        elif not math.isfinite(age) or not 0 <= age <= .8: reason = 'depth is stale'
        elif not math.isfinite(observation.clearance_m) or observation.clearance_m < .65:
            reason = 'obstacle stopping margin'
        elif not observation.path_clear: reason = 'path is not verified clear'
        elif not all(math.isfinite(v) for v in (vx, omega)) or not 0 < vx <= 1 or abs(omega) > .2:
            reason = 'invalid continuous velocity'
        if reason:
            self.stop(reason)
            return False
        if now < self.next_refresh:
            return True
        duration = min(.8, self.deadline - now)
        if duration < .1:
            self.stop('trial time limit')
            return False
        try:
            # The board client supplied by the runner MUST have retries=1.
            if self.board.config.retries != 1:
                raise ValueError('motion retries are prohibited')
            self.board.move(vx=vx, omega=omega, speed=self.speed, duration=duration)
        except Exception as exc:
            self.stop('command failure: ' + str(exc))
            return False
        self.commands += 1
        self.last_sent = now
        elapsed = self.clock() - now
        if elapsed > .5:
            self.stop('command latency exceeded 500 ms')
            return False
        self.next_refresh = now + .2
        return True
