"""Runs before main.py. Kills the boot-time motor twitch as early as possible.

Kept deliberately tiny and wrapped in try/except: an exception here would leave
the board in a reset loop with the REPL hard to reach.
"""

try:
    from machine import Pin
    import pins

    for _p in pins.ALL_PINS:
        Pin(_p, Pin.OUT, value=0)
    del _p
except Exception as _e:
    print("boot.py: pin lockdown FAILED:", _e)
