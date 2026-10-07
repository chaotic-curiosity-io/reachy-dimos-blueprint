"""Runs after boot.py. Brings up the drive layer and hands back the REPL.

Nothing here blocks unless netcfg.AUTOSTART is set. A main.py that loops
forever makes the REPL unreachable, which is a bad place to be while the wheel
map is still unconfirmed.
"""

import netcfg
from motors import MecanumDrive

bot = MecanumDrive()
bot.stop()

print("Mecanum chassis ready. All 8 inputs held LOW.")
print("  bot.forward(0.6) / bot.strafe_left() / bot.rotate_cw()")
print("  bot.move(vx, vy, omega, speed)   bot.timed('forward', 1.5, speed=0.6)")
print("  bot.stop()  bot.brake()  bot.state()")

if netcfg.AUTOSTART:
    import server
    server.run(bot)
else:
    print("  server idle (netcfg.AUTOSTART=False) -> import server; server.run(bot)")
