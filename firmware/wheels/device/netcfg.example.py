# Copy this file to netcfg.py (same folder) and edit it before deploying:
#
#     cp device/netcfg.example.py device/netcfg.py
#
# netcfg.py is git-ignored on purpose -- it holds your WiFi password. The
# deploy scripts refuse to run until it exists. main.py, wifi.py and server.py
# all `import netcfg`, so on the board the file must be named exactly netcfg.py.
"""Network and server configuration.

AP mode is the default because it needs no credentials and works on a bench
with no infrastructure: the chassis becomes its own access point and the
command node joins it. Switch to STA when you want the robot and the agent on
the same LAN.
"""

MODE = "ap"                 # "ap" -> robot hosts its own network
                            # "sta" -> robot joins an existing network

# --- AP mode -----------------------------------------------------------------
# Also used as the WPA2 key for the ranging beacon (see below).
AP_SSID = "mecanum-bot"
AP_PASSWORD = "change-me"   # >=8 chars, WPA2. Change this.

# --- STA mode ----------------------------------------------------------------
# The ESP32 has a 2.4 GHz radio only: a 5 GHz / 6 GHz-only network is invisible
# to it. On a WPA2/WPA3 transition network it joins over the WPA2 half.
# A failed join falls back to AP mode (see wifi.py), so a typo here costs you
# a reconnect to "mecanum-bot", not a USB cable.
STA_SSID = "YOUR-WIFI-SSID"
STA_PASSWORD = "change-me"

PORT = 80

# Shared secret for the file-write and reset endpoints, sent as an `X-Token`
# header. Empty means NO AUTH: anyone who can reach the board can overwrite any
# file on it via POST /put and reboot it via POST /reset. That is tolerable on
# the board's own private AP; on a shared LAN, set a token here and pass the
# same value as TOKEN=... to deploy_wifi.sh / --token to teleop.py.
API_TOKEN = ""

# Deadman timer. Any command without an explicit duration keeps the wheels
# turning for at most this long unless another command arrives. This is what
# stops the chassis when the controlling agent crashes or the link drops --
# without it, the last "forward" command runs until the battery is flat.
COMMAND_TIMEOUT = 2.0       # seconds

# Start the server automatically from main.py. With this True the board is
# usable with no USB cable attached; the serial REPL is then only reachable by
# interrupting with ctrl-C.
AUTOSTART = True

# --- proximity ranging -------------------------------------------------------
# Raise a second, beaconing SSID alongside the STA link so another device can
# measure this board's signal strength with a passive scan. A station on its own
# is invisible to a scan -- it only ever transmits to the router -- so without
# this there is nothing for the robot to range against. Costs one AP interface
# and a beacon every ~100ms; it does not affect the STA link or the drive API.
# Only started in STA mode, and only after the STA join succeeds.
RANGING_BEACON = True
RANGING_SSID = "mecanum-beacon"
