"""
config.py — Dum-E's single knob file.

Edit THIS on the Windows desktop; every script reads its settings from here so
you never have to touch the script bodies.
"""

# ── Arm ────────────────────────────────────────────────────────────────────
# Follower controller-board serial port.
#   Windows: run `lerobot-find-port`   ->  e.g. "COM5"
#   Mac:     e.g. "/dev/tty.usbmodem5B141157431"
PORT = "COM5"                  # <-- EDIT THIS to your port

ROBOT_ID = "dum_e_follower"    # must match the --robot.id you calibrated with
USE_DEGREES = True             # report/command joint angles in degrees (readable)

# Safety: max degrees any joint may move in ONE command step. Small = safe
# (the arm physically cannot lunge). Raise later once you trust it.
MAX_STEP_DEG = 8.0

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0               # if the wrong camera opens, try 1, 2, ...
FRAME_W, FRAME_H = 1280, 720

# ── FOLLOW loop ────────────────────────────────────────────────────────────
PAN_JOINT  = "shoulder_pan"    # left/right "look"  (joint 1)
TILT_JOINT = "wrist_flex"      # up/down "look"     (joint 4)

GAIN_PAN,  GAIN_TILT  = 0.03, 0.03   # chase strength (raise if sluggish, lower if jittery)
SIGN_PAN,  SIGN_TILT  = +1, +1       # flip a sign if it steers AWAY from your hand
SMOOTHING = 0.15                     # 0..1 glide toward target (lower = smoother/slower)
DEADZONE_PX = 30                     # ignore errors smaller than this (anti-twitch)

# How far the "head" may sweep from its startup pose, in degrees (anti-windup).
PAN_LIMIT_DEG  = 70.0
TILT_LIMIT_DEG = 45.0

FOLLOW_HZ = 25                 # loop rate
