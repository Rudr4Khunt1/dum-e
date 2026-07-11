"""
config.py — Dum-E's single knob file.

Edit THIS on the Windows desktop; every script reads its settings from here so
you never have to touch the script bodies.
"""

# ── Arm ────────────────────────────────────────────────────────────────────
# Follower controller-board serial port.
#   Windows: run `lerobot-find-port`   ->  e.g. "COM5"
#   Mac:     e.g. "/dev/tty.usbmodem5B141157431"
PORT = "COM7"                  # <-- EDIT THIS to your port (yours showed COM7)

ROBOT_ID = "dum_e_follower"    # must match the --robot.id you calibrated with
USE_DEGREES = True             # report/command joint angles in degrees (readable)

# Safety: max degrees any joint may move in ONE command step. Small = safe
# (the arm physically cannot lunge). Raise later once you trust it.
MAX_STEP_DEG = 8.0

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0               # if the wrong camera opens, try 1, 2, ...
FRAME_W, FRAME_H = 1280, 720

# Mirror the frame horizontally (natural "selfie" feel in the preview).
# Harmless for FOLLOW -- it only decides which way SIGN_PAN must point.
# Set False for the homography/PICK phase so pixel coords map honestly to the world.
MIRROR = True

# Detect up to N hands. With 2, follow.py locks onto the hand nearest the one it
# was already tracking -- so it won't jump to your other hand resting on the keyboard.
NUM_HANDS = 2

# ── FOLLOW loop ────────────────────────────────────────────────────────────
PAN_JOINT  = "shoulder_pan"    # left/right "look"  (joint 1)
TILT_JOINT = "wrist_flex"      # up/down "look"     (joint 4)

# Because the camera is FIXED and side-offset, moving the arm does NOT change the
# image, so "follow" is a look-at MAPPING (pixel offset -> absolute joint angle),
# NOT error-integration. K_* is degrees of joint travel per pixel of hand offset
# from image center.  pan_target = start_pan + SIGN_PAN * K_PAN * (u - w/2).
#
# CALIBRATE (~2 min, once): aim the arm at your hand held at the LEFT frame edge,
# note the pan angle; repeat at the RIGHT edge; then
#     K_PAN = (pan_right - pan_left) / (u_right - u_left)
# Same idea for tilt using the top/bottom edges. Defaults below are conservative.
# Derived from the camera's real field of view -- K ~= FOV_deg / frame_pixels.
# C920: its advertised "78 deg" is the DIAGONAL. The real ones are
#   horizontal ~70.4 deg / 1280 px -> 0.055     vertical ~43.3 deg / 720 px -> 0.060
K_PAN,  K_TILT  = 0.055, 0.060       # deg of joint travel per pixel of hand offset
SIGN_PAN,  SIGN_TILT  = +1, +1       # flip a sign if it steers AWAY from your hand
SMOOTHING = 0.15                     # 0..1 glide toward target (lower = smoother/slower)
DEADZONE_PX = 30                     # hold position for errors smaller than this (anti-twitch)

# How far the "head" may sweep from its startup pose, in degrees. This is the
# hard safety envelope: the FINAL summed pose (follow + future idle/gesture) is
# clamped to it every tick, AFTER mixing, right before it reaches the servos.
PAN_LIMIT_DEG  = 70.0
TILT_LIMIT_DEG = 45.0

FOLLOW_HZ = 25                 # loop rate

# On startup, jog the head until it points at the image-center crosshair, then lock
# that pose as the reference. THIS MATTERS: the mapping is pan = start_pan + K*err,
# so the startup pose IS the definition of "image center". If the arm starts aimed
# at your face while the crosshair is on your chest, every angle is offset by that
# gap and it will orbit your head instead of landing on your hand.
AIM_ON_START = True
AIM_STEP_DEG = 2.0             # degrees per key press while aiming

# safe_park: on every exit (q / Ctrl-C / crash) the arm glides back to its startup
# rest pose over this many seconds BEFORE torque is released, so it never sags from
# an extended pose onto the desk. Set 0 to disable the glide (releases in place).
PARK_SECONDS = 1.5
