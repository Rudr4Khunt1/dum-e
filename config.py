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
SIGN_PAN,  SIGN_TILT  = -1, +1       # flip a sign if it steers AWAY from your hand
                                     # (pan is -1: the mirrored frame inverts left/right)
SMOOTHING = 0.28                     # 0..1 glide toward target (HIGHER = snappier, less lag)
DEADZONE_PX = 25                     # hold position for errors smaller than this (anti-twitch)

# ── Jitter control ─────────────────────────────────────────────────────────
# MediaPipe landmarks wobble several px even when your hand is perfectly still, and
# at K deg/px that noise turns into visible servo shake. Two defenses:
#   1) track the PALM CENTER (mean of 5 landmarks) instead of the lone wrist point
#   2) a One-Euro filter on the pixel position: smooths HARD when the hand is still
#      (jitter dies) and loosens when it moves fast (stays responsive). A plain EMA
#      can't do both -- one constant forces you to trade jitter against lag.
USE_PALM_CENTER = True
FILTER_MIN_CUTOFF = 1.0              # LOWER = smoother when still (more lag). Try 0.5 if still shaky.
FILTER_BETA = 0.03                   # HIGHER = less lag when moving fast. This is the knob that
                                     # fixes "it lags behind my hand". Costs nothing at rest.

# How far the "head" may sweep from its startup pose, in degrees. This is the
# hard safety envelope: the FINAL summed pose (follow + future idle/gesture) is
# clamped to it every tick, AFTER mixing, right before it reaches the servos.
PAN_LIMIT_DEG  = 70.0
TILT_LIMIT_DEG = 45.0

# ── Gripper roll: mirror your hand's rotation (turn it like a safe dial) ────
# Read from the wrist -> middle-finger-base vector, i.e. the hand's IN-PLANE angle.
# (True forearm twist/pronation is out-of-plane and 2D landmarks read it poorly, so
# rotate your hand like a steering wheel facing the camera.)
ENABLE_ROLL = True
ROLL_JOINT  = "wrist_roll"           # motor 5
K_ROLL      = 1.0                    # 1.0 = 1:1 (turn hand 30 deg -> gripper 30 deg)
SIGN_ROLL   = -1                     # -1: the gripper FACES you, so it must mirror your
                                     # roll (and the frame is mirrored too). Flip if wrong.
ROLL_LIMIT_DEG = 90.0

# ── Personality (soul.md, rungs 1-2: it droops when it loses you) ──────────
# The FOLLOW loop already knows when the hand appears and disappears -- that's all an
# emotional beat needs. Mood also scales the SMOOTHING, which is where the feeling
# actually lives: sad = slow + heavy, excited = fast + eager. Same poses, different
# timing, completely different character.
ENABLE_PERSONALITY = True
LOST_AFTER_S    = 1.5                # no hand for this long -> droop ("where'd you go...")
DROOP_DEG       = 25.0               # how far the head sags (FLIP SIGN if it droops UP)
DROOP_SMOOTHING = 0.04               # slow + heavy = sad
PERK_DEG        = 12.0               # overshoot upward when you come back ("there you are!")
PERK_SECONDS    = 0.6
PERK_SMOOTHING  = 0.50               # fast + eager = excited. Kept well ABOVE the normal
                                     # SMOOTHING so the perk still reads as a burst of
                                     # excitement -- the CONTRAST is the character, not the
                                     # absolute speed. (droop 0.04 << track 0.28 << perk 0.50)

# ── Slew-rate caps, PER JOINT (deg/sec) ────────────────────────────────────
# Only the BASE rings: it swings the arm's entire mass, and the printed structure is
# compliant, so a fast slew sets it oscillating like a pendulum. The wrist joints move
# almost no mass and can snap as fast as the servo allows -- throttling THEM to the
# base's limit is what made the whole robot feel lazy.
#   shoulder_pan -> LOWER this if the base starts ringing again (90 is very safe)
#   wrist_roll   -> nearly massless, let it fly
MAX_DEG_PER_SEC = {
    "shoulder_pan": 130.0,   # the heavy one (the only one that rings)
    "shoulder_lift": 120.0,
    "elbow_flex":   140.0,
    "wrist_flex":   240.0,
    "wrist_roll":   340.0,   # snappy
    "gripper":      300.0,
}
DEFAULT_MAX_DEG_PER_SEC = 160.0   # for any joint not listed above

FOLLOW_HZ = 30                 # loop rate (more updates/sec = smoother AND more responsive)

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
