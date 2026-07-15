"""
kinematics.py — closed-form FK/IK for the SO-101, specialized for TOP-DOWN grasps.

WHY NOT A GENERIC SOLVER: for a gripper that always points STRAIGHT DOWN, the
6-DOF problem collapses to a planar 2-link triangle + two trivial angles. ~60
lines, no placo/URDF deps (placo wheels are shaky on Windows), fully debuggable.

FRAMES & CONVENTIONS
  Robot frame: origin = shoulder_pan axis at TABLE level. +z up.
               pan = atan2(y, x); r = planar distance from the pan axis.
  Radial plane (after pan): shoulder pivot S sits at (R_OFF, H0).
  Geometric joint angles (degrees):
    pan      : rotation toward the target
    shoulder : 0 = upper arm straight UP,        + = leaning toward target
    elbow    : 0 = forearm aligned w/ upper arm, + = folding
    wrist    : 0 = gripper aligned w/ forearm,   + = folding further
  Vertical-gripper constraint:  shoulder + elbow + wrist = 180
  (all-zero = arm pointing straight up; 180 of total fold = fingertip down)

ROBOT vs GEOMETRIC ANGLES
  LeRobot's calibrated degrees have arbitrary per-joint zero/direction. The map
      robot_deg = SIGN * geom_deg + OFFSET
  is captured empirically by `ik_test.py capture` (pose the arm straight up once,
  then answer three "which way did it move" questions) and stored in arm_geom.json.

  Absolute accuracy is NOT critical: X,Y residuals are absorbed by the homography
  (we calibrate against where the arm ACTUALLY lands), and Z by the table-touch step.

Link lengths from the official SO-101 URDF (so101_new_calib.urdf); tweak in config.
"""
import json
import math
import os

import config as C

_GEOM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_geom.json")

GEOM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


class NotReachable(ValueError):
    pass


def load_geom():
    """Per-joint {sign, offset} captured by `ik_test.py capture`."""
    if not os.path.exists(_GEOM_PATH):
        raise SystemExit(
            f"{_GEOM_PATH} not found.\nRun `python ik_test.py capture` first "
            "(one-time: pose the arm straight up + three jog questions)."
        )
    with open(_GEOM_PATH) as f:
        return json.load(f)


def save_geom(geom):
    with open(_GEOM_PATH, "w") as f:
        json.dump(geom, f, indent=2)
    print(f"saved {_GEOM_PATH}")


def geom_to_robot(geom_deg: dict, gm: dict) -> dict:
    """geometric angles -> LeRobot '<joint>.pos' command dict (no gripper)."""
    out = {}
    for j in GEOM_JOINTS:
        g = gm[j]
        out[j + ".pos"] = g["sign"] * geom_deg[j] + g["offset"]
    # wrist_roll: hold the captured reference so the jaws keep a fixed orientation
    out["wrist_roll.pos"] = gm["wrist_roll"]["offset"]
    return out


def ik_vertical(x: float, y: float, z: float) -> dict:
    """TCP at (x, y, z) meters in the robot frame, gripper pointing straight down.
    Returns geometric angles {shoulder_pan, shoulder_lift, elbow_flex, wrist_flex} deg.
    Elbow-up solution (the natural pose for reaching down at a table).
    """
    L1, L2, L3 = C.LINK_L1, C.LINK_L2, C.LINK_L3
    pan = math.degrees(math.atan2(y, x))

    r = math.hypot(x, y) - C.LINK_R_OFF        # radial dist from the shoulder column
    # wrist_flex axis sits L3 straight above the fingertip (vertical gripper)
    dx = r
    dz = (z + L3) - C.LINK_H0                  # relative to the shoulder pivot
    D = math.hypot(dx, dz)

    if D > (L1 + L2) * 0.999:
        raise NotReachable(f"target r={r:.3f} z={z:.3f}: out of reach (D={D:.3f} > {L1+L2:.3f})")
    if D < abs(L1 - L2) * 1.001 or r < 0.02:
        raise NotReachable(f"target r={r:.3f} z={z:.3f}: too close to the base column")

    # interior elbow angle via law of cosines; elbow geometric = 180 - interior
    cos_int = (L1 * L1 + L2 * L2 - D * D) / (2 * L1 * L2)
    interior = math.degrees(math.acos(max(-1.0, min(1.0, cos_int))))
    elbow = 180.0 - interior

    # shoulder: angle of S->W from vertical, minus the triangle's first angle (elbow-up)
    psi = math.degrees(math.atan2(dx, dz))
    cos_g = (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D)
    gamma = math.degrees(math.acos(max(-1.0, min(1.0, cos_g))))
    shoulder = psi - gamma

    wrist = 180.0 - shoulder - elbow           # vertical-gripper constraint

    return {"shoulder_pan": pan, "shoulder_lift": shoulder,
            "elbow_flex": elbow, "wrist_flex": wrist}


def fk_vertical(shoulder: float, elbow: float, pan: float) -> tuple:
    """Sanity-check FK for the same convention (assumes the wrist keeps the gripper
    vertical). Returns (x, y, z) of the TCP."""
    L1, L2, L3 = C.LINK_L1, C.LINK_L2, C.LINK_L3
    a1 = math.radians(shoulder)                 # from vertical
    a2 = math.radians(shoulder + elbow)         # forearm direction from vertical
    r = L1 * math.sin(a1) + L2 * math.sin(a2)
    zw = C.LINK_H0 + L1 * math.cos(a1) + L2 * math.cos(a2)
    z = zw - L3
    r_full = r + C.LINK_R_OFF
    p = math.radians(pan)
    return (r_full * math.cos(p), r_full * math.sin(p), z)
