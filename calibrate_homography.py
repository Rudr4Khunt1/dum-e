"""
calibrate_homography.py — Stage 2: the pixel -> table map (replaces a depth camera).

WHY THIS WORKS: a pixel is a whole 3D ray, but if the object sits on a known flat
plane (the table), the ray hits exactly ONE point. A single 3x3 homography matrix H
maps image pixels <-> table (X, Y) in the ROBOT's frame. We calibrate it with the
arm itself as the ruler: the fingertip touches known IK-commanded points, you click
where the tip appears in the image. Lens distortion, camera pose and the robot's own
systematic offsets all get absorbed into H — because we map against where the arm
ACTUALLY lands.

PREREQS:  ik_test.py capture  ->  points (gate passed)  ->  touch (table_z saved).
CAMERA:   final position, focus locked, MUST NOT MOVE after this (recal = redo this).
NOTE:     the feed here is UNMIRRORED (unlike follow.py) — pixels map honestly.

MODES
  python calibrate_homography.py             calibrate: arm touches a 3x3 grid,
                                             you click its fingertip at each point
  python calibrate_homography.py validate    the party trick: click any pixel on
                                             the table -> the fingertip goes there

CALIBRATE KEYS   click = record fingertip   r = redo this point   s = skip   q = abort
VALIDATE KEYS    click = arm hovers there   t = touch table       u = back up   q = quit
"""
import os
import platform
import sys

import cv2
import numpy as np

import config as C
import kinematics as K
from arm_utils import connect, goto_xyz, safe_park

GRID_X = (0.14, 0.19, 0.24)      # meters, robot frame (forward from the pan axis)
GRID_Y = (-0.10, 0.0, 0.10)      # left/right of the arm's forward axis
HOVER = 0.05                     # travel height above the table
GRAZE = 0.003                    # fingertip height while you click it (on-plane)
H_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "homography.npz")

_click: dict = {"uv": None}


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click["uv"] = (x, y)


def open_cam():
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(f"Camera {C.CAMERA_INDEX} did not open.")
    return cap


def table_z():
    gm = K.load_geom()
    if "table_z" not in gm:
        raise SystemExit("No table_z in arm_geom.json — run `python ik_test.py touch` first.")
    return gm["table_z"]


def show(frame, lines, colour=(0, 255, 0)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.imshow("Dum-E homography", frame)
    return cv2.waitKey(1) & 0xFF


def calibrate(robot, cap):
    z0 = table_z()
    pts = [(x, y) for x in GRID_X for y in GRID_Y]
    pixels, table = [], []
    i = 0
    while i < len(pts):
        x, y = pts[i]
        print(f"[{i + 1}/{len(pts)}] fingertip -> ({x:.2f}, {y:+.2f})")
        try:
            goto_xyz(robot, x, y, z0 + HOVER, seconds=2.0)
            goto_xyz(robot, x, y, z0 + GRAZE, seconds=1.2)
        except K.NotReachable as e:
            print("  skipped (unreachable):", e)
            i += 1
            continue

        _click["uv"] = None
        action = None
        while action is None:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("camera feed died")
            if _click["uv"]:
                action = "record"
                break
            key = show(frame, [
                f"point {i + 1}/{len(pts)}   CLICK the fingertip TIP",
                "r = redo   s = skip   q = abort",
            ])
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord("s"):
                action = "skip"
            if key == ord("r"):
                action = "redo"

        if action == "record":
            u, v = _click["uv"]
            pixels.append((u, v))
            table.append((x, y))
            print(f"  recorded pixel ({u},{v}) <-> table ({x:.2f},{y:+.2f})")
        goto_xyz(robot, x, y, z0 + HOVER, seconds=1.2)
        if action != "redo":
            i += 1

    if len(pixels) < 4:
        raise SystemExit(f"Only {len(pixels)} points — need at least 4 for a homography.")

    px = np.array(pixels, dtype=np.float64)
    xy = np.array(table, dtype=np.float64)
    H, _mask = cv2.findHomography(px, xy, cv2.RANSAC)
    if H is None:
        raise SystemExit("findHomography failed — recollect the points.")

    # reprojection residuals: how far off each calibrated point maps, in mm
    ones = np.hstack([px, np.ones((len(px), 1))])
    proj = (H @ ones.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err_mm = np.linalg.norm(proj - xy, axis=1) * 1000
    np.savez(H_PATH, H=H, pixels=px, table=xy)
    print(f"\nsaved {H_PATH}")
    print(f"residuals: mean {err_mm.mean():.1f} mm  max {err_mm.max():.1f} mm  (per point: "
          + " ".join(f"{e:.0f}" for e in err_mm) + ")")
    print("Rule of thumb: mean under ~8 mm is plenty for a gripper-width grasp.")
    print("Next:  python calibrate_homography.py validate")


def pixel_to_xy(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]


def validate(robot, cap):
    if not os.path.exists(H_PATH):
        raise SystemExit("No homography.npz — run calibration first.")
    H = np.load(H_PATH)["H"]
    z0 = table_z()
    cur = None
    print("Click anywhere on the table — the fingertip goes there. t=touch u=up q=quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        key = show(frame, [
            "VALIDATE: click the table -> fingertip goes there",
            "t = touch table   u = back up   q = quit",
        ], colour=(0, 255, 255))
        if _click["uv"]:
            u, v = _click["uv"]
            _click["uv"] = None
            x, y = pixel_to_xy(H, u, v)
            print(f"pixel ({u},{v}) -> table ({x:.3f}, {y:+.3f})", end="  ")
            try:
                goto_xyz(robot, x, y, z0 + HOVER, seconds=1.8)
                cur = (x, y)
                print("[hovering]")
            except K.NotReachable as e:
                cur = None
                print("unreachable:", e)
        elif key == ord("t") and cur:
            goto_xyz(robot, cur[0], cur[1], z0 + GRAZE, seconds=1.0)
        elif key == ord("u") and cur:
            goto_xyz(robot, cur[0], cur[1], z0 + HOVER, seconds=1.0)
        elif key == ord("q"):
            break


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    cap = open_cam()
    cv2.namedWindow("Dum-E homography")
    cv2.setMouseCallback("Dum-E homography", _on_mouse)
    robot = connect()
    try:
        if mode == "validate":
            validate(robot, cap)
        else:
            calibrate(robot, cap)
    except KeyboardInterrupt:
        print("\naborted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        safe_park(robot)


if __name__ == "__main__":
    main()
