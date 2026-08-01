"""
pick.py — Stage 3: Dum-E fetches. Detect -> pick -> deliver (keypress v1; voice later).

    YOLO-seg spots objects -> you press the NUMBER on the one you want -> mask math
    picks the true grasp point + jaw angle -> homography H -> table (x, y) -> IK
    (vertical, or auto-tilted for far targets): hover, open, descend, close, lift.
        h  = hand it over     r  = put it back     q  = quit (parks)

GRASP-POINT MATH (why not just "the centroid"):
  * elongated flat objects (banana!): a crescent's centroid lies OFF the body, in
    the hollow of the curve — aiming there misses. We take the mask points near the
    MIDDLE of the long axis and use their mean: on-body, mid-arc. The local axis
    there also gives the jaw angle, so wrist_roll turns the jaws ACROSS the object.
  * roundish flat objects: mask centroid (it's on the body for convex shapes).
  * tall objects: bottom band of the mask = the table footprint (a tall object's
    centroid projects past its base from an angled camera).

TEACH A GRIP (per object class):     python pick.py --teach banana
  Torque releases; you pose the jaws around the object at the spot the robot will
  actually grab (gripper roughly down, jaws across the width), closed to JUST
  TOUCHING, then Enter. We grip GRIP_SQUEEZE tighter than what you set. Saved to
  grips.json; classes without a taught grip fall back to GRIPPER_CLOSED.

PREREQS: ik_test capture/points/touch + calibrate_homography (validate OK).
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import config as C
import kinematics as K
from arm_utils import connect, goto_xyz, pose_now, ramp_to, safe_park
from calibrate_homography import H_PATH, open_cam, pixel_to_xy, table_z

GRIPS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.GRIPS_FILE)
ELONGATED_MIN = 1.8       # long/short axis ratio above which an object is "long"


def load_H():
    if not os.path.exists(H_PATH):
        raise SystemExit("No homography.npz — run calibrate_homography.py first.")
    return np.load(H_PATH)["H"]


def load_grips():
    if os.path.exists(GRIPS_PATH):
        with open(GRIPS_PATH) as f:
            return json.load(f)
    return {}


def set_gripper(robot, pos, seconds=0.6):
    ramp_to(robot, {"gripper.pos": pos}, seconds)


# ────────────────────────── grasp geometry ──────────────────────────

def grasp_geometry(label, poly, bbox, H):
    """-> ((u, v) aim pixel, world_angle_deg | None).
    world_angle is the object's long-axis direction in the ROBOT frame (via H),
    present only for elongated flat objects — that's when jaw alignment matters."""
    x1, y1, x2, y2 = bbox
    if poly is None or len(poly) < 3:
        uv = ((x1 + x2) / 2, (y1 + y2) / 2) if label in C.FLAT_CLASSES \
            else ((x1 + x2) / 2, y2)
        return uv, None
    pts = np.asarray(poly, dtype=np.float32)

    if label not in C.FLAT_CLASSES:
        band = pts[pts[:, 1] >= y2 - 0.25 * (y2 - y1)]
        if len(band) == 0:
            band = pts
        return (float(band[:, 0].mean()), float(band[:, 1].mean())), None

    # flat object: PCA for the long axis + elongation
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / len(pts)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    ratio = math.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9))
    e1 = evecs[:, 1]                            # long-axis direction (image space)
    cnt = pts.reshape(-1, 1, 2)

    def on_body(p):
        return cv2.pointPolygonTest(cnt, (float(p[0]), float(p[1])), False) >= 0

    def band_aim():
        """Mid-band along the long axis -> on-body for arcs/crescents."""
        proj = centered @ e1
        band = pts[np.abs(proj - np.median(proj)) < 0.15 * (proj.max() - proj.min() + 1e-6)]
        if len(band) == 0:
            band = pts
        return (float(band[:, 0].mean()), float(band[:, 1].mean()))

    if ratio < ELONGATED_MIN:
        m = cv2.moments(pts)
        aim = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] > 1e-6 \
            else (float(mean[0]), float(mean[1]))
        # concave shapes (deep crescents, horseshoes) put the centroid OFF the
        # body even when not "elongated" — verify, and fall back to the band.
        if on_body(aim):
            return aim, None
        aim = band_aim()
    else:
        aim = band_aim()                        # elongated (banana case)

    if not on_body(aim):                        # last resort: nearest mask point
        d = np.linalg.norm(pts - np.asarray(aim), axis=1)
        aim = (float(pts[d.argmin(), 0]), float(pts[d.argmin(), 1]))

    # local axis in ROBOT frame: map two nearby pixels along e1 through H
    ax0 = pixel_to_xy(H, aim[0], aim[1])
    ax1 = pixel_to_xy(H, aim[0] + 50 * e1[0], aim[1] + 50 * e1[1])
    world_angle = math.degrees(math.atan2(ax1[1] - ax0[1], ax1[0] - ax0[0]))
    return aim, world_angle


def roll_for(world_angle, pan_deg, gm):
    """wrist_roll command that puts the jaws ACROSS the object's long axis.
    Jaw orientation is mod-180 (the jaws are symmetric), so we take the nearest
    equivalent and clamp travel. See config ROLL_JAW_REF_DEG / ROLL_ALIGN_SIGN
    for the one-time physical sign/reference tune."""
    jaw_des = world_angle + 90.0                       # across the axis
    raw = jaw_des - pan_deg - C.ROLL_JAW_REF_DEG
    raw = (raw + 90.0) % 180.0 - 90.0                  # nearest mod-180 equivalent
    off = gm["wrist_roll"]["offset"]
    cmd = off + C.ROLL_ALIGN_SIGN * raw
    return max(off - C.ROLL_MAX_DEG, min(off + C.ROLL_MAX_DEG, cmd))


# ────────────────────────── detection ──────────────────────────

def detect(model, frame, H):
    """-> [(label, conf, (u, v), bbox, poly, world_angle)] for allowed classes."""
    res = model.predict(frame, conf=C.PICK_CONF, verbose=False)[0]
    polys = res.masks.xy if res.masks is not None else None
    out = []
    for i, b in enumerate(res.boxes):
        label = model.names[int(b.cls[0])]
        if label not in C.PICK_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        poly = polys[i] if polys is not None and i < len(polys) else None
        uv, ang = grasp_geometry(label, poly, (x1, y1, x2, y2), H)
        out.append((label, float(b.conf[0]), uv, (x1, y1, x2, y2), poly, ang))
    return out


# ────────────────────────── actions ──────────────────────────

def jaw_gap_target(x, y, tilt):
    """Shift the commanded point outward so the JAW GAP — not the fingertip tip —
    lands on the aim point. At tilt t the gap sits h*tan(t) radially short of the
    tip at object height h; plus the manual vertical-pick trim from config."""
    shift = C.JAW_CONTACT_H * math.tan(math.radians(tilt)) + C.GRASP_RADIAL_NUDGE
    r = math.hypot(x, y)
    if r < 1e-6 or shift == 0.0:
        return x, y
    return x + shift * x / r, y + shift * y / r


def execute_pick(robot, x, y, z0, grip_close, tilt, roll):
    print(f"  pick at ({x:.3f}, {y:+.3f})  tilt={tilt:.0f}"
          + (f"  roll={roll:.0f}" if roll is not None else ""))
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=2.0, tilt=tilt, roll=roll)
    set_gripper(robot, C.GRIPPER_OPEN)
    goto_xyz(robot, x, y, z0 + C.GRASP_HEIGHT, seconds=1.2, tilt=tilt, roll=roll)
    set_gripper(robot, grip_close, seconds=0.8)
    time.sleep(0.2)
    goto_xyz(robot, x, y, z0 + C.CARRY_HEIGHT, seconds=1.2, tilt=tilt, roll=roll)
    print("  lifted. h = hand over | r = put back")


def hand_over(robot):
    hx, hy, hz = C.HANDOVER_XYZ
    goto_xyz(robot, hx, hy, hz, seconds=2.0)
    time.sleep(C.HANDOVER_PAUSE_S)
    set_gripper(robot, C.GRIPPER_OPEN)
    time.sleep(0.4)
    goto_xyz(robot, hx, hy, hz + 0.04, seconds=1.0)
    print("  delivered.")


def put_back(robot, x, y, z0, tilt, roll):
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.6, tilt=tilt, roll=roll)
    goto_xyz(robot, x, y, z0 + C.GRASP_HEIGHT + 0.004, seconds=1.2, tilt=tilt, roll=roll)
    set_gripper(robot, C.GRIPPER_OPEN)
    time.sleep(0.3)
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.2, tilt=tilt)
    print("  returned.")


def teach_grip(robot, label):
    print(
        f"\n== TEACH GRIP: {label} ==\n"
        "Torque is OFF. Pose the jaws around the object at the spot the robot will\n"
        "actually grab it (gripper roughly pointing down, jaws across the WIDTH,\n"
        "near table height), closed to JUST TOUCHING — no squeeze. Then Enter.\n"
    )
    robot.bus.disable_torque()
    input("pose it, hold, then Enter... ")
    pose = pose_now(robot)
    robot.bus.enable_torque()
    robot.send_action(pose)                    # hold so nothing collapses
    g0 = pose["gripper.pos"]
    close_dir = 1.0 if C.GRIPPER_CLOSED > C.GRIPPER_OPEN else -1.0
    target = g0 + close_dir * C.GRIP_SQUEEZE
    grips = load_grips()
    grips[label] = target
    with open(GRIPS_PATH, "w") as f:
        json.dump(grips, f, indent=2)
    print(f"touch = {g0:.1f}  ->  grip = {target:.1f}  (squeeze {C.GRIP_SQUEEZE})")
    print(f"saved {GRIPS_PATH}: {grips}")


# ────────────────────────── main ──────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dum-E pick")
    ap.add_argument("--teach", metavar="LABEL",
                    help="teach the grip width for an object class (e.g. banana)")
    args = ap.parse_args()

    if args.teach:
        robot = connect()
        try:
            teach_grip(robot, args.teach)
        finally:
            safe_park(robot)
        return

    from ultralytics import YOLO
    model = YOLO(C.YOLO_MODEL)
    H = load_H()
    z0 = table_z()
    gm = K.load_geom()
    grips = load_grips()
    cap = open_cam()
    robot = connect()
    carrying = None          # (x, y, tilt, roll) while holding something
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect(model, frame, H)
            for i, (label, conf, (u, v), (x1, y1, x2, y2), poly, ang) in enumerate(dets[:9]):
                if poly is not None and len(poly) >= 3:
                    cv2.polylines(frame, [np.asarray(poly, dtype=np.int32)],
                                  True, (0, 200, 0), 2)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 0), 2)
                cv2.circle(frame, (int(u), int(v)), 6, (0, 255, 255), -1)  # aim point
                taught = "*" if label in grips else ""
                cv2.putText(frame, f"[{i + 1}] {label}{taught} {conf:.2f}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status = ("CARRYING -- h = hand over | r = put back"
                      if carrying else "press [n] to pick | q quit  (* = taught grip)")
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255) if carrying else (0, 255, 0), 2)
            cv2.imshow("Dum-E pick", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if carrying:
                if key == ord("h"):
                    hand_over(robot)
                    carrying = None
                elif key == ord("r"):
                    put_back(robot, *carrying, )
                    carrying = None
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(dets):
                    label, _c, (u, v), _b, _p, ang = dets[idx]
                    x, y = pixel_to_xy(H, u, v)
                    print(f"[{label}] pixel ({u:.0f},{v:.0f}) -> table ({x:.3f},{y:+.3f})")
                    try:
                        _, tilt = K.ik_reach(x, y, z0 + C.GRASP_HEIGHT)
                        x, y = jaw_gap_target(x, y, tilt)   # mouth on the dot, not the tip
                        roll = None
                        if C.ROLL_ALIGN and ang is not None:
                            pan = math.degrees(math.atan2(y, x))
                            roll = roll_for(ang, pan, gm)
                        grip = grips.get(label, C.GRIPPER_CLOSED) \
                            if C.USE_TAUGHT_GRIPS else C.GRIPPER_CLOSED
                        execute_pick(robot, x, y, z0, grip, tilt, roll)
                        carrying = (x, y, z0, tilt, roll)
                    except K.NotReachable as e:
                        print("  out of reach (even tilted):", e)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        safe_park(robot)


if __name__ == "__main__":
    main()
