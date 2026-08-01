"""
pick.py — Stage 3: Dum-E fetches, with human-in-the-loop tuning.

    YOLO-seg spots objects -> press the NUMBER of the one you want -> the pick runs
    as THREE PAUSED PHASES you can jog to exact, and your corrections are SAVED per
    object class (pick_tune.json) and applied automatically on the next pick of
    that class. First pick of a class is guided; a few picks later it converges to
    Enter-Enter-Enter.

PHASES (after pressing a number)
  1. HOVER   arm hovers over where it thinks the dot is, jaws OPEN.
             jog XY:  i/k = away from / toward the base   j/l = arm's left/right
             o/c = jaws more open / more closed            ENTER = accept
  2. DESCEND arm steps down; bring the open jaws around the object until it's at
             grasp depth (near the table).  d/u = down/up, XY jog still live.
             ENTER = close the jaws
  3. CLOSE   c/o tightens/loosens in small steps until the grip looks solid.
             ENTER = accept -> lift.   Then:  h = hand over   r = put back
  q at any phase = abort this pick (lifts back to hover).

WHAT GETS SAVED per class: {dx, dy, dz, open, close} — your XY correction, grasp
depth, and the REAL open/close values for your build (the config GRIPPER_* numbers
are only first-run defaults).

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
from arm_utils import connect, goto_xyz, ramp_to, safe_park
from calibrate_homography import H_PATH, open_cam, pixel_to_xy, table_z
from hand_tracker import HandTracker, palm_center

TUNE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.PICK_TUNE_FILE)
ELONGATED_MIN = 1.8       # long/short axis ratio above which an object is "long"
WINDOW = "Dum-E pick"


class PickAborted(Exception):
    pass


def load_H():
    if not os.path.exists(H_PATH):
        raise SystemExit("No homography.npz — run calibrate_homography.py first.")
    return np.load(H_PATH)["H"]


def load_tune():
    if os.path.exists(TUNE_PATH):
        with open(TUNE_PATH) as f:
            return json.load(f)
    return {}


def save_tune(tune):
    with open(TUNE_PATH, "w") as f:
        json.dump(tune, f, indent=2)


def set_gripper(robot, pos, seconds=0.5):
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
        # concave shapes (deep crescents) put the centroid OFF the body — verify.
        if on_body(aim):
            return aim, None
        aim = band_aim()
    else:
        aim = band_aim()                        # elongated (banana case)

    if not on_body(aim):                        # last resort: nearest mask point
        d = np.linalg.norm(pts - np.asarray(aim), axis=1)
        aim = (float(pts[d.argmin(), 0]), float(pts[d.argmin(), 1]))

    ax0 = pixel_to_xy(H, aim[0], aim[1])
    ax1 = pixel_to_xy(H, aim[0] + 50 * e1[0], aim[1] + 50 * e1[1])
    world_angle = math.degrees(math.atan2(ax1[1] - ax0[1], ax1[0] - ax0[0]))
    return aim, world_angle


def roll_for(world_angle, pan_deg, gm):
    """wrist_roll command that puts the jaws ACROSS the object's long axis (mod-180
    nearest equivalent, clamped). See config ROLL_JAW_REF_DEG / ROLL_ALIGN_SIGN."""
    jaw_des = world_angle + 90.0
    raw = jaw_des - pan_deg - C.ROLL_JAW_REF_DEG
    raw = (raw + 90.0) % 180.0 - 90.0
    off = gm["wrist_roll"]["offset"]
    cmd = off + C.ROLL_ALIGN_SIGN * raw
    return max(off - C.ROLL_MAX_DEG, min(off + C.ROLL_MAX_DEG, cmd))


def jaw_gap_target(x, y, tilt):
    """Shift the commanded point outward so the JAW GAP — not the fingertip tip —
    lands on the aim point (at tilt t the gap sits h*tan(t) radially short)."""
    shift = C.JAW_CONTACT_H * math.tan(math.radians(tilt)) + C.GRASP_RADIAL_NUDGE
    r = math.hypot(x, y)
    if r < 1e-6 or shift == 0.0:
        return x, y
    return x + shift * x / r, y + shift * y / r


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


# ────────────────────────── interactive phases ──────────────────────────

def _phase(cap, robot, state, title, hints, keymap, do_move):
    """Shared phase loop: live camera + jog keys until ENTER. keymap maps a key to
    a state-mutating fn; do_move re-sends the arm after any jog. q aborts."""
    dirty = False
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera feed died")
        lines = [title] + hints + [
            f"x {state['x']:+.3f}  y {state['y']:+.3f}  z +{state['z'] - state['z0']:.3f}"
            f"  grip {state['grip']:.0f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 30 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):                     # ENTER
            return
        if key == ord("q"):
            raise PickAborted
        if key in keymap:
            keymap[key]()
            dirty = True
        elif dirty:
            do_move()
            dirty = False


def auto_pick(robot, label, x0, y0, z0, tilt, roll, t):
    """Hands-off pick using a class's saved corrections (no phase pauses)."""
    x, y = x0 + t["dx"], y0 + t["dy"]
    grasp_z = z0 + C.GRASP_HEIGHT + t["dz"]
    print(f"  auto pick ({label}): ({x:.3f},{y:+.3f})")
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=2.0, tilt=tilt, roll=roll)
    set_gripper(robot, t["open"])
    goto_xyz(robot, x, y, grasp_z, seconds=1.2, tilt=tilt, roll=roll)
    set_gripper(robot, t["close"], seconds=0.7)
    time.sleep(0.2)
    goto_xyz(robot, x, y, z0 + C.CARRY_HEIGHT, seconds=1.2, tilt=tilt, roll=roll)
    print("  lifted. h = hand over | r = put back")
    return x, y, tilt, roll


def interactive_pick(robot, cap, label, x0, y0, z0, tilt, roll, tune):
    """The three-phase guided pick. Returns the final (x, y, tilt, roll) and saves
    this class's corrections."""
    t = tune.get(label, {})
    state = {
        "x": x0 + t.get("dx", 0.0),
        "y": y0 + t.get("dy", 0.0),
        "z": z0 + C.PICK_HOVER,
        "z0": z0,
        "grip": t.get("open", C.GRIPPER_OPEN),
    }
    grasp_z = z0 + C.GRASP_HEIGHT + t.get("dz", 0.0)

    def move():
        goto_xyz(robot, state["x"], state["y"], state["z"], seconds=0.35,
                 tilt=tilt, roll=roll)

    def grip():
        set_gripper(robot, state["grip"], seconds=0.25)

    jog = {
        ord("i"): lambda: state.update(x=state["x"] + C.JOG_XY),
        ord("k"): lambda: state.update(x=state["x"] - C.JOG_XY),
        ord("j"): lambda: state.update(y=state["y"] + C.JOG_XY),
        ord("l"): lambda: state.update(y=state["y"] - C.JOG_XY),
    }

    # PHASE 1 — hover, jaws open, align XY over the object
    goto_xyz(robot, state["x"], state["y"], state["z"], seconds=2.0, tilt=tilt, roll=roll)
    grip()
    _phase(cap, robot, state, f"HOVER over the {label} — align, then ENTER",
           ["i/k = away/toward base   j/l = left/right   o/c = jaws open/close"],
           {**jog,
            ord("o"): lambda: state.update(grip=state["grip"] + C.JOG_GRIP),
            ord("c"): lambda: state.update(grip=state["grip"] - C.JOG_GRIP)},
           lambda: (move(), grip()))

    # PHASE 2 — descend with open jaws to grasp depth
    state["z"] = grasp_z
    move()
    _phase(cap, robot, state, "DESCEND — jaws around the object, then ENTER to close",
           ["d/u = down/up   i/k/j/l = XY   (get the jaws straddling it)"],
           {**jog,
            ord("d"): lambda: state.update(z=state["z"] - C.JOG_Z),
            ord("u"): lambda: state.update(z=state["z"] + C.JOG_Z)},
           move)
    open_val = state["grip"]

    # PHASE 3 — close until solid
    state["grip"] = tune.get(label, {}).get("close", C.GRIPPER_CLOSED)
    grip()
    _phase(cap, robot, state, "CLOSE — c = tighter, o = looser, ENTER = lift",
           [],
           {ord("c"): lambda: state.update(grip=state["grip"] - C.JOG_GRIP),
            ord("o"): lambda: state.update(grip=state["grip"] + C.JOG_GRIP)},
           grip)

    # save this class's corrections
    tune[label] = {
        "dx": round(state["x"] - x0, 4),
        "dy": round(state["y"] - y0, 4),
        "dz": round(state["z"] - (z0 + C.GRASP_HEIGHT), 4),
        "open": round(open_val, 1),
        "close": round(state["grip"], 1),
    }
    save_tune(tune)
    print(f"  saved corrections for '{label}': {tune[label]}")

    goto_xyz(robot, state["x"], state["y"], z0 + C.CARRY_HEIGHT, seconds=1.2,
             tilt=tilt, roll=roll)
    print("  lifted. h = hand over | r = put back")
    return state["x"], state["y"], tilt, roll


def hand_over(robot):
    """Fixed presenting pose (fallback when no palm is found)."""
    hx, hy, hz = C.HANDOVER_XYZ
    goto_xyz(robot, hx, hy, hz, seconds=2.0, tilt=C.HANDOVER_TILT)
    time.sleep(C.HANDOVER_PAUSE_S)
    set_gripper(robot, C.GRIPPER_OPEN)
    time.sleep(0.4)
    goto_xyz(robot, hx, hy, hz + 0.04, seconds=1.0, tilt=C.HANDOVER_TILT)
    print("  delivered.")


def find_palm(cap, tracker, H):
    """Scan the feed for a hand; return its palm center as table (x, y), or None.
    Valid only when the palm rests ON the table (H maps the table plane)."""
    t0 = time.time()
    while time.time() - t0 < C.PALM_SEARCH_S:
        ok, frame = cap.read()
        if not ok:
            return None
        hands = tracker.detect(frame)
        if hands:
            hx, hy = palm_center(hands[0])
            fh, fw = frame.shape[:2]
            return pixel_to_xy(H, hx * fw, hy * fh)
    return None


def hand_over_to_palm(robot, cap, tracker, H, z0):
    """Place the object ON the user's palm (palm flat on the table). Falls back to
    the fixed presenting pose if no hand is seen or the palm is out of reach."""
    print("  looking for your palm (flat on the table)...")
    spot = find_palm(cap, tracker, H)
    if spot is None:
        print("  no hand seen -- fixed handover instead.")
        hand_over(robot)
        return
    x, y = spot
    print(f"  palm at ({x:.3f},{y:+.3f}) -- placing")
    try:
        goto_xyz(robot, x, y, z0 + C.CARRY_HEIGHT, seconds=1.5, tilt=0)
        goto_xyz(robot, x, y, z0 + C.PALM_DROP_HEIGHT, seconds=1.0, tilt=0)
        set_gripper(robot, C.GRIPPER_OPEN)
        time.sleep(0.3)
        goto_xyz(robot, x, y, z0 + C.CARRY_HEIGHT, seconds=1.0, tilt=0)
        print("  placed on your palm.")
    except K.NotReachable as e:
        print("  palm outside the vertical zone:", e, "-- fixed handover instead.")
        hand_over(robot)


def put_back(robot, x, y, z0, tilt, roll):
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.6, tilt=tilt, roll=roll)
    goto_xyz(robot, x, y, z0 + C.GRASP_HEIGHT + 0.004, seconds=1.2, tilt=tilt, roll=roll)
    set_gripper(robot, C.GRIPPER_OPEN)
    time.sleep(0.3)
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.2, tilt=tilt)
    print("  returned.")


# ────────────────────────── main ──────────────────────────

def main():
    argparse.ArgumentParser(description="Dum-E pick (interactive)").parse_args()
    from ultralytics import YOLO
    model = YOLO(C.YOLO_MODEL)
    H = load_H()
    z0 = table_z()
    gm = K.load_geom()
    tune = load_tune()
    cap = open_cam()
    tracker = HandTracker(num_hands=1)
    robot = connect()
    carrying = None          # (x, y, z0, tilt, roll) while holding something
    force_guided = False     # 'g' arms guided mode for the next pick (re-tune)
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
                tuned = "*" if label in tune else ""
                cv2.putText(frame, f"[{i + 1}] {label}{tuned} {conf:.2f}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if carrying:
                status = "CARRYING -- h = hand over | r = put back"
            else:
                status = "press [n] to pick | q quit  (* = tuned -> auto)"
                if force_guided:
                    status += "  [GUIDED armed]"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255) if carrying else (0, 255, 0), 2)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g") and not carrying:
                force_guided = not force_guided
                print(f"guided mode for next pick: {force_guided}")
                continue
            if carrying:
                if key == ord("h"):
                    hand_over_to_palm(robot, cap, tracker, H, z0)
                    carrying = None
                elif key == ord("r"):
                    put_back(robot, *carrying)
                    carrying = None
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(dets):
                    label, _c, (u, v), _b, _p, ang = dets[idx]
                    x, y = pixel_to_xy(H, u, v)
                    print(f"[{label}] pixel ({u:.0f},{v:.0f}) -> table ({x:.3f},{y:+.3f})")
                    tilt = 0.0
                    try:
                        _, tilt = K.ik_reach(x, y, z0 + C.GRASP_HEIGHT)
                        x, y = jaw_gap_target(x, y, tilt)
                        roll = None
                        if C.ROLL_ALIGN and ang is not None:
                            pan = math.degrees(math.atan2(y, x))
                            roll = roll_for(ang, pan, gm)
                        use_auto = (C.AUTO_WHEN_TUNED and label in tune
                                    and not force_guided)
                        if use_auto:
                            fx, fy, tilt, roll = auto_pick(
                                robot, label, x, y, z0, tilt, roll, tune[label])
                        else:
                            fx, fy, tilt, roll = interactive_pick(
                                robot, cap, label, x, y, z0, tilt, roll, tune)
                        force_guided = False
                        carrying = (fx, fy, z0, tilt, roll)
                    except K.NotReachable as e:
                        print("  out of reach (even tilted):", e)
                    except PickAborted:
                        print("  aborted — lifting clear.")
                        try:
                            goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.2, tilt=tilt)
                        except K.NotReachable:
                            pass
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        safe_park(robot)


if __name__ == "__main__":
    main()
