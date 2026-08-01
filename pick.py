"""
pick.py — Stage 3: Dum-E fetches. Detect -> pick -> deliver (keypress v1; voice later).

    YOLO spots objects on the table -> you press the NUMBER shown on the one you
    want -> bbox bottom-center pixel -> homography H -> table (x, y) -> IK grasp
    recipe: hover, open, descend, close, lift -> then:
        h  = hand it over  (moves to HANDOVER_XYZ, pauses, opens)
        r  = put it back   (returns to where it picked it, sets it down)

PREREQS: ik_test capture/points/touch done + calibrate_homography done (validate OK).
CAMERA:  unmirrored feed (same as calibration — pixels must map honestly).

KEYS
  1..9   pick that detection
  h / r  while carrying: hand over / put back
  q      quit (parks into the rest pose)

FIRST OBJECTS: start with something with a graspable body — a marker, small box,
tape roll, standing bottle. A phone LYING FLAT is geometrically hard for parallel
jaws (you'd grip its full width); save it for after tuning GRIPPER_* in config.
"""
import time

import cv2

import config as C
import kinematics as K
from arm_utils import connect, goto_xyz, ramp_to, safe_park
from calibrate_homography import H_PATH, open_cam, pixel_to_xy, table_z


def load_H():
    import os

    import numpy as np
    if not os.path.exists(H_PATH):
        raise SystemExit("No homography.npz — run calibrate_homography.py first.")
    return np.load(H_PATH)["H"]


def set_gripper(robot, pos, seconds=0.6):
    ramp_to(robot, {"gripper.pos": pos}, seconds)


def detect(model, frame):
    """Run YOLO, return [(label, conf, (u, v), bbox)] for allowed classes only,
    where (u, v) is the bbox BOTTOM-CENTER (the object's table footprint)."""
    res = model.predict(frame, conf=C.PICK_CONF, verbose=False)[0]
    out = []
    for b in res.boxes:
        label = model.names[int(b.cls[0])]
        if label not in C.PICK_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append((label, float(b.conf[0]), ((x1 + x2) / 2, y2), (x1, y1, x2, y2)))
    return out


def execute_pick(robot, x, y, z0):
    """The scripted grasp recipe. Positions computed live; only the SEQUENCE is fixed."""
    print(f"  pick at ({x:.3f}, {y:+.3f})")
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=2.0)
    set_gripper(robot, C.GRIPPER_OPEN)
    goto_xyz(robot, x, y, z0 + C.GRASP_HEIGHT, seconds=1.2)
    set_gripper(robot, C.GRIPPER_CLOSED, seconds=0.8)
    time.sleep(0.2)
    goto_xyz(robot, x, y, z0 + C.CARRY_HEIGHT, seconds=1.2)
    print("  lifted. h = hand over | r = put back")


def hand_over(robot):
    hx, hy, hz = C.HANDOVER_XYZ
    goto_xyz(robot, hx, hy, hz, seconds=2.0)
    time.sleep(C.HANDOVER_PAUSE_S)
    set_gripper(robot, C.GRIPPER_OPEN)
    time.sleep(0.4)
    goto_xyz(robot, hx, hy, hz + 0.04, seconds=1.0)
    print("  delivered.")


def put_back(robot, x, y, z0):
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.6)
    goto_xyz(robot, x, y, z0 + C.GRASP_HEIGHT + 0.004, seconds=1.2)
    set_gripper(robot, C.GRIPPER_OPEN)
    time.sleep(0.3)
    goto_xyz(robot, x, y, z0 + C.PICK_HOVER, seconds=1.2)
    print("  returned.")


def main():
    from ultralytics import YOLO
    model = YOLO(C.YOLO_MODEL)
    H = load_H()
    z0 = table_z()
    cap = open_cam()
    robot = connect()
    carrying = None          # (x, y) it picked from, while holding something
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect(model, frame)
            for i, (label, conf, (u, v), (x1, y1, x2, y2)) in enumerate(dets[:9]):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(u), int(v)), 5, (0, 255, 255), -1)
                cv2.putText(frame, f"[{i + 1}] {label} {conf:.2f}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status = ("CARRYING -- h = hand over | r = put back"
                      if carrying else "press [n] to pick | q quit")
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
                    put_back(robot, *carrying, z0)
                    carrying = None
            elif ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(dets):
                    label, _conf, (u, v), _bbox = dets[idx]
                    x, y = pixel_to_xy(H, u, v)
                    print(f"[{label}] pixel ({u:.0f},{v:.0f}) -> table ({x:.3f},{y:+.3f})")
                    try:
                        execute_pick(robot, x, y, z0)
                        carrying = (x, y)
                    except K.NotReachable as e:
                        print("  out of reach:", e)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        safe_park(robot)


if __name__ == "__main__":
    main()
