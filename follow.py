"""
follow.py — Phase 1: Dum-E follows your hand (2-joint look-at, fixed camera).

HOW IT KNOWS WHERE TO LOOK
--------------------------
The camera is bolted right NEXT TO the arm, and both are fixed. So the direction
the *camera* sees your hand is (near enough) the direction the *arm* must point.
No 3D, no depth needed:

    pixel offset from image center  ->  angle offset from the camera axis
                                    ->  add to the arm's start angle

    pan  = start_pan  + SIGN_PAN  * K_PAN  * (u - w/2)
    tilt = start_tilt + SIGN_TILT * K_TILT * (v - h/2)

K_* is deg-of-joint per pixel, and it is *physically* derivable from the camera's
field of view:  K ~= FOV_degrees / frame_pixels.  For a C920 (78 deg H over 1280 px)
that is ~0.06 deg/px.  See config.py.

IMPORTANT: `start_pan/start_tilt` become the "image center" reference. So pose the
arm pointing at roughly the CENTER of the camera's view before starting, or the
whole mapping is offset by a constant.

This is a look-at MAPPING (absolute), not error-integration: with a fixed camera,
moving the arm never changes the image, so integrating pixel error has no feedback
path and would just ramp the arm into its clamp.

It's approximate (the camera sits a few cm off the arm, so there's parallax) --
which is fine for "look at me", and exactly why the PICK phase needs the homography
instead.

CONTROLS
  q          quit cleanly (parks, then releases torque)
  space      FREEZE / unfreeze -- emergency stop: hold pose, ignore tracking
  a          flip SIGN_PAN     (if it turns the wrong way)
  z          flip SIGN_TILT
  [ / ]      decrease / increase K (both axes) -- live tuning, no restart
  Ctrl-C     same as q
On exit it prints the tuned values so you can paste them into config.py.

FIRST-RUN CHECKLIST
  * `lerobot-calibrate` already done for this robot id.
  * Workspace clear, hand near the power switch.
  * Dry-run first (no arm):  python follow.py --dry-run
  * Pose the arm so its head points at the CENTER of the camera view.
"""
import argparse
import time
import platform

import cv2

import config as C
from hand_tracker import HandTracker, WRIST


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_pose(pose, start, pan, tilt):
    """The single safety gate: clamp the FINAL (already-mixed) pose to the sweep
    envelope right before it reaches the servos. Today only follow moves pan/tilt;
    when the idle/gesture layers (soul.md) are summed in, they get clamped here too
    so their sum can never drive a joint past its limit."""
    pose[pan] = clamp(pose[pan], start[pan] - C.PAN_LIMIT_DEG, start[pan] + C.PAN_LIMIT_DEG)
    pose[tilt] = clamp(pose[tilt], start[tilt] - C.TILT_LIMIT_DEG, start[tilt] + C.TILT_LIMIT_DEG)
    return pose


def pick_hand(hands, last_uv, w, h):
    """Choose which hand to follow.

    You typically have TWO hands in frame -- the one you're commanding with, and
    the one resting on the keyboard. A single-hand detector flips between them and
    the arm jitters. So: stay locked on whichever hand is nearest the one we were
    already tracking; on first sight, take the most confident.
    """
    if not hands:
        return None
    if last_uv is None:
        return hands[0]
    lx, ly = last_uv

    def dist2(hand):
        u, v = hand[WRIST][0] * w, hand[WRIST][1] * h
        return (u - lx) ** 2 + (v - ly) ** 2

    return min(hands, key=dist2)


def safe_park(robot, cmd, start):
    """Glide back to the startup pose, THEN release torque, so the arm never sags
    onto the desk from an extended pose. Best-effort: a park failure must never stop
    the torque release."""
    try:
        steps = max(1, int(C.FOLLOW_HZ * C.PARK_SECONDS))
        period = 1.0 / C.FOLLOW_HZ
        for i in range(1, steps + 1):
            a = i / steps
            pose = {k: cmd[k] + a * (start[k] - cmd[k]) for k in cmd}
            robot.send_action(pose)
            time.sleep(period)
        print("Parked at rest pose.")
    except Exception as e:  # noqa: BLE001
        print(f"safe_park warning (releasing torque anyway): {e}")
    finally:
        # An overloaded motor rejects the torque-disable write and would otherwise
        # crash the exit with a traceback. Catch it and point at the fix.
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torque-disable on exit failed: {e}")
            print("       If it says 'Overload', POWER-CYCLE the arm to clear the alarm.")


def aim_mode(robot, cap, pan, tilt):
    """Jog the head until it points at the image-center crosshair, then lock that
    pose in as the reference.

    WHY THIS EXISTS: the mapping is  pan = start_pan + K * err_x.  So the startup
    pose *defines* where "image center" is. If the arm starts aimed at your face
    while the crosshair sits on your chest, every angle inherits that offset and
    the head orbits your head instead of landing on your hand.
    """
    pose = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    print("\n== AIM ==  point the head at the RED crosshair (image center)")
    print("   j / l  pan left / right      i / k  tilt up / down      ENTER  lock it in")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if C.MIRROR:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        for i, line in enumerate([
            "AIM: point the head at the RED cross, then ENTER",
            f"pan {pose[pan]:+7.2f}   tilt {pose[tilt]:+7.2f}  deg",
            "j/l = pan   i/k = tilt   ENTER = lock",
        ]):
            cv2.putText(frame, line, (10, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Dum-E FOLLOW  (q quit, space freeze, a/z signs, [ ] K)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):           # ENTER
            break
        elif key == ord("j"):
            pose[pan] -= C.AIM_STEP_DEG
        elif key == ord("l"):
            pose[pan] += C.AIM_STEP_DEG
        elif key == ord("i"):
            pose[tilt] -= C.AIM_STEP_DEG
        elif key == ord("k"):
            pose[tilt] += C.AIM_STEP_DEG
        else:
            continue
        robot.send_action(pose)

    print(f"Reference locked: pan {pose[pan]:+.2f}  tilt {pose[tilt]:+.2f}\n")
    return pose


def main():
    ap = argparse.ArgumentParser(description="Dum-E FOLLOW loop")
    ap.add_argument("--dry-run", action="store_true",
                    help="no arm: print commanded angles instead of moving servos")
    args = ap.parse_args()
    dry_run = args.dry_run

    pan = C.PAN_JOINT + ".pos"
    tilt = C.TILT_JOINT + ".pos"

    # Live-tunable copies (keys can change these; config holds the defaults).
    k_pan, k_tilt = C.K_PAN, C.K_TILT
    sign_pan, sign_tilt = C.SIGN_PAN, C.SIGN_TILT

    # ---- arm ----
    robot = None
    if dry_run:
        start = {pan: 0.0, tilt: 0.0}
        print("[DRY-RUN] No arm. Printing commanded pan/tilt; nothing moves.")
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(
            port=C.PORT,
            id=C.ROBOT_ID,
            use_degrees=C.USE_DEGREES,
            max_relative_target=C.MAX_STEP_DEG,
        ))
        print(f"Connecting to arm on {C.PORT} ...")
        robot.connect()
        start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}

    # ---- camera + tracker ----
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        if robot is not None:
            safe_park(robot, dict(start), start)
        raise SystemExit(
            f"Camera {C.CAMERA_INDEX} did not open. Try another CAMERA_INDEX in config.py"
        )

    tracker = HandTracker(num_hands=C.NUM_HANDS)

    # ---- aim: define where "image center" is, in joint angles ----
    if robot is not None and C.AIM_ON_START:
        start = aim_mode(robot, cap, pan, tilt)
    if robot is not None:
        print("Tracking. q quit | space freeze | a/z flip signs | [ ] adjust K")

    cmd = dict(start)
    target = dict(start)

    frozen = False
    last_uv = None            # the hand we're locked onto (for continuity)
    err_x = err_y = 0.0
    period = 1.0 / C.FOLLOW_HZ

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            if C.MIRROR:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            hands = tracker.detect(frame)
            lms = pick_hand(hands, last_uv, w, h)

            if lms is not None:
                u, v = lms[WRIST][0] * w, lms[WRIST][1] * h
                last_uv = (u, v)
                err_x = u - w / 2   # +ve => hand right of center
                err_y = v - h / 2   # +ve => hand below center

                if not frozen:
                    # Look-at MAPPING: pixel offset -> ABSOLUTE joint angle ('=' not '+=')
                    if abs(err_x) > C.DEADZONE_PX:
                        target[pan] = start[pan] + sign_pan * k_pan * err_x
                    if abs(err_y) > C.DEADZONE_PX:
                        target[tilt] = start[tilt] + sign_tilt * k_tilt * err_y

                cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 255), -1)  # locked hand
            else:
                last_uv = None  # lost it; re-acquire fresh next time

            if not frozen:
                for k in (pan, tilt):
                    cmd[k] += C.SMOOTHING * (target[k] - cmd[k])

            # [safety gate] clamp the final pose AFTER mixing, before the servos
            clamp_pose(cmd, start, pan, tilt)

            if robot is None:
                if not frozen:
                    print(f"pan {cmd[pan]:7.2f}  tilt {cmd[tilt]:7.2f}", end="\r")
            else:
                robot.send_action(cmd)

            # ---- HUD (so you can tune by eye, not by guesswork) ----
            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            hud = [
                f"hands:{len(hands)}  err=({err_x:+6.0f},{err_y:+6.0f})px",
                f"pan {cmd[pan]:+7.2f}  tilt {cmd[tilt]:+7.2f}  deg",
                f"K={k_pan:.3f}  sign=({sign_pan:+d},{sign_tilt:+d})",
            ]
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if frozen:
                cv2.putText(frame, "FROZEN (space to resume)", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Dum-E FOLLOW  (q quit, space freeze, a/z signs, [ ] K)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                frozen = not frozen
                print("\n[FROZEN] holding pose." if frozen else "\n[RESUMED] tracking.")
            elif key == ord("a"):
                sign_pan = -sign_pan
                print(f"\nSIGN_PAN -> {sign_pan:+d}")
            elif key == ord("z"):
                sign_tilt = -sign_tilt
                print(f"\nSIGN_TILT -> {sign_tilt:+d}")
            elif key == ord("]"):
                k_pan, k_tilt = k_pan + 0.01, k_tilt + 0.01
                print(f"\nK -> {k_pan:.3f}")
            elif key == ord("["):
                k_pan, k_tilt = max(0.0, k_pan - 0.01), max(0.0, k_tilt - 0.01)
                print(f"\nK -> {k_pan:.3f}")

            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        if robot is not None:
            safe_park(robot, cmd, start)
            print("Stopped. Torque released.")
        else:
            print("\n[DRY-RUN] done.")
        print("\n--- tuned values (paste into config.py) ---")
        print(f"K_PAN,  K_TILT  = {k_pan:.3f}, {k_tilt:.3f}")
        print(f"SIGN_PAN,  SIGN_TILT  = {sign_pan:+d}, {sign_tilt:+d}")


if __name__ == "__main__":
    main()
