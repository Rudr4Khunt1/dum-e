"""
follow.py — Phase 1: Dum-E follows your hand (look-at, fixed camera) + first soul.

WHAT IT DOES
  * pan  (shoulder_pan) + tilt (wrist_flex) point the head at your hand
  * roll (wrist_roll)   mirrors your hand's rotation -- turn it like a safe dial
  * personality: it DROOPS when it loses you, PERKS UP when you come back

HOW IT KNOWS WHERE TO LOOK
The camera is bolted next to the arm and both are fixed, so the direction the camera
sees your hand IS (near enough) the direction the arm must point. No 3D, no depth:

    pan  = start_pan  + SIGN_PAN  * K_PAN  * (u - w/2)
    tilt = start_tilt + SIGN_TILT * K_TILT * (v - h/2)

K is deg-of-joint per pixel, derived from the camera FOV (K ~= FOV_deg / pixels).
`start_*` IS the definition of "image center" -- which is why we AIM first.

This is an absolute MAPPING, not error-integration: with a fixed camera, moving the
arm never changes the image, so integrating error has no feedback path and would just
ramp the arm into its clamp.

ANTI-SHAKE (three layers, each attacking a different cause)
  1. palm center (5-landmark mean) -- less landmark noise than the lone wrist point
  2. One-Euro filter on the pixel  -- smooths hard when still, loosens when moving
  3. slew-rate cap on the servos   -- the base carries the whole arm's mass, and the
     printed structure RINGS if you slew it fast. That's the shake that shows up only
     when the base turns; filtering can't fix it, only gentler motion can.

CONTROLS
  q          quit (parks, then releases torque)
  space      FREEZE / unfreeze (emergency stop: hold pose, ignore tracking)
  a / z      flip SIGN_PAN / SIGN_TILT
  [ / ]      decrease / increase K live
  AIM phase: j/l pan, i/k tilt, ENTER to lock the reference
On exit it prints the tuned values to paste into config.py.
"""
import argparse
import time
import platform

import cv2

import config as C
from filters import OneEuro
from hand_tracker import HandTracker, WRIST, palm_center, hand_roll_deg


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_pose(pose, start, limits):
    """The single safety gate: clamp the FINAL (already-mixed) pose to its sweep
    envelope right before it reaches the servos. Follow, personality, and any future
    gesture layers all get summed BEFORE this, so their sum can never drive a joint
    past its limit."""
    for key, lim in limits.items():
        pose[key] = clamp(pose[key], start[key] - lim, start[key] + lim)
    return pose


def pick_hand(hands, last_uv, w, h):
    """You usually have TWO hands in frame -- the commanding one and the one on your
    keyboard. A naive single-hand pick flips between them and the arm jitters. So stay
    locked on whichever hand is nearest the one we were already tracking."""
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
    """Glide back to the startup pose, THEN release torque, so the arm never sags onto
    the desk from an extended pose. Best-effort: a park failure must never prevent the
    torque release."""
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
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torque-disable on exit failed: {e}")
            print("       If it says 'Overload', POWER-CYCLE the arm to clear the alarm.")


def aim_mode(robot, cap, pan, tilt):
    """Jog the head until it points at the image-center crosshair, then lock that pose
    as the reference.

    WHY: the mapping is pan = start_pan + K*err. The startup pose *defines* where
    "image center" is. Start it aimed at your face while the crosshair is on your chest
    and every angle inherits that offset -- it orbits your head forever.
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
        cv2.imshow("Dum-E FOLLOW", frame)

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
    roll = C.ROLL_JOINT + ".pos"

    # Live-tunable copies (keys change these; config holds the defaults).
    k_pan, k_tilt = C.K_PAN, C.K_TILT
    sign_pan, sign_tilt = C.SIGN_PAN, C.SIGN_TILT

    # ---- arm ----
    robot = None
    if dry_run:
        start = {pan: 0.0, tilt: 0.0, roll: 0.0}
        print("[DRY-RUN] No arm. Printing commanded angles; nothing moves.")
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

    limits = {pan: C.PAN_LIMIT_DEG, tilt: C.TILT_LIMIT_DEG}
    if C.ENABLE_ROLL:
        limits[roll] = C.ROLL_LIMIT_DEG

    # One-Euro filters on the raw signals -- jitter dies here, at the source.
    fu = OneEuro(C.FOLLOW_HZ, C.FILTER_MIN_CUTOFF, C.FILTER_BETA)
    fv = OneEuro(C.FOLLOW_HZ, C.FILTER_MIN_CUTOFF, C.FILTER_BETA)
    fr = OneEuro(C.FOLLOW_HZ, C.FILTER_MIN_CUTOFF, C.FILTER_BETA)

    # ---- personality state (soul.md rungs 1-2) ----
    mood = "track"                 # track | droop | perk
    last_seen = time.time()
    perk_until = 0.0

    frozen = False
    last_uv = None
    err_x = err_y = roll_deg = 0.0

    # Slew caps are PER JOINT: only the base rings (it swings the whole arm's mass).
    # The wrist joints move almost nothing and can snap -- capping them at the base's
    # limit is what made the robot feel lazy.
    caps = {}
    for key in (pan, tilt, roll):
        name = key.rsplit(".", 1)[0]
        caps[key] = C.MAX_DEG_PER_SEC.get(name, C.DEFAULT_MAX_DEG_PER_SEC) / C.FOLLOW_HZ

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
            now = time.time()

            # ---- personality: the hand appearing / disappearing IS the emotional beat
            if C.ENABLE_PERSONALITY:
                if lms is not None:
                    if mood == "droop":                 # you came back!
                        mood, perk_until = "perk", now + C.PERK_SECONDS
                        print("\n[perk] there you are!")
                    elif mood == "perk" and now > perk_until:
                        mood = "track"
                    last_seen = now
                elif mood != "droop" and now - last_seen > C.LOST_AFTER_S:
                    mood = "droop"
                    print("\n[droop] ...where'd you go")
            smoothing = {"droop": C.DROOP_SMOOTHING,
                         "perk": C.PERK_SMOOTHING}.get(mood, C.SMOOTHING)

            if lms is not None:
                # palm center (5-landmark mean) is far steadier than the lone wrist
                hx, hy = palm_center(lms) if C.USE_PALM_CENTER else lms[WRIST]
                u_raw, v_raw = hx * w, hy * h
                u, v = fu(u_raw), fv(v_raw)     # One-Euro -> the anti-shake
                last_uv = (u, v)
                err_x = u - w / 2
                err_y = v - h / 2

                if not frozen:
                    # Look-at MAPPING: pixel offset -> ABSOLUTE angle ('=' not '+=')
                    if abs(err_x) > C.DEADZONE_PX:
                        target[pan] = start[pan] + sign_pan * k_pan * err_x
                    if abs(err_y) > C.DEADZONE_PX:
                        target[tilt] = start[tilt] + sign_tilt * k_tilt * err_y
                    if C.ENABLE_ROLL:
                        roll_deg = fr(hand_roll_deg(lms))
                        target[roll] = start[roll] + C.SIGN_ROLL * C.K_ROLL * roll_deg
                    if mood == "perk":          # eager overshoot on re-acquire
                        target[tilt] -= C.PERK_DEG

                cv2.circle(frame, (int(u_raw), int(v_raw)), 5, (0, 140, 255), 1)  # raw
                cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 255), -1)        # filtered
            else:
                last_uv = None
                fu.reset(); fv.reset(); fr.reset()
                if not frozen and mood == "droop":
                    # sad: head sags, faces forward, moves slow and heavy
                    target[tilt] = start[tilt] + C.DROOP_DEG
                    target[pan] = start[pan]

            if not frozen:
                keys = [pan, tilt] + ([roll] if C.ENABLE_ROLL else [])
                for k in keys:
                    step = smoothing * (target[k] - cmd[k])
                    # per-joint slew cap: keeps the heavy base from ringing while letting
                    # the near-massless wrist joints actually snap
                    cmd[k] += clamp(step, -caps[k], caps[k])

            # [safety gate] clamp the final summed pose, right before the servos
            clamp_pose(cmd, start, limits)

            if robot is None:
                if not frozen:
                    print(f"[{mood:5s}] pan {cmd[pan]:7.2f}  tilt {cmd[tilt]:7.2f}  "
                          f"roll {cmd[roll]:7.2f}", end="\r")
            else:
                robot.send_action(cmd)

            # ---- HUD ----
            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            colour = {"droop": (140, 140, 255), "perk": (0, 255, 255)}.get(mood, (0, 255, 0))
            for i, line in enumerate([
                f"[{mood}]  hands:{len(hands)}  err=({err_x:+6.0f},{err_y:+6.0f})px",
                f"pan {cmd[pan]:+7.2f}  tilt {cmd[tilt]:+7.2f}  roll {cmd[roll]:+7.2f}",
                f"K={k_pan:.3f}  sign=({sign_pan:+d},{sign_tilt:+d})  handroll={roll_deg:+.0f}",
            ]):
                cv2.putText(frame, line, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            if frozen:
                cv2.putText(frame, "FROZEN (space to resume)", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Dum-E FOLLOW", frame)
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
                k_pan, k_tilt = k_pan + 0.005, k_tilt + 0.005
                print(f"\nK -> {k_pan:.3f}")
            elif key == ord("["):
                k_pan, k_tilt = max(0.0, k_pan - 0.005), max(0.0, k_tilt - 0.005)
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
