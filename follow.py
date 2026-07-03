"""
follow.py — Phase 1: Dum-E follows your hand (2-joint look-at, fixed camera).

    camera -> MediaPipe hand -> pixel offset -> pan (shoulder_pan) + tilt (wrist_flex)

The camera is FIXED and side-offset, so moving the arm never changes the image.
That means "follow" is a look-at MAPPING (a hand at pixel X maps to a fixed joint
angle), NOT error-integration -- integrating pixel error with a fixed camera has
no feedback path and just ramps the arm into its clamp. See config.py K_PAN/K_TILT.

Non-tracking joints hold their startup pose. On exit the arm parks (safe_park)
before torque is released, so it never sags onto the desk.

CONTROLS
  q      quit cleanly (parks, then releases torque)
  space  FREEZE / unfreeze -- emergency stop: hold current pose, ignore tracking
  Ctrl-C same as q

FIRST-RUN CHECKLIST
  * You have already run `lerobot-calibrate` for this robot id.
  * Workspace clear, hand near the power switch.
  * Dry-run it first with no arm:  python follow.py --dry-run
  * If it steers AWAY from your hand -> flip SIGN_PAN / SIGN_TILT in config.py.
  * Too twitchy -> lower K_*.   Too sluggish -> raise K_* or SMOOTHING.
"""
import argparse
import time
import platform

import cv2
import mediapipe as mp

import config as C


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_pose(pose, start, pan, tilt):
    """The single safety gate: clamp the FINAL (already-mixed) pose to the sweep
    envelope right before it reaches the servos. Today only follow moves pan/tilt;
    when the idle/gesture layers (soul.md) are summed in, they get clamped here too
    so their sum can never drive a joint past its limit. (Table-plane keep-out will
    also live here once we have forward kinematics.)"""
    pose[pan] = clamp(pose[pan], start[pan] - C.PAN_LIMIT_DEG, start[pan] + C.PAN_LIMIT_DEG)
    pose[tilt] = clamp(pose[tilt], start[tilt] - C.TILT_LIMIT_DEG, start[tilt] + C.TILT_LIMIT_DEG)
    return pose


def safe_park(robot, cmd, start):
    """Glide every joint back to the startup rest pose, THEN release torque. Called
    on every exit path (q / Ctrl-C / exception). A bare disconnect() drops torque
    wherever the arm sits and it sags -- this prevents that. Best-effort: never let
    a park failure stop us from releasing torque."""
    try:
        steps = max(1, int(C.FOLLOW_HZ * C.PARK_SECONDS))
        period = 1.0 / C.FOLLOW_HZ
        for i in range(1, steps + 1):
            a = i / steps
            pose = {k: cmd[k] + a * (start[k] - cmd[k]) for k in cmd}
            robot.send_action(pose)
            time.sleep(period)
        print("Parked at rest pose.")
    except Exception as e:  # noqa: BLE001 -- park is best-effort; torque must still release
        print(f"safe_park warning (releasing torque anyway): {e}")
    finally:
        robot.disconnect()


def main():
    ap = argparse.ArgumentParser(description="Dum-E FOLLOW loop")
    ap.add_argument("--dry-run", action="store_true",
                    help="no arm: print commanded angles instead of moving servos "
                         "(validates the look-at mapping + clamps at your desk)")
    args = ap.parse_args()
    dry_run = args.dry_run

    pan = C.PAN_JOINT + ".pos"
    tilt = C.TILT_JOINT + ".pos"

    # ---- arm ----
    robot = None
    if dry_run:
        # No hardware: fabricate a startup pose so the mapping + clamps still run.
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
        robot.connect()  # loads calibration saved by lerobot-calibrate
        start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        print("Connected. Show your hand. 'q' quit, space = freeze, Ctrl-C stop.")

    cmd = dict(start)      # the pose we send, smoothed toward `target`
    target = dict(start)   # goal the hand-tracker maps to

    # ---- camera ----
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        if robot is not None:
            safe_park(robot, cmd, start)
        raise SystemExit(
            f"Camera {C.CAMERA_INDEX} did not open. Try another CAMERA_INDEX in config.py"
        )

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    frozen = False
    period = 1.0 / C.FOLLOW_HZ
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror = natural
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks and not frozen:
                wrist = res.multi_hand_landmarks[0].landmark[0]
                u, v = wrist.x * w, wrist.y * h
                err_x = u - w / 2   # +ve => hand is right of center
                err_y = v - h / 2   # +ve => hand is below center
                # Look-at MAPPING: pixel offset -> ABSOLUTE joint angle (note '=', not '+=').
                # Outside the deadzone only, so tiny errors near center don't twitch.
                if abs(err_x) > C.DEADZONE_PX:
                    target[pan] = start[pan] + C.SIGN_PAN * C.K_PAN * err_x
                if abs(err_y) > C.DEADZONE_PX:
                    target[tilt] = start[tilt] + C.SIGN_TILT * C.K_TILT * err_y
                cv2.circle(frame, (int(u), int(v)), 8, (0, 255, 0), -1)

            if not frozen:
                # smooth the commanded pose toward the target (glide, not snap)
                for k in (pan, tilt):
                    cmd[k] += C.SMOOTHING * (target[k] - cmd[k])

            # [safety gate] clamp the final summed pose AFTER mixing, before servos
            clamp_pose(cmd, start, pan, tilt)

            if robot is None:  # dry-run
                if not frozen:
                    print(f"pan {cmd[pan]:7.2f}  tilt {cmd[tilt]:7.2f}", end="\r")
            else:
                robot.send_action(cmd)

            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            if frozen:
                cv2.putText(frame, "FROZEN (space to resume)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Dum-E FOLLOW  (q quit, space freeze)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                frozen = not frozen
                print("\n[FROZEN] holding pose." if frozen else "\n[RESUMED] tracking.")
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot is not None:
            safe_park(robot, cmd, start)
            print("Stopped. Torque released.")
        else:
            print("\n[DRY-RUN] done.")


if __name__ == "__main__":
    main()
