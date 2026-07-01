"""
follow.py — Phase 1: Dum-E follows your hand (2-joint visual servoing).

    camera -> MediaPipe hand -> pixel error -> pan (shoulder_pan) + tilt (wrist_flex)

Non-tracking joints hold their startup pose. 'q' or Ctrl-C releases torque cleanly.

FIRST-RUN CHECKLIST
  * You have already run `lerobot-calibrate` for this robot id.
  * Workspace clear, hand near the power switch.
  * If it steers AWAY from your hand -> flip SIGN_PAN / SIGN_TILT in config.py.
  * Too twitchy -> lower GAIN_*.   Too sluggish -> raise GAIN_* or SMOOTHING.
"""
import time
import platform

import cv2
import mediapipe as mp

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import config as C


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    # ---- arm ----
    robot = SO101Follower(SO101FollowerConfig(
        port=C.PORT,
        id=C.ROBOT_ID,
        use_degrees=C.USE_DEGREES,
        max_relative_target=C.MAX_STEP_DEG,
    ))
    print(f"Connecting to arm on {C.PORT} ...")
    robot.connect()  # loads calibration saved by lerobot-calibrate
    start = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    cmd = dict(start)      # the pose we send, smoothed toward `target`
    target = dict(start)   # goal the hand-tracker pushes around
    pan = C.PAN_JOINT + ".pos"
    tilt = C.TILT_JOINT + ".pos"
    print("Connected. Show your hand. Press 'q' or Ctrl-C to stop.")

    # ---- camera ----
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        robot.disconnect()
        raise SystemExit(
            f"Camera {C.CAMERA_INDEX} did not open. Try another CAMERA_INDEX in config.py"
        )

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

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

            if res.multi_hand_landmarks:
                wrist = res.multi_hand_landmarks[0].landmark[0]
                u, v = wrist.x * w, wrist.y * h
                err_x = u - w / 2   # +ve => hand is to the right of center
                err_y = v - h / 2   # +ve => hand is below center
                if abs(err_x) > C.DEADZONE_PX:
                    target[pan] += C.SIGN_PAN * C.GAIN_PAN * err_x
                if abs(err_y) > C.DEADZONE_PX:
                    target[tilt] += C.SIGN_TILT * C.GAIN_TILT * err_y
                # keep the goal within a safe sweep of the startup pose (anti-windup)
                target[pan] = clamp(target[pan], start[pan] - C.PAN_LIMIT_DEG, start[pan] + C.PAN_LIMIT_DEG)
                target[tilt] = clamp(target[tilt], start[tilt] - C.TILT_LIMIT_DEG, start[tilt] + C.TILT_LIMIT_DEG)
                cv2.circle(frame, (int(u), int(v)), 8, (0, 255, 0), -1)

            # smooth the commanded pose toward the target (glide, not snap)
            for k in (pan, tilt):
                cmd[k] += C.SMOOTHING * (target[k] - cmd[k])
            # non-tracking joints stay at the startup pose (already in `cmd`)
            robot.send_action(cmd)

            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.imshow("Dum-E FOLLOW  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Stopped. Torque released.")


if __name__ == "__main__":
    main()
