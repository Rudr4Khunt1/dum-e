"""
camera_test.py — verify the C920 + MediaPipe hand tracking BEFORE the arm moves.

Opens the camera, tracks one hand, draws it, and prints the tracked pixel + FPS.
Use it to check camera placement/focus and to find the right CAMERA_INDEX.

Press 'q' to quit. (No arm involved — safe to run anytime.)
"""
import time
import platform

import cv2
import mediapipe as mp

import config as C


def main():
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(
            f"Camera {C.CAMERA_INDEX} did not open. Try another CAMERA_INDEX in config.py"
        )

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    draw = mp.solutions.drawing_utils
    conns = mp.solutions.hands.HAND_CONNECTIONS

    n, t0 = 0, time.time()
    print("Show your hand. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, lm, conns)
            wrist = lm.landmark[0]
            u, v = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame, (u, v), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"hand ({u},{v})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        n += 1
        if n % 30 == 0:
            print(f"FPS ~{n / (time.time() - t0):.1f}")

        cv2.imshow("Dum-E camera test  (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
