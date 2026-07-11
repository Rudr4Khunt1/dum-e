"""
camera_test.py — verify the C920 + hand tracking BEFORE the arm moves.

Opens the camera, tracks one hand, draws the skeleton, prints the wrist pixel + FPS.
Use it to check camera placement/focus and to find the right CAMERA_INDEX.

Press 'q' to quit. (No arm involved — safe to run anytime.)
"""
import time
import platform

import cv2

import config as C
from hand_tracker import HandTracker, HAND_CONNECTIONS, WRIST


def main():
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(C.CAMERA_INDEX, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(
            f"Camera {C.CAMERA_INDEX} did not open. Try another CAMERA_INDEX in config.py"
        )

    tracker = HandTracker(num_hands=C.NUM_HANDS)
    n, t0 = 0, time.time()
    print("Show your hand. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if C.MIRROR:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            hands = tracker.detect(frame)
            for lms in hands:  # draw every hand it sees (incl. your keyboard hand)
                pts = [(int(x * w), int(y * h)) for (x, y) in lms]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 3, (0, 255, 0), -1)
                u, v = pts[WRIST]
                cv2.circle(frame, (u, v), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"({u},{v})", (u + 12, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"hands: {len(hands)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            n += 1
            if n % 30 == 0:
                print(f"FPS ~{n / (time.time() - t0):.1f}")

            cv2.imshow("Dum-E camera test  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
