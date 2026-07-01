"""
control_test.py — Phase 0 gate: prove the arm obeys CODE (not just your hand).

Connects to the follower, prints the live joint angles, then nudges shoulder_pan
+8 deg and back so you can confirm the full code -> serial -> servo path works.

Run AFTER `lerobot-calibrate`. Ctrl-C is safe.

SAFETY: clear the workspace; keep a hand near the power switch. First powered
motion is when a wrong number shows itself.
"""
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import config as C


def main():
    robot = SO101Follower(SO101FollowerConfig(
        port=C.PORT,
        id=C.ROBOT_ID,
        use_degrees=C.USE_DEGREES,
        max_relative_target=C.MAX_STEP_DEG,
    ))
    print(f"Connecting on {C.PORT} ...")
    robot.connect()  # loads the calibration saved by lerobot-calibrate
    print("Connected.\n")
    try:
        obs = robot.get_observation()
        action = {k: v for k, v in obs.items() if k.endswith(".pos")}

        print("Current joint angles (deg):")
        for k, v in action.items():
            print(f"  {k:18s} {v:8.2f}")

        j = "shoulder_pan.pos"
        home = action[j]
        for target in (home + 8.0, home):
            print(f"\nMoving {j} -> {target:.1f}")
            action[j] = target
            robot.send_action(action)
            time.sleep(1.5)

        print("\n[OK] Arm obeyed the code. Phase 0 complete.")
    finally:
        robot.disconnect()
        print("Disconnected (torque released).")


if __name__ == "__main__":
    main()
