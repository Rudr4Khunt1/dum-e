"""
control_test.py — exercise the arm through code with visible motion.

Sweeps three joints so you can watch it move:
  * shoulder_pan (motor 1) — base, side to side
  * wrist_roll   (motor 5) — gripper rotate
  * gripper      (motor 6) — mouth open / close

WHY RAMP: config.MAX_STEP_DEG caps how far one send_action may move (safety).
So to cover a BIG angle we step toward the target in small increments -> smooth,
large motion without removing the clamp. Tune the *_SWEEP / GRIP_OPEN numbers below.

Run AFTER lerobot-calibrate. Ctrl-C safe. Clear the workspace first.

Joint names <-> motor IDs:
  1 shoulder_pan  2 shoulder_lift  3 elbow_flex  4 wrist_flex  5 wrist_roll  6 gripper
"""
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import config as C

# --- tune these (degrees) ---
PAN_SWEEP = 30.0     # base swings +/- this from its start angle
ROLL_SWEEP = 45.0    # wrist rotates +/- this
GRIP_OPEN = 40.0     # how far to open the gripper from its start (flip sign if it closes)
HZ = 25              # ramp update rate


def move_to(robot, joint, target, step=None):
    """Ramp `joint` (e.g. 'shoulder_pan') to `target` degrees in increments no
    larger than MAX_STEP_DEG, so it covers a big range while staying under the
    safety clamp. All other joints hold their current position."""
    key = joint + ".pos"
    step = step or C.MAX_STEP_DEG
    period = 1.0 / HZ
    pose = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
    cur = pose[key]
    n = max(1, int(abs(target - cur) / step))
    for i in range(1, n + 1):
        pose[key] = cur + (target - cur) * i / n
        robot.send_action(pose)
        time.sleep(period)


def main():
    robot = SO101Follower(SO101FollowerConfig(
        port=C.PORT,
        id=C.ROBOT_ID,
        use_degrees=C.USE_DEGREES,
        max_relative_target=C.MAX_STEP_DEG,
    ))
    print(f"Connecting on {C.PORT} ...")
    robot.connect()
    print("Connected.\n")
    try:
        obs = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        print("Current joint angles (deg):")
        for k, v in obs.items():
            print(f"  {k:18s} {v:8.2f}")

        pan0 = obs["shoulder_pan.pos"]
        roll0 = obs["wrist_roll.pos"]
        grip0 = obs["gripper.pos"]

        print("\n[1] base (shoulder_pan): side to side")
        move_to(robot, "shoulder_pan", pan0 - PAN_SWEEP); time.sleep(0.3)
        move_to(robot, "shoulder_pan", pan0 + PAN_SWEEP); time.sleep(0.3)
        move_to(robot, "shoulder_pan", pan0)

        print("[5] wrist_roll: rotate")
        move_to(robot, "wrist_roll", roll0 - ROLL_SWEEP); time.sleep(0.3)
        move_to(robot, "wrist_roll", roll0 + ROLL_SWEEP); time.sleep(0.3)
        move_to(robot, "wrist_roll", roll0)

        print("[6] gripper: open / close")
        move_to(robot, "gripper", grip0 + GRIP_OPEN); time.sleep(0.4)   # open
        move_to(robot, "gripper", grip0)                                 # close

        print("\n[OK] Sweep done. The arm obeyed the code.")
    finally:
        # Hardened: an overloaded motor rejects torque-disable and would crash the
        # exit. Catch it and point at the fix instead.
        try:
            robot.disconnect()
            print("Disconnected (torque released).")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torque-disable on exit failed: {e}")
            print("       If it says 'Overload', POWER-CYCLE the arm to clear the alarm.")


if __name__ == "__main__":
    main()
