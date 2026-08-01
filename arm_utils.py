"""
arm_utils.py — shared helpers every arm script uses.

The important one is the PARKING RITUAL: on any exit, glide to the user-defined
REST POSE first, then release torque — so the servos aren't left energized (best
practice) and gravity has nothing to drop (no plop).

The rest pose is YOURS, captured once:

    python follow.py --set-rest

(torque releases, you fold the arm into a compact / low / gravity-stable position —
elbow folded, gripper tucked near the base — press Enter, saved to rest_pose.json.)
Every script's exit then reuses it: follow, ik_test, pick, all of them.
"""
import json
import os
import time

import config as C

_REST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.REST_POSE_FILE)
HZ = 25


def connect(max_step=None):
    """Standard follower connection used by all scripts."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    robot = SO101Follower(SO101FollowerConfig(
        port=C.PORT, id=C.ROBOT_ID, use_degrees=True,
        max_relative_target=max_step or C.MAX_STEP_DEG))
    print(f"Connecting on {C.PORT} ...")
    robot.connect()
    print("Connected.")
    return robot


def goto_xyz(robot, x, y, z, seconds=2.5, verbose=False):
    """IK-move the fingertip (TCP) to (x, y, z) meters in the robot frame, gripper
    pointing straight down. Raises kinematics.NotReachable for bad targets."""
    import kinematics as K
    gm = K.load_geom()
    geom = K.ik_vertical(x, y, z)
    if verbose:
        print("  geom:", {k: round(v, 1) for k, v in geom.items()})
    ramp_to(robot, K.geom_to_robot(geom, gm), seconds)


def pose_now(robot):
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


def ramp_to(robot, target: dict, seconds=2.0, hz=HZ):
    """Glide from the current pose to `target` in small interpolated steps (stays
    under the safety clamps; slow enough not to ring the base)."""
    cur = pose_now(robot)
    full = dict(cur)
    steps = max(1, int(hz * seconds))
    for i in range(1, steps + 1):
        a = i / steps
        for k, v in target.items():
            full[k] = cur[k] + a * (v - cur[k])
        robot.send_action(full)
        time.sleep(1.0 / hz)


def load_rest_pose():
    if not os.path.exists(_REST_PATH):
        return None
    with open(_REST_PATH) as f:
        return json.load(f)


def set_rest_interactive(robot):
    """Capture the rest pose from the physical arm (torque off, you pose it)."""
    print(
        "\n== SET REST POSE ==\n"
        "Torque is now OFF. Fold the arm into the position it should settle into on\n"
        "every exit: compact, LOW, gravity-stable (elbow folded, gripper tucked near\n"
        "the base -- so releasing torque drops nothing). Then press Enter.\n"
    )
    robot.bus.disable_torque()
    input("pose it, hold, then Enter... ")
    pose = pose_now(robot)
    with open(_REST_PATH, "w") as f:
        json.dump(pose, f, indent=2)
    print(f"saved {_REST_PATH}")
    for k, v in pose.items():
        print(f"  {k:18s} {v:8.2f}")
    print("\nEvery script now parks here on exit.")


def safe_park(robot, fallback: dict | None = None):
    """The exit ritual: glide to the rest pose (or `fallback` if none captured yet),
    THEN release torque. Best-effort — a park failure must never block the release."""
    try:
        target = load_rest_pose() or fallback
        if target:
            ramp_to(robot, target, seconds=C.PARK_SECONDS)
            print("Parked at rest pose; releasing torque.")
        else:
            print("No rest pose captured (python follow.py --set-rest) -- "
                  "releasing torque in place.")
    except Exception as e:  # noqa: BLE001
        print(f"park warning (releasing torque anyway): {e}")
    finally:
        # An overloaded motor rejects the torque-disable write and would otherwise
        # crash the exit with a traceback. Catch it and point at the fix.
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torque-disable on exit failed: {e}")
            print("       If it says 'Overload', POWER-CYCLE the arm to clear the alarm.")
