"""
ik_test.py — Stage 1 validation: teach the IK your arm's conventions, then prove it.

Run the modes IN THIS ORDER:

  python ik_test.py capture
      One-time. (1) Torque goes off; you pose the WHOLE arm straight up (upper arm,
      forearm and gripper all vertical, fingertip to the ceiling) and press Enter --
      that reading defines each joint's OFFSET. (2) For each joint the script jogs
      +10 deg and asks which way it moved -- that defines each SIGN.
      Saves arm_geom.json. Eyeball precision is fine: X,Y residuals get absorbed by
      the homography later, Z by `touch`.

  python ik_test.py points
      IK-commands the fingertip to a few hover points over the table (gripper down).
      You confirm each with Enter and eyeball that it goes where claimed. THE GATE:
      if these land roughly right (couple of cm), Stage 1 is done.

  python ik_test.py touch
      Hovers at one point, then you jog the fingertip down in small steps until it
      just touches the table. Saves TABLE_Z into arm_geom.json -- the real-world
      z-correction that the pick will descend to.

  python ik_test.py goto X Y Z
      Direct move for debugging (meters, robot frame: origin at the pan axis on the
      table, +x = arm-forward at pan 0).

Ctrl-C safe: parks low and releases torque.
"""
import sys
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import config as C
import kinematics as K

HZ = 25


def connect():
    robot = SO101Follower(SO101FollowerConfig(
        port=C.PORT, id=C.ROBOT_ID, use_degrees=True,
        max_relative_target=C.MAX_STEP_DEG,
    ))
    print(f"Connecting on {C.PORT} ...")
    robot.connect()
    print("Connected.")
    return robot


def pose_now(robot):
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


def ramp_to(robot, target: dict, seconds=2.0):
    """Glide from the current pose to `target` in small steps (stays under the
    safety clamp, and slow enough not to ring the base)."""
    cur = pose_now(robot)
    full = dict(cur)
    steps = max(1, int(HZ * seconds))
    for i in range(1, steps + 1):
        a = i / steps
        for k, v in target.items():
            full[k] = cur[k] + a * (v - cur[k])
        robot.send_action(full)
        time.sleep(1.0 / HZ)


def park(robot):
    try:
        gm = K.load_geom()
        # tucked: slight fold, low over the base -- never a cantilevered faceplant
        tucked = K.geom_to_robot(
            {"shoulder_pan": 0.0, "shoulder_lift": 15.0, "elbow_flex": 130.0,
             "wrist_flex": 35.0}, gm)
        ramp_to(robot, tucked, seconds=2.5)
        print("Parked tucked.")
    except SystemExit:
        pass  # no geom captured yet -- release where it stands
    except Exception as e:  # noqa: BLE001
        print(f"park warning: {e}")
    finally:
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] torque-disable failed: {e} (if 'Overload': power-cycle the arm)")


def goto_xyz(robot, x, y, z, seconds=2.5):
    gm = K.load_geom()
    geom = K.ik_vertical(x, y, z)
    print("  geom:", {k: round(v, 1) for k, v in geom.items()})
    target = K.geom_to_robot(geom, gm)
    ramp_to(robot, target, seconds)


# ────────────────────────── modes ──────────────────────────

def mode_capture(robot):
    print(
        "\n== CAPTURE ==\n"
        "1) I'll release torque. Pose the ENTIRE arm STRAIGHT UP -- upper arm,\n"
        "   forearm and gripper all vertical, fingertip pointing at the ceiling.\n"
        "   (A set square / phone level against the links helps. ±3-5° is fine.)\n"
        "2) Hold it there and press Enter.\n"
    )
    robot.bus.disable_torque()
    input("torque OFF -- pose it straight up, hold, then Enter... ")
    zero = pose_now(robot)
    robot.bus.enable_torque()
    robot.send_action(zero)          # hold the pose so it doesn't fall
    print("Captured. Now three quick direction questions.\n")

    geom = {j: {"sign": 1, "offset": zero[j + ".pos"]}
            for j in K.GEOM_JOINTS}
    geom["wrist_roll"] = {"sign": 1, "offset": zero["wrist_roll.pos"]}

    # pan sign: geometric + must be counter-clockwise seen from above (+x toward +y).
    questions = [
        ("shoulder_pan",  "Did the arm rotate COUNTER-CLOCKWISE (seen from above)? [y/n] "),
        ("shoulder_lift", "Did the arm LEAN AWAY from vertical (toward the table)? [y/n] "),
        ("elbow_flex",    "Did the FOREARM FOLD toward the upper arm? [y/n] "),
        ("wrist_flex",    "Did the GRIPPER FOLD further inward? [y/n] "),
    ]
    for joint, q in questions:
        key = joint + ".pos"
        jog = dict(zero)
        jog[key] = zero[key] + 10.0
        ramp_to(robot, {key: jog[key]}, seconds=0.8)
        ans = input(f"  {joint}: {q}").strip().lower()
        geom[joint]["sign"] = 1 if ans.startswith("y") else -1
        ramp_to(robot, {key: zero[key]}, seconds=0.8)

    K.save_geom(geom)
    print("\nDone. Next:  python ik_test.py points")


def mode_points(robot):
    gm = K.load_geom()
    z_hover = gm.get("table_z", 0.0) + 0.06     # 6 cm above (calibrated or nominal) table
    pts = [(0.18, 0.00), (0.16, -0.10), (0.16, 0.10), (0.23, 0.00)]
    print("\n== POINTS ==  fingertip should hover ~6 cm above the table at each spot,")
    print("gripper pointing straight down. Eyeball each; a couple of cm off is FINE.\n")
    for i, (x, y) in enumerate(pts, 1):
        input(f"[{i}/{len(pts)}] move to x={x:.2f} y={y:+.2f} -- Enter... ")
        try:
            goto_xyz(robot, x, y, z_hover)
        except K.NotReachable as e:
            print("  skipped:", e)
    print("\nIf those looked right, Stage 1 gate PASSED. Next: python ik_test.py touch")


def mode_touch(robot):
    gm = K.load_geom()
    x, y = 0.18, 0.0
    z = 0.05
    print("\n== TOUCH ==  jog the fingertip down until it JUST touches the table.")
    print("keys: d = down 3mm   u = up 3mm   Enter = it's touching\n")
    goto_xyz(robot, x, y, z)
    while True:
        k = input("d/u/Enter> ").strip().lower()
        if k == "d":
            z -= 0.003
        elif k == "u":
            z += 0.003
        elif k == "":
            break
        else:
            continue
        try:
            goto_xyz(robot, x, y, z, seconds=0.5)
        except K.NotReachable as e:
            print("  ", e)
            z += 0.003
    gm["table_z"] = z
    K.save_geom(gm)
    print(f"\nTABLE_Z = {z:+.4f} m saved. The pick will descend to this. Stage 1 complete.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "points"
    robot = connect()
    try:
        if mode == "capture":
            mode_capture(robot)
        elif mode == "points":
            mode_points(robot)
        elif mode == "touch":
            mode_touch(robot)
        elif mode == "goto":
            x, y, z = (float(v) for v in sys.argv[2:5])
            goto_xyz(robot, x, y, z)
            input("holding -- Enter to park... ")
        else:
            print("modes: capture | points | touch | goto X Y Z")
    except KeyboardInterrupt:
        pass
    finally:
        park(robot)


if __name__ == "__main__":
    main()
