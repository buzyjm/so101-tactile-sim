"""Interactive keyboard teleoperation for the SO-101 arm in Isaac Sim.

Two control modes share the same USD drive targets used by ``grasp_v2.py``:

* ``joint``      -- jog one selected joint at a time.
* ``cartesian``  -- translate the gripper frame in world XYZ; joint targets
                    come from the same Lula solver and descriptor the scripted
                    grasp uses, warm started from the current pose.

The script needs a real window: keyboard input is read through the carb input
interface of the Kit app window, so it cannot run headless or over the
``--no-window`` WebRTC stream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--usd",
    default=str(PROJECT_ROOT / "lab_scene_task.usda"),
    help="Stage to open. Defaults to the task-ready layout.",
)
parser.add_argument(
    "--mode",
    choices=["joint", "cartesian"],
    default="joint",
    help="Initial control mode; TAB switches at runtime.",
)
parser.add_argument(
    "--joint-speed",
    type=float,
    default=45.0,
    help="Joint jog rate in degrees per second.",
)
parser.add_argument(
    "--linear-speed",
    type=float,
    default=0.08,
    help="Cartesian translation rate in metres per second.",
)
parser.add_argument(
    "--gripper-speed",
    type=float,
    default=90.0,
    help="Gripper open/close rate in degrees per second.",
)
parser.add_argument(
    "--self-test",
    type=int,
    default=0,
    metavar="STEPS",
    help=(
        "Run STEPS frames of synthetic input in both modes and exit, instead "
        "of waiting for the keyboard. Used to regress the control path "
        "without a human at the window."
    ),
)
args = parser.parse_args()
# SimulationApp forwards anything left in sys.argv to the underlying Kit app,
# which does not know these flags.
sys.argv = sys.argv[:1]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": False, "active_gpu": 0, "physics_gpu": 0})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

import os  # noqa: E402

from scene_config import (  # noqa: E402
    ROBOT_BASE_POSITION,
    ROBOT_BASE_YAW_DEG,
)

JOINT_ROOT = "/World/Robot/Physics"
FRAME = "gripper_frame_link"
GRIPPER = "gripper"
DESCRIPTOR = str(PROJECT_ROOT / "so101_descriptor.yaml")
URDF = os.environ.get(
    "SO101_URDF_PATH",
    str(
        PROJECT_ROOT.parent
        / "SO-ARM100"
        / "Simulation"
        / "SO101"
        / "so101_new_calib.urdf"
    ),
)

# The Lula solver only knows the five arm joints; the gripper is always driven
# directly, in both modes.
solver = LulaKinematicsSolver(robot_description_path=DESCRIPTOR, urdf_path=URDF)
base_yaw = np.radians(ROBOT_BASE_YAW_DEG)
solver.set_robot_base_pose(
    np.array(ROBOT_BASE_POSITION),
    np.array([np.cos(base_yaw / 2.0), 0.0, 0.0, np.sin(base_yaw / 2.0)]),
)
ARM = solver.get_joint_names()
ALL_JOINTS = ARM + [GRIPPER]

omni.usd.get_context().open_stage(args.usd)
stage = omni.usd.get_context().get_stage()

# The scripted entry points drive these same attributes, so teleop stays
# compatible with anything else that reads the stage.
target_attrs = {}
limits = {}
targets = {}
for name in ALL_JOINTS:
    prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}")
    if not prim.IsValid():
        raise RuntimeError(f"joint prim not found: {JOINT_ROOT}/{name}")
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        raise RuntimeError(f"no angular drive on {JOINT_ROOT}/{name}")
    target_attrs[name] = drive.CreateTargetPositionAttr()
    limits[name] = (
        float(prim.GetAttribute("physics:lowerLimit").Get()),
        float(prim.GetAttribute("physics:upperLimit").Get()),
    )
    # Start from whatever the stage authored so teleop never snaps the arm.
    authored = target_attrs[name].Get()
    targets[name] = float(authored) if authored is not None else 0.0


def arm_q() -> np.ndarray:
    """Current arm targets in radians, in solver joint order."""
    return np.radians([targets[n] for n in ARM])


# Cartesian mode holds the tool orientation that was current when it engaged
# and only translates, which keeps the IK branch stable near the table.
tcp_target = None
tcp_rotation = None


def enter_cartesian() -> bool:
    global tcp_target, tcp_rotation
    position, rotation = solver.compute_forward_kinematics(FRAME, arm_q())
    tcp_target = np.array(position, dtype=float)
    tcp_rotation = np.array(rotation, dtype=float)
    return True


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


KB = carb.input.KeyboardInput
JOINT_SELECT = {
    KB.KEY_1: 0, KB.KEY_2: 1, KB.KEY_3: 2,
    KB.KEY_4: 3, KB.KEY_5: 4, KB.KEY_6: 5,
}
# Held-key axes: each entry is (key, key) for (+, -).
JOINT_AXIS = (KB.UP, KB.DOWN)
CARTESIAN_AXES = [
    (KB.W, KB.S),   # world X
    (KB.A, KB.D),   # world Y
    (KB.Q, KB.E),   # world Z
]
WRIST_ROLL_AXIS = (KB.Z, KB.X)
GRIPPER_AXIS = (KB.O, KB.C)

mode = args.mode
selected = 0
held: set = set()
home = dict(targets)
running = True
ik_failures = 0


def print_help() -> None:
    print("=" * 78)
    print(f"mode: {mode}")
    print("  TAB          switch joint / cartesian mode")
    print("  1..6         select joint (joint mode)")
    print("           ", " ".join(f"{i + 1}:{n}" for i, n in enumerate(ALL_JOINTS)))
    print("  UP / DOWN    jog selected joint (joint mode)")
    print("  W/S A/D Q/E  move gripper along world X / Y / Z (cartesian mode)")
    print("  Z / X        jog wrist_roll (cartesian mode)")
    print("  O / C        open / close gripper (both modes)")
    print("  R            return to the pose the stage was loaded with")
    print("  H            print this help")
    print("  ESC          quit")
    print("=" * 78)


def on_key(event, *_) -> bool:
    global mode, selected, running
    key = event.input
    if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held.discard(key)
        return True
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return True

    held.add(key)
    if key == KB.ESCAPE:
        running = False
    elif key == KB.TAB:
        if mode == "joint":
            enter_cartesian()
            mode = "cartesian"
        else:
            mode = "joint"
        print_help()
    elif key == KB.H:
        print_help()
    elif key == KB.R:
        targets.update(home)
        if mode == "cartesian":
            enter_cartesian()
        print("[teleop] returned to load pose")
    elif key in JOINT_SELECT and mode == "joint":
        selected = JOINT_SELECT[key]
        print(f"[teleop] joint {selected + 1}: {ALL_JOINTS[selected]}")
    return True


def axis_value(axis) -> float:
    plus, minus = axis
    return (1.0 if plus in held else 0.0) - (1.0 if minus in held else 0.0)


def clamp(name: str, value: float) -> float:
    lower, upper = limits[name]
    return float(np.clip(value, lower, upper))


input_iface = carb.input.acquire_input_interface()
keyboard = omni.appwindow.get_default_app_window().get_keyboard()
keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)

world = World(stage_units_in_meters=1.0)
world.reset()
if mode == "cartesian":
    enter_cartesian()
print_help()

dt = float(world.get_physics_dt())


def control_step() -> None:
    """Integrate one frame of held-key input into the drive targets."""
    global tcp_target, ik_failures

    gripper_delta = axis_value(GRIPPER_AXIS) * args.gripper_speed * dt
    if gripper_delta:
        targets[GRIPPER] = clamp(GRIPPER, targets[GRIPPER] + gripper_delta)

    if mode == "joint":
        delta = axis_value(JOINT_AXIS) * args.joint_speed * dt
        if delta:
            name = ALL_JOINTS[selected]
            targets[name] = clamp(name, targets[name] + delta)
    else:
        roll = axis_value(WRIST_ROLL_AXIS) * args.joint_speed * dt
        if roll:
            targets["wrist_roll"] = clamp(
                "wrist_roll", targets["wrist_roll"] + roll
            )
        step = np.array([axis_value(a) for a in CARTESIAN_AXES], dtype=float)
        if np.any(step):
            candidate = tcp_target + step * args.linear_speed * dt
            q, ok = solver.compute_inverse_kinematics(
                frame_name=FRAME,
                target_position=candidate,
                target_orientation=rot_to_quat(tcp_rotation),
                warm_start=arm_q(),
                position_tolerance=0.005,
                orientation_tolerance=0.3,
            )
            if ok:
                # Only commit the Cartesian goal that the solver actually
                # reached, so a rejected step cannot accumulate drift.
                tcp_target = candidate
                for name, value in zip(ARM, np.degrees(q)):
                    targets[name] = clamp(name, float(value))
                ik_failures = 0
            else:
                ik_failures += 1
                if ik_failures % 30 == 1:
                    print(f"[teleop] IK unreachable at {np.round(candidate, 4)}")

    for name in ALL_JOINTS:
        target_attrs[name].Set(targets[name])


def self_test(steps: int) -> None:
    """Drive both modes from synthetic held keys and report what moved."""
    global mode, selected

    def run(label: str, keys: set) -> None:
        before = dict(targets)
        held.clear()
        held.update(keys)
        for _ in range(steps):
            control_step()
            world.step(render=True)
        held.clear()
        moved = {
            n: (before[n], targets[n])
            for n in ALL_JOINTS
            if abs(targets[n] - before[n]) > 1e-6
        }
        print(f"[self-test] {label}: {steps} steps")
        if not moved:
            print("             no joint moved")
        for n, (b, a) in moved.items():
            print(f"             {n:14s} {b:8.3f} -> {a:8.3f} deg")

    mode = "joint"
    for index, name in enumerate(ALL_JOINTS):
        selected = index
        run(f"joint mode, jog {name}", {KB.UP})
    run("joint mode, close gripper", {KB.C})

    mode = "cartesian"
    enter_cartesian()
    start = np.array(tcp_target)
    for label, key in (("+X", KB.W), ("-X", KB.S), ("+Y", KB.A),
                       ("-Y", KB.D), ("+Z", KB.Q), ("-Z", KB.E)):
        run(f"cartesian mode, translate {label}", {key})
    print(f"[self-test] TCP {np.round(start, 4)} -> {np.round(tcp_target, 4)}")
    print(f"[self-test] IK failures on the last direction: {ik_failures}")


if args.self_test:
    self_test(args.self_test)
else:
    while running and app.is_running():
        control_step()
        world.step(render=True)

input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
print("[teleop] final targets (deg):")
for name in ALL_JOINTS:
    print(f"  {name:14s} {targets[name]:8.2f}")
app.close()
