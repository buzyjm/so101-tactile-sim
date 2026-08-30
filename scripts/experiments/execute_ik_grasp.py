from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os
import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import cv2
import omni.usd
import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, UsdPhysics, Gf

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
DESCRIPTOR = str(PROJECT_ROOT / "so101_descriptor.yaml")
URDF = os.environ.get(
    "SO101_URDF_PATH",
    str(PROJECT_ROOT.parent / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
)
OUT_DIR = str(PROJECT_ROOT / "renders")
RESOLUTION = (1280, 720)

JOINT_ROOT = "/World/Robot/Physics"
CUBE_PATH = "/World/TargetCube"
FRAME = "gripper_frame_link"

BASE_POS = np.array([-0.20, 0.0, 0.75])
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
CUBE_POS = np.array([0.15, 0.0, 0.775])

os.makedirs(OUT_DIR, exist_ok=True)

solver = LulaKinematicsSolver(robot_description_path=DESCRIPTOR, urdf_path=URDF)
solver.set_robot_base_pose(BASE_POS, BASE_QUAT)
ARM_JOINTS = solver.get_joint_names()

# Gripper pointing straight down: rotate 180 deg about X so the tool -Z axis
# maps onto world -Z. Quaternion is (w, x, y, z).
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def solve(pos, quat=None, tol_pos=0.005, tol_ori=0.5):
    q, ok = solver.compute_inverse_kinematics(
        frame_name=FRAME,
        target_position=pos,
        target_orientation=quat,
        position_tolerance=tol_pos,
        orientation_tolerance=tol_ori if quat is not None else None,
    )
    return q, ok


print("=" * 78)
print("IK SOLUTIONS AND RESULTING GRIPPER ORIENTATION")
print("=" * 78)

WAYPOINTS = {
    "approach": CUBE_POS + np.array([0.0, 0.0, 0.12]),
    "pregrasp": CUBE_POS + np.array([0.0, 0.0, 0.06]),
    "grasp": CUBE_POS + np.array([0.0, 0.0, 0.01]),
    "lift": CUBE_POS + np.array([0.0, 0.0, 0.15]),
}

solutions = {}
for label, pos in WAYPOINTS.items():
    q, ok = solve(pos, DOWN_QUAT)
    tag = "with down-orientation"
    if not ok:
        q, ok = solve(pos, None)
        tag = "POSITION ONLY (down-orientation failed)"
    solutions[label] = (q, ok)
    print(f"\n[{label}] {pos}  -> success={ok}  ({tag})")
    if ok:
        fk_pos, fk_rot = solver.compute_forward_kinematics(FRAME, q)
        # Third column of the rotation matrix is the tool Z axis in world frame
        tool_z = np.asarray(fk_rot)[:, 2]
        print(f"    joints(deg): {np.round(np.degrees(q), 2)}")
        print(f"    tool Z axis in world: {np.round(tool_z, 3)}")

print("=" * 78)

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

drives = {}
for name in ARM_JOINTS + ["gripper"]:
    prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{name}")
    drives[name] = UsdPhysics.DriveAPI.Get(prim, "angular")

eye = Gf.Vec3d(0.75, -0.70, 1.10)
view = Gf.Matrix4d()
view.SetLookAt(eye, Gf.Vec3d(0.10, 0.0, 0.82), Gf.Vec3d(0, 0, 1))
cam = UsdGeom.Camera.Define(stage, "/World/Cameras/Grasp")
cam.CreateFocalLengthAttr(24.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse())

world = World(stage_units_in_meters=1.0)
world.reset()

render_product = rep.create.render_product("/World/Cameras/Grasp", RESOLUTION)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])
for _ in range(20):
    world.step(render=True)


def cube_z():
    m = UsdGeom.Xformable(stage.GetPrimAtPath(CUBE_PATH)).ComputeLocalToWorldTransform(0)
    return m.ExtractTranslation()[2]


def go(label, gripper_deg, n_steps):
    q, ok = solutions[label]
    if ok:
        # USD drive targets are in DEGREES, Lula returns RADIANS
        for name, val in zip(ARM_JOINTS, np.degrees(q)):
            drives[name].CreateTargetPositionAttr().Set(float(val))
    drives["gripper"].CreateTargetPositionAttr().Set(float(gripper_deg))
    for _ in range(n_steps):
        world.step(render=True)
        d = rgb.get_data()
        if d is not None and d.size:
            frames.append(d[:, :, :3].copy())
    print(f"[{label:9s}] cube_z = {cube_z():.4f}")


frames = []
z_start = cube_z()
go("approach", 60, 90)
go("pregrasp", 60, 70)
go("grasp", 60, 70)
go("grasp", 2, 70)
go("lift", 2, 100)
z_end = cube_z()

print()
print("=" * 78)
print(f"cube z: {z_start:.4f} -> {z_end:.4f}   rise = {z_end - z_start:+.4f} m")
print("GRASP SUCCESS" if z_end - z_start > 0.02 else "GRASP FAILED")
print("=" * 78)

out = os.path.join(OUT_DIR, "ik_grasp.mp4")
w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, RESOLUTION)
for f in frames:
    w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
w.release()
print("saved:", out, f"({len(frames)} frames)")

app.close()