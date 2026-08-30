import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import numpy as np

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

DESCRIPTOR = str(PROJECT_ROOT / "so101_descriptor.yaml")
URDF = os.environ.get(
    "SO101_URDF_PATH",
    str(PROJECT_ROOT.parent / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
)

ROBOT_BASE_POS = np.array([-0.20, 0.0, 0.75])
ROBOT_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

CUBE_POS = np.array([0.15, 0.0, 0.775])

solver = LulaKinematicsSolver(
    robot_description_path=DESCRIPTOR,
    urdf_path=URDF,
)

print("=" * 78)
print("joint names:", solver.get_joint_names())
print()
print("available frames:")
for f in solver.get_all_frame_names():
    print("   ", f)
print()
print("cspace position limits:", solver.get_cspace_position_limits())
print("default position tolerance:", solver.get_default_position_tolerance())
print("default orientation tolerance:", solver.get_default_orientation_tolerance())
print("=" * 78)

solver.set_robot_base_pose(ROBOT_BASE_POS, ROBOT_BASE_QUAT)

FRAME = "gripper_frame_link"

targets = {
    "above_cube_10cm": CUBE_POS + np.array([0.0, 0.0, 0.10]),
    "above_cube_5cm": CUBE_POS + np.array([0.0, 0.0, 0.05]),
    "at_cube": CUBE_POS.copy(),
}

for label, pos in targets.items():
    # Position only. SO-101 has 5 DoF, so a full 6 DoF pose is over constrained.
    q, ok = solver.compute_inverse_kinematics(
        frame_name=FRAME,
        target_position=pos,
        target_orientation=None,
        position_tolerance=0.005,
    )
    print(f"\n[{label}] target = {pos}")
    print(f"  success: {ok}")
    if ok:
        deg = np.degrees(q)
        for name, r, d in zip(solver.get_joint_names(), q, deg):
            print(f"    {name:<16} {r:8.4f} rad   {d:8.2f} deg")
        fk_pos, fk_rot = solver.compute_forward_kinematics(FRAME, q)
        err = np.linalg.norm(np.asarray(fk_pos) - pos)
        print(f"  FK check: {np.round(fk_pos, 4)}  error = {err:.5f} m")

print("=" * 78)
app.close()