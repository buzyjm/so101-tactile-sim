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
FRAME = "gripper_frame_link"

BASE_POS = np.array([-0.20, 0.0, 0.75])
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
CUBE_POS = np.array([0.15, 0.0, 0.775])

solver = LulaKinematicsSolver(robot_description_path=DESCRIPTOR, urdf_path=URDF)
solver.set_robot_base_pose(BASE_POS, BASE_QUAT)


def rot_to_quat(R):
    """Rotation matrix to (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
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


def frame_from_z(z_dir, hint=np.array([0.0, 1.0, 0.0])):
    z = z_dir / np.linalg.norm(z_dir)
    x = np.cross(hint, z)
    if np.linalg.norm(x) < 1e-6:
        hint = np.array([1.0, 0.0, 0.0])
        x = np.cross(hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


print("=" * 84)
print("APPROACH ANGLE FEASIBILITY AT CUBE POSITION + 2cm")
print("tilt = 0 means tool Z points horizontally (+x), 90 means straight down")
print("=" * 84)
print(f"{'tilt':>5} | {'ok':>5} | {'joints (deg)':>44} | {'roll':>7}")
print("=" * 84)

target = CUBE_POS + np.array([0.0, 0.0, 0.02])

for tilt_deg in range(0, 91, 15):
    t = np.radians(tilt_deg)
    z_dir = np.array([np.cos(t), 0.0, -np.sin(t)])
    for roll_deg in [0, 90]:
        R = frame_from_z(z_dir)
        c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        R_roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R_final = R @ R_roll
        q, ok = solver.compute_inverse_kinematics(
            frame_name=FRAME,
            target_position=target,
            target_orientation=rot_to_quat(R_final),
            position_tolerance=0.005,
            orientation_tolerance=0.3,
        )
        joints = np.round(np.degrees(q), 1) if ok else None
        print(f"{tilt_deg:5.0f} | {str(ok):>5} | {str(joints):>44} | {roll_deg:7.0f}")

print("=" * 84)
app.close()