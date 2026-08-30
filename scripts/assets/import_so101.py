from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

URDF_PATH = os.environ.get(
    "SO101_URDF_PATH",
    str(PROJECT_ROOT.parent / "SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
)
OUT_DIR = str(PROJECT_ROOT / "assets")

os.makedirs(OUT_DIR, exist_ok=True)

cfg = URDFImporterConfig(
    urdf_path=URDF_PATH,
    usd_path=OUT_DIR,
    fix_base=True,
    merge_fixed_joints=False,
    merge_mesh=False,
    # The gripper jaw is a C shape; convex hull would seal the opening
    collision_type="Convex Decomposition",
    collision_from_visuals=False,
    allow_self_collision=False,
    joint_drive_type="position",
    # Measured STS3215 values from joints_properties.xml (MuJoCo units).
    # Verify against Isaac unit convention with a step response test.
    override_joint_stiffness=17.8,
    override_joint_damping=0.6,
    debug_mode=False,
)

importer = URDFImporter(config=cfg)
result = importer.import_urdf()

print("=" * 60)
print("result:", result)
print("=" * 60)

app.close()