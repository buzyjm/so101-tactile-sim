from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from pxr import UsdPhysics

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
JOINT_ROOT = "/World/Robot/Physics"
ROBOT_PRIM = "/World/Robot/Geometry"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
         "wrist_flex", "wrist_roll", "gripper"]

TARGET = dict(shoulder_pan=0.0, shoulder_lift=-50.0, elbow_flex=70.0,
              wrist_flex=-25.0, wrist_roll=0.0, gripper=60.0)

# (stiffness, damping, maxForce) triples spanning several orders of magnitude
COMBOS = [
    (0.31067, 0.010472, 3.35),
    (3.1067, 0.10472, 3.35),
    (17.8, 0.6, 3.35),
    (17.8, 0.6, 50.0),
    (100.0, 5.0, 50.0),
    (1000.0, 50.0, 200.0),
]

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

drives = {n: UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{JOINT_ROOT}/{n}"), "angular")
          for n in NAMES}

world = World(stage_units_in_meters=1.0)
world.reset()

art = Articulation(ROBOT_PRIM)
art.initialize()

print("=" * 84)
print(f"{'stiffness':>10} {'damping':>9} {'maxForce':>9} | "
      f"{'sh_lift':>9} {'elbow':>9} {'wr_flex':>9} {'gripper':>9}")
print("=" * 84)

for stiff, damp, force in COMBOS:
    for n in NAMES:
        d = drives[n]
        d.CreateStiffnessAttr().Set(stiff)
        d.CreateDampingAttr().Set(damp)
        d.CreateMaxForceAttr().Set(force)
        d.CreateTargetPositionAttr().Set(0.0)

    world.reset()
    for _ in range(60):
        world.step(render=False)

    for n, v in TARGET.items():
        drives[n].CreateTargetPositionAttr().Set(v)
    for _ in range(300):
        world.step(render=False)

    pos = {n: float(v) for n, v in zip(art.dof_names, art.get_joint_positions()[0])}
    print(f"{stiff:10.4f} {damp:9.4f} {force:9.1f} | "
          f"{pos['shoulder_lift']:9.2f} {pos['elbow_flex']:9.2f} "
          f"{pos['wrist_flex']:9.2f} {pos['gripper']:9.2f}")

print("=" * 84)
print("targets:", {k: v for k, v in TARGET.items() if v != 0})
print("=" * 84)

app.close()