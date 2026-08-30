from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import omni.usd
from isaacsim.core.api import World
from pxr import Usd, UsdGeom, UsdPhysics

USD_PATH = str(PROJECT_ROOT / "lab_scene.usda")
JOINT_ROOT = "/World/Robot/Physics"
ROBOT_PRIM = "/World/Robot/Geometry"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
         "wrist_flex", "wrist_roll", "gripper"]

omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

world = World(stage_units_in_meters=1.0)
world.reset()

drives = {n: UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{JOINT_ROOT}/{n}"), "angular")
          for n in NAMES}

art = None
try:
    from isaacsim.core.prims import Articulation
    art = Articulation(ROBOT_PRIM)
    art.initialize()
    print("articulation dof names:", art.dof_names)
except Exception as e:
    print("Articulation unavailable:", e)


def read_angles():
    if art is not None:
        try:
            return {n: float(v) for n, v in zip(art.dof_names, art.get_joint_positions()[0])}
        except Exception:
            pass
    out = {}
    for n in NAMES:
        prim = stage.GetPrimAtPath(f"{JOINT_ROOT}/{n}")
        attr = prim.GetAttribute("state:angular:physics:position")
        out[n] = attr.Get() if attr else None
    return out


TARGET = dict(shoulder_pan=0, shoulder_lift=-50, elbow_flex=70,
              wrist_flex=-25, wrist_roll=0, gripper=60)

print("\nsettling at zero pose")
for _ in range(60):
    world.step(render=False)
print(" ", read_angles())

print("\ncommanding target:", TARGET)
for n, v in TARGET.items():
    drives[n].CreateTargetPositionAttr().Set(float(v))

for i in range(300):
    world.step(render=False)
    if i % 60 == 59:
        a = read_angles()
        line = "  ".join(f"{n}={a[n]:.1f}" if a[n] is not None else f"{n}=?"
                         for n in NAMES)
        print(f"step {i+1:3d}  {line}")

print("\ntarget vs actual:")
a = read_angles()
for n in NAMES:
    t = TARGET[n]
    v = a[n]
    if v is None:
        print(f"  {n:<16} target={t:6.1f}  actual=?")
    else:
        print(f"  {n:<16} target={t:6.1f}  actual={v:7.2f}  error={t - v:7.2f}")

app.close()