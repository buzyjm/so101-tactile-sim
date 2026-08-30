import inspect

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

import isaacsim.robot_motion.motion_generation as mg

print("=" * 78)
print("motion_generation exports:")
print([x for x in dir(mg) if not x.startswith("_")])
print("=" * 78)

try:
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    print("LulaKinematicsSolver __init__:")
    print(inspect.signature(LulaKinematicsSolver.__init__))
    print()
    print("methods:", [m for m in dir(LulaKinematicsSolver) if not m.startswith("_")])
    print()
    print("compute_inverse_kinematics:")
    print(inspect.signature(LulaKinematicsSolver.compute_inverse_kinematics))
    print(LulaKinematicsSolver.compute_inverse_kinematics.__doc__)
except Exception as e:
    print("LulaKinematicsSolver import failed:", e)

print("=" * 78)
try:
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
    print("ArticulationKinematicsSolver __init__:")
    print(inspect.signature(ArticulationKinematicsSolver.__init__))
    print("methods:", [m for m in dir(ArticulationKinematicsSolver) if not m.startswith("_")])
except Exception as e:
    print("ArticulationKinematicsSolver import failed:", e)

print("=" * 78)
app.close()