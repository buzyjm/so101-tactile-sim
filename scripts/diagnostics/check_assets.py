from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

from isaacsim.storage.native import get_assets_root_path

root = get_assets_root_path()
print("=" * 60)
print("ASSETS ROOT:", root)
print("=" * 60)

app.close()