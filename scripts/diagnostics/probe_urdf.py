import inspect

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})

from isaacsim.asset.importer.urdf import URDFImporterConfig, impl

print("=" * 78)
src = inspect.getsource(URDFImporterConfig)
print(src)
print("=" * 78)
print("impl submodules:", [x for x in dir(impl) if not x.startswith("_")])
print("=" * 78)

app.close()