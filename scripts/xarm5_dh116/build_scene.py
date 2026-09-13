"""Build and render the xArm5 + DH116 showcase scene.

    conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/build_scene.py --preview
    conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/build_scene.py \
        --output renders/xarm5_dh116_lab.png            # 4K RTX final

``--env lab`` (default) puts the robot on the table of Isaac Sim's
Simple_Room with a single prop; ``--env warehouse`` is the busier
Simple_Warehouse + packing-table variant.  The merged xArm5/DH116
articulation comes from the current manual-calibrated self-collision asset. The arm is posed under
physics (shoulder/elbow searched so the flange hovers at HOVER_TARGET), the
stage is saved to xarm5_dh116_<env>.usda, and the requested views are
rendered.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--env", choices=["lab", "warehouse"], default="lab")
ap.add_argument("--preview", action="store_true", help="fast 1600x900 ray-traced previews")
ap.add_argument("--renderer", choices=["rt", "pt"], default="rt", help="final renderer: RTX real-time (sharper) or path tracing")
ap.add_argument("--views", default="main", help="comma-separated view names (see VIEWS per env)")
ap.add_argument("--output", default=None, help="PNG for the 'main' view (default renders/xarm5_dh116_<env>.png)")
ap.add_argument("--width", type=int, default=3840)
ap.add_argument("--height", type=int, default=2160)
ap.add_argument("--spp", type=int, default=512, help="path-tracing samples per pixel (final only)")
ap.add_argument("--gpu", type=int, default=0)
args = ap.parse_args()
PT = (not args.preview) and args.renderer == "pt"

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({
    "headless": True, "active_gpu": args.gpu, "physics_gpu": args.gpu,
    "renderer": "PathTracing" if PT else "RaytracedLighting",
    "samples_per_pixel_per_frame": 1,
    "width": args.width, "height": args.height,
})

import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import UsdPhysics, Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = (ROOT / "assets" / "xarm5_dh116" / "usd_manual_2026_04_self_collision"
             / "xarm5_dh116" / "xarm5_dh116.usda")
ASSETS = get_assets_root_path()
PT_STEPS = 8
YCB = "/Isaac/Props/YCB/Axis_Aligned/"

HAND_POSE = {"hand_finger11": 0.45,
             "hand_finger21": 0.35, "hand_finger22": 0.55, "hand_finger31": 0.30, "hand_finger32": 0.50,
             "hand_finger41": 0.35, "hand_finger42": 0.55, "hand_finger51": 0.40, "hand_finger52": 0.60,
             "hand_finger12": 0.20, "hand_finger13": 0.30}

ENVS = {
    "lab": dict(
        env="/Isaac/Environments/Simple_Room/simple_room.usd",
        table=None, table_prim_name="table_low",      # use the room's own table (top z = 0.01)
        robot_xy=(-0.42, 0.0), robot_yaw=0.0,           # reach along +x
        hover=(0.55, 0.34),
        props=[("cracker", YCB + "003_cracker_box.usd", (0.16, 0.02), 25)],
        klt=None,
        lights=[("Key", (1.1, -1.5, 1.7), 40000.0, 0.15), ("Rim", (-1.4, 1.3, 1.6), 25000.0, 0.25)],
        views={  # name: (eye, target, focal mm)
            "main": ((1.65, -1.75, 1.02), (-0.02, 0.02, 0.30), 40.0),
            "main2": ((1.9, -1.5, 0.8), (-0.05, 0.0, 0.38), 45.0),
            "front": ((0.25, -2.4, 0.72), (-0.1, 0.0, 0.42), 50.0),
            "high": ((1.6, -1.55, 1.15), (-0.1, 0.0, 0.40), 40.0),
            "wrist": (None, None, 60.0),
        },
    ),
    "warehouse": dict(
        env="/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
        table=("/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd",
               (-3.2, -1.0, 0.0), 90.0, 0.01), table_prim_name=None,
        robot_xy=(-3.2, -1.62), robot_yaw=90.0,
        hover=(0.58, 0.36),
        props=[("cracker", YCB + "003_cracker_box.usd", (-3.08, -0.98), 25),
               ("mustard", YCB + "006_mustard_bottle.usd", (-3.38, -0.72), -30),
               ("soup", YCB + "005_tomato_soup_can.usd", (-3.02, -0.62), 0),
               ("drill", YCB + "035_power_drill.usd", (-3.28, -0.30), 70)],
        klt=((-3.22, 0.02), 105.0),
        lights=[("Key", (-2.0, -2.6, 2.9), 60000.0, 0.25), ("Fill", (-5.0, -2.4, 2.4), 20000.0, 0.4)],
        views={
            "main": ((-1.25, -3.35, 1.38), (-3.2, -0.9, 1.08), 40.0),
            "alt": ((-5.0, -3.3, 1.7), (-3.2, -1.0, 1.05), 35.0),
            "wrist": (None, None, 60.0),
        },
    ),
}
CFG = ENVS[args.env]


def add_ref(stage, path, asset, pos=(0, 0, 0), yaw=0.0, scale=1.0):
    p = stage.DefinePrim(path, "Xform")
    xf = UsdGeom.Xformable(p)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateZOp().Set(float(yaw))
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    child = stage.DefinePrim(path + "/asset", "Xform")
    child.GetReferences().AddReference(asset)
    return p


def world_bbox(prim):
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def set_material(stage, name, **inputs):
    """Override interface inputs on every imported material called `name`."""
    for p in stage.Traverse():
        if p.GetName() == name and p.IsA(UsdShade.Material):
            m = UsdShade.Material(p)
            for k, v in inputs.items():
                t = Sdf.ValueTypeNames.Color3f if isinstance(v, tuple) else Sdf.ValueTypeNames.Float
                m.CreateInput(k, t).Set(Gf.Vec3f(*v) if isinstance(v, tuple) else float(v))


def main() -> int:
    world = World(stage_units_in_meters=1.0, physics_dt=1/120)
    stage = world.stage
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            prim.AddAppliedSchema("PhysxSceneAPI")
            prim.CreateAttribute("physxScene:solveArticulationContactLast", Sdf.ValueTypeNames.Bool).Set(True)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    add_ref(stage, "/World/Env", ASSETS + CFG["env"])
    if CFG["table"]:
        asset, pos, yaw, scale = CFG["table"]
        table = add_ref(stage, "/World/Table", ASSETS + asset, pos, yaw, scale)
    for _ in range(3):
        app.update()
    if CFG["table"] is None:
        table = next(p for p in stage.Traverse() if p.GetName().startswith(CFG["table_prim_name"]))
    tmin, tmax = world_bbox(table)
    top = float(tmax[2])
    print(f"TABLE {table.GetPath()} bbox min={np.round(tmin,3)} max={np.round(tmax,3)} top={top:.3f}")

    rx, ry = CFG["robot_xy"]
    robot_pos = (rx, ry, top)
    add_ref(stage, "/World/Robot", str(ROBOT_USD), robot_pos, CFG["robot_yaw"])
    # YCB "Axis_Aligned" props are static visuals with a mid-body origin:
    # place them, then lift each so its bbox bottom rests on the table top.
    for name, asset, xy, yaw in CFG["props"]:
        add_ref(stage, f"/World/Props/{name}", ASSETS + asset, (xy[0], xy[1], top), yaw)
    if CFG["klt"]:
        (kx, ky), kyaw = CFG["klt"]
        add_ref(stage, "/World/Props/klt", ASSETS + "/Isaac/Props/KLT_Bin/small_KLT.usd", (kx, ky, top + 0.09), kyaw)
    for _ in range(3):
        app.update()
    for name, _, xy, _ in CFG["props"]:
        prim = stage.GetPrimAtPath(f"/World/Props/{name}")
        pmin, _ = world_bbox(prim)
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(xy[0], xy[1], top + (top - pmin[2]) + 0.001))

    for name, pos, intensity, radius in CFG["lights"]:
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/{name}")
        light.CreateIntensityAttr(intensity); light.CreateRadiusAttr(radius)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*pos))

    # materials: slightly off-white, tighter roughness so the arm's shape reads
    set_material(stage, "White", diffuseColor=(0.90, 0.90, 0.91), roughness=0.38)
    set_material(stage, "Silver", diffuseColor=(0.70, 0.70, 0.72), metallic=0.85, roughness=0.30)
    set_material(stage, "adapter_anodized", metallic=0.9, roughness=0.35)

    # pose robot under physics
    art = SingleArticulation("/World/Robot/asset/Geometry", name="robot")
    world.scene.add(art)
    world.reset()
    names = art.dof_names
    flange = next(x for x in stage.Traverse() if x.GetName() == "flange_adapter")

    def flange_pos():
        return np.array(UsdGeom.Xformable(flange).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())

    def set_pose(arm):
        q = np.zeros(len(names))
        for k, v in {**arm, **HAND_POSE}.items():
            q[names.index(k)] = v
        art.set_joint_positions(q)
        art.apply_action(ArticulationAction(joint_positions=q))
        return q

    # shoulder/elbow pair whose flange lands closest to hover target, hand
    # pointing straight down (joint2 + joint3 + joint4 = 0)
    reach_t, height_t = CFG["hover"]
    best = None
    for j2 in np.arange(0.15, 0.95, 0.05):
        for j3 in np.arange(-1.7, -0.5, 0.05):
            set_pose({"joint1": 0.0, "joint2": j2, "joint3": j3, "joint4": -(j2 + j3), "joint5": 0.0})
            world.step(render=False)
            f = flange_pos()
            reach = math.hypot(f[0] - robot_pos[0], f[1] - robot_pos[1]); height = f[2] - top
            err = math.hypot(reach - reach_t, height - height_t)
            if best is None or err < best[0]:
                best = (err, j2, j3, reach, height)
    err, j2, j3, reach, height = best
    print(f"POSE joint2={j2:.2f} joint3={j3:.2f} joint4={-(j2+j3):.2f} reach={reach:.3f} height={height:.3f} err={err:.3f}")
    q = set_pose({"joint1": 0.0, "joint2": j2, "joint3": j3, "joint4": -(j2 + j3), "joint5": 0.0})
    for _ in range(120):
        world.step(render=False)
    print("DRIFT(deg):", np.round(np.degrees(art.get_joint_positions() - q), 2).tolist())
    fpos = flange_pos()
    print("FLANGE world pos:", np.round(fpos, 3).tolist(), " table top:", round(top, 3))
    for name, *_ in CFG["props"] + ([("klt",)] if CFG["klt"] else []):
        pmin, pmax = world_bbox(stage.GetPrimAtPath(f"/World/Props/{name}"))
        on = tmin[0] <= pmin[0] and pmax[0] <= tmax[0] and tmin[1] <= pmin[1] and pmax[1] <= tmax[1] and pmin[2] > top - 0.02
        print(f"PROP {name:8s} z={pmin[2]:.3f}..{pmax[2]:.3f} xy=({pmin[0]:.2f},{pmin[1]:.2f})..({pmax[0]:.2f},{pmax[1]:.2f}) on_table={on}")
    world.pause()
    stage_out = ROOT / f"xarm5_dh116_{args.env}.usda"
    stage.GetRootLayer().Export(str(stage_out))
    print("STAGE saved:", stage_out)

    # cameras + render
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", 1)
    if PT:
        settings.set("/rtx/pathtracing/spp", max(1, args.spp // PT_STEPS))
        settings.set("/rtx/pathtracing/totalSpp", args.spp)
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    cam.CreateFocusDistanceAttr(2.0); cam.CreateFStopAttr(0.0)
    xf_op = UsdGeom.Xformable(cam).AddTransformOp()
    res = (1600, 900) if args.preview else (args.width, args.height)
    rp = rep.create.render_product("/World/Cam", res)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach([rp])
    out = Path(args.output) if args.output else ROOT / "renders" / f"xarm5_dh116_{args.env}.png"
    for name in args.views.split(","):
        eye, tgt, focal = CFG["views"][name]
        if name == "wrist":
            eye, tgt = fpos + np.array([0.42, -0.42, 0.22]), fpos - np.array([0, 0, 0.03])
        cam.GetFocalLengthAttr().Set(focal)
        xf_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
        steps, sub = (6, 8) if args.preview else ((PT_STEPS, 1) if PT else (12, 16))
        px = None
        for i in range(steps + 20):
            rep.orchestrator.step(rt_subframes=sub, pause_timeline=True)
            px = rgb.get_data()
            if i >= steps - 1 and getattr(px, "ndim", 0) == 3:
                break
        suffix = "" if (name == "main" and not args.preview) else f"_{name}{'_preview' if args.preview else ''}"
        if not args.preview and args.renderer == "pt":
            suffix += "_pt"
        path = out.with_name(f"{out.stem}{suffix}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(px[:, :, :3]).save(path)
        print("SAVED", path)
    rgb.detach([rp]); rp.destroy()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
