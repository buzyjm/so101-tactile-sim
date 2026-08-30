"""Exercise both fingertip contact sensors without using a grasp policy."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_GPU = int(os.environ.get("SO101_TACTILE_TEST_GPU", "0"))
# Dynamic bodies generate the same solver contacts as real grasped objects.
# Keep the old kinematic probes available for diagnostics, but do not use them
# for the release gate: pose-driven kinematic overlaps can be visible to scene
# queries without producing contact reports on articulation links.
DYNAMIC_PROBES = os.environ.get("SO101_TACTILE_DYNAMIC_PROBES", "1") == "1"
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": TEST_GPU,
        "physics_gpu": TEST_GPU,
        "multi_gpu": False,
    }
)

import omni.usd
import carb
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
from omni.physx.scripts import physicsUtils
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.sensors.experimental.physics import ContactSensor
from omni.physx import get_physx_scene_query_interface
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)

from scene_config import (
    FIXED_STOCK_COLLISION_PRIM_PATH,
    MOVING_STOCK_COLLISION_PRIM_PATH,
    PROJECT_ROOT,
    TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME,
    TACTILE_MIN_FORCE_N,
    TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    TACTILE_ROOT_PRIM_PATHS,
)
from tactile_sensor import DPS2015EliteMvp


USD_PATH = os.environ.get(
    "SO101_SCENE_USD", str(PROJECT_ROOT / "lab_scene.usda")
)
RESULT_PATH = PROJECT_ROOT / "tactile_logs" / "contact_mvp_result.json"
PROBE_RADIUS = float(
    os.environ.get("SO101_TACTILE_PROBE_RADIUS_M", "0.004")
)
APPROACH_CLEARANCE = 0.0015
FINAL_PENETRATION = float(
    os.environ.get("SO101_TACTILE_PROBE_PENETRATION_M", "0.002")
)
PROBE_INWARD_SPEED = float(
    os.environ.get("SO101_TACTILE_PROBE_SPEED_M_S", "0.75")
)
SETTLE_STEPS = 120
APPROACH_STEPS = 120
HOLD_STEPS = 120
RELEASE_STEPS = 60


def create_kinematic_probe(stage, path, color):
    root = UsdGeom.Xform.Define(stage, path)
    translate = root.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, -10.0))

    sphere = UsdGeom.Sphere.Define(stage, path + "/Collision")
    sphere.CreateRadiusAttr(PROBE_RADIUS)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    material_path = "/World/Looks/TactileRegressionProbeMaterial"
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        UsdShade.Material.Define(stage, material_path)
        material_prim = stage.GetPrimAtPath(material_path)
        UsdPhysics.MaterialAPI.Apply(material_prim)
        material = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        material.CreateCompliantContactStiffnessAttr().Set(1500.0)
        material.CreateCompliantContactDampingAttr().Set(5.0)
    physicsUtils.add_physics_material_to_prim(
        stage, sphere.GetPrim(), material_path
    )

    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateKinematicEnabledAttr(not DYNAMIC_PROBES)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.001)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    # The regression controls probe poses explicitly. Gravity would otherwise
    # pull a dynamic probe away from the moving fingertip between updates and
    # make the result depend on the sensor's world-space orientation.
    physx_body.CreateDisableGravityAttr().Set(DYNAMIC_PROBES)
    PhysxSchema.PhysxContactReportAPI.Apply(
        root.GetPrim()
    ).CreateThresholdAttr().Set(0.0)
    sensor = IsaacSensorSchema.IsaacContactSensor.Define(
        stage, path + "/ContactSensor"
    )
    sensor.CreateThresholdAttr().Set((0.0, 1.0e6))
    sensor.CreateRadiusAttr().Set(0.05)
    return path


def sensor_offset_world(stage, sensor_path, outward_distance):
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(sensor_path)
    ).ComputeLocalToWorldTransform(0)
    apex = Gf.Vec3d(*TACTILE_ACTIVE_SURFACE_APEX_SENSOR_FRAME)
    outward_sign = 1.0 if apex[2] >= 0.0 else -1.0
    return transform.Transform(
        apex + Gf.Vec3d(0.0, 0.0, outward_sign * outward_distance)
    )


def overlap_hits(position, radius):
    hits = []

    def report_hit(hit):
        hits.append(
            {
                "rigid_body": str(hit.rigid_body),
                "collision": str(hit.collision),
            }
        )
        return True

    get_physx_scene_query_interface().overlap_sphere(
        radius,
        carb.Float3(*position),
        report_hit,
        False,
    )
    return hits



def capture_if_due(model, world, samples):
    sample = model.maybe_sample(
        world.current_time,
        world.current_time_step_index,
    )
    if sample is not None:
        samples.append(sample)


omni.usd.get_context().open_stage(USD_PATH)
stage = omni.usd.get_context().get_stage()

if os.environ.get("SO101_DISABLE_STOCK_COLLISIONS", "0") == "1":
    for stock_collision_path in (
        FIXED_STOCK_COLLISION_PRIM_PATH,
        MOVING_STOCK_COLLISION_PRIM_PATH,
    ):
        stage.GetPrimAtPath(stock_collision_path).SetActive(False)

missing = [
    path for path in TACTILE_ROOT_PRIM_PATHS
    if not stage.GetPrimAtPath(path).IsValid()
]
if missing:
    app.close()
    raise RuntimeError(
        "Tactile prims are missing. Rebuild the scene first with: "
        "conda run -n isaacsim python build_scene.py. Missing: "
        + ", ".join(missing)
    )

probe_paths = (
    create_kinematic_probe(
        stage,
        "/World/TactileMvpProbeFixed",
        (0.1, 0.8, 1.0),
    ),
    create_kinematic_probe(
        stage,
        "/World/TactileMvpProbeMoving",
        (1.0, 0.6, 0.1),
    ),
)
probes = RigidPrim(list(probe_paths))
probe_sensors = [ContactSensor(path + "/ContactSensor") for path in probe_paths]

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    rendering_dt=1.0 / 60.0,
)
model = DPS2015EliteMvp(stage)
world.reset()
model.reset(world.current_time)


def move_single_probe(sensor_index, outward_distance):
    positions = np.full((2, 3), -10.0, dtype=np.float32)
    positions[sensor_index] = sensor_offset_world(
        stage,
        TACTILE_ROOT_PRIM_PATHS[sensor_index],
        outward_distance,
    )
    probes.set_world_poses(positions=positions)
    if DYNAMIC_PROBES:
        apex = np.asarray(
            sensor_offset_world(
                stage, TACTILE_ROOT_PRIM_PATHS[sensor_index], 0.0
            ),
            dtype=np.float32,
        )
        outward = positions[sensor_index] - apex
        outward /= np.linalg.norm(outward)
        velocities = np.zeros((2, 3), dtype=np.float32)
        velocities[sensor_index] = -PROBE_INWARD_SPEED * outward
        probes.set_velocities(linear_velocities=velocities)
    return positions[sensor_index].copy()


samples = []
phase_ranges = {}

phase_ranges["baseline"] = [len(samples), None]
for _ in range(SETTLE_STEPS):
    world.step(render=False)
    capture_if_due(model, world, samples)
phase_ranges["baseline"][1] = len(samples)

active_phase_names = []
hold_target_positions = np.full((2, 3), np.nan, dtype=np.float64)
hold_actual_positions = np.full((2, 3), np.nan, dtype=np.float64)
hold_sensor_apex_positions = np.full((2, 3), np.nan, dtype=np.float64)
hold_overlap_hits = [[], []]
hold_probe_sensor_readings = [{}, {}]
hold_robot_sensor_readings = [{}, {}]

for sensor_index, sensor_name in enumerate(("fixed", "moving")):
    approach_name = sensor_name + "_approach"
    active_phase_names.append(approach_name)
    phase_ranges[approach_name] = [len(samples), None]
    for step in range(APPROACH_STEPS):
        phase = (step + 1) / APPROACH_STEPS
        distance = (
            PROBE_RADIUS
            + APPROACH_CLEARANCE
            - phase * (APPROACH_CLEARANCE + FINAL_PENETRATION)
        )
        move_single_probe(sensor_index, distance)
        world.step(render=False)
        capture_if_due(model, world, samples)
    phase_ranges[approach_name][1] = len(samples)

    hold_name = sensor_name + "_hold"
    active_phase_names.append(hold_name)
    phase_ranges[hold_name] = [len(samples), None]
    distance = PROBE_RADIUS - FINAL_PENETRATION
    for _ in range(HOLD_STEPS):
        hold_target_positions[sensor_index] = move_single_probe(
            sensor_index, distance
        )
        world.step(render=False)
        capture_if_due(model, world, samples)
    actual_positions, _ = probes.get_world_poses()
    hold_actual_positions[sensor_index] = actual_positions.numpy()[sensor_index]
    hold_sensor_apex_positions[sensor_index] = sensor_offset_world(
        stage, TACTILE_ROOT_PRIM_PATHS[sensor_index], 0.0
    )
    hold_overlap_hits[sensor_index] = overlap_hits(
        hold_actual_positions[sensor_index], PROBE_RADIUS
    )
    reading = probe_sensors[sensor_index].get_sensor_reading()
    hold_probe_sensor_readings[sensor_index] = {
        "is_valid": bool(reading.is_valid),
        "in_contact": bool(reading.in_contact),
        "value": float(reading.value),
        "raw_contact_count": len(probe_sensors[sensor_index].get_raw_data()),
    }
    robot_sensor = model._sensors[sensor_index]
    robot_reading = robot_sensor.get_sensor_reading()
    robot_raw_contacts = robot_sensor.get_raw_data()
    hold_robot_sensor_readings[sensor_index] = {
        "is_valid": bool(robot_reading.is_valid),
        "in_contact": bool(robot_reading.in_contact),
        "value": float(robot_reading.value),
        "raw_contacts": [
            {
                "body0": str(
                    PhysicsSchemaTools.intToSdfPath(int(contact["body0"]))
                ),
                "body1": str(
                    PhysicsSchemaTools.intToSdfPath(int(contact["body1"]))
                ),
                "impulse": {
                    axis: float(contact["impulse"][axis])
                    for axis in ("x", "y", "z")
                },
                "dt": float(contact["dt"]),
            }
            for contact in robot_raw_contacts
        ],
    }
    phase_ranges[hold_name][1] = len(samples)

    probes.set_world_poses(
        positions=np.full((2, 3), -10.0, dtype=np.float32)
    )
    for _ in range(RELEASE_STEPS // 2):
        world.step(render=False)
        capture_if_due(model, world, samples)


phase_ranges["release"] = [len(samples), None]
probes.set_world_poses(
    positions=np.full((2, 3), -10.0, dtype=np.float32)
)
for _ in range(RELEASE_STEPS):
    world.step(render=False)
    capture_if_due(model, world, samples)
phase_ranges["release"][1] = len(samples)


def phase_samples(name):
    start, stop = phase_ranges[name]
    return samples[start:stop]


baseline = phase_samples("baseline")
active = []
for phase_name in active_phase_names:
    active.extend(phase_samples(phase_name))
released = phase_samples("release")


def max_abs_force(sample_group):
    if not sample_group:
        return np.zeros((2, 3), dtype=np.float32)
    return np.max(
        np.abs(np.stack([sample.total_forces for sample in sample_group])),
        axis=0,
    )


def max_abs_taxel_force(sample_group, field):
    if not sample_group:
        return np.zeros((2, 52, 3), dtype=np.float32)
    return np.max(
        np.abs(np.stack([getattr(sample, field) for sample in sample_group])),
        axis=0,
    )


baseline_max = max_abs_force(baseline)
active_max = max_abs_force(active)
active_taxel_max = max_abs_taxel_force(active, "taxel_forces")
active_raw_taxel_max = max_abs_taxel_force(active, "raw_taxel_forces")
release_last = (
    released[-1].total_forces
    if released
    else np.zeros((2, 3), dtype=np.float32)
)
detected = np.any(active_max >= TACTILE_MIN_FORCE_N, axis=1)
taxel_detected = np.any(
    active_taxel_max >= TACTILE_MIN_FORCE_N, axis=(1, 2)
)
raw_conservation_error = np.max(
    np.abs(
        np.stack(
            [
                sample.raw_total_forces
                - np.sum(sample.raw_taxel_forces, axis=1)
                for sample in samples
            ]
        )
    ),
    axis=(0, 2),
)
loaded_taxel_ids = [
    (
        np.flatnonzero(
            np.any(active_taxel_max[sensor_index] > 0.0, axis=1)
        )
        + 1
    ).tolist()
    for sensor_index in range(2)
]
passed = bool(
    np.all(detected)
    and np.all(taxel_detected)
    and np.all(raw_conservation_error <= 1.0e-5)
)

result = {
    "passed": passed,
    "physics_frequency_hz": TACTILE_MVP_PHYSICS_FREQUENCY_HZ,
    "output_frequency_hz": model.output_frequency_hz,
    "sample_count": len(samples),
    "probe_penetration_m": FINAL_PENETRATION,
    "probe_mode": "dynamic" if DYNAMIC_PROBES else "kinematic",
    "probe_inward_speed_m_s": PROBE_INWARD_SPEED if DYNAMIC_PROBES else 0.0,
    "hold_target_positions": np.asarray(hold_target_positions).tolist(),
    "hold_actual_positions": np.asarray(hold_actual_positions).tolist(),
    "hold_sensor_apex_positions": hold_sensor_apex_positions.tolist(),
    "hold_overlap_hits": hold_overlap_hits,
    "hold_probe_sensor_readings": hold_probe_sensor_readings,
    "hold_robot_sensor_readings": hold_robot_sensor_readings,
    "phase_sample_ranges": phase_ranges,
    "baseline_max_abs_force_n": baseline_max.tolist(),
    "active_max_abs_force_n": active_max.tolist(),
    "taxel_output_shape": [2, 52, 3],
    "active_taxel_peak_component_n": np.max(
        active_taxel_max, axis=(1, 2)
    ).tolist(),
    "active_raw_taxel_peak_component_n": np.max(
        active_raw_taxel_max, axis=(1, 2)
    ).tolist(),
    "active_loaded_taxel_ids": loaded_taxel_ids,
    "raw_force_conservation_max_error_n": (
        raw_conservation_error.tolist()
    ),
    "release_last_force_n": release_last.tolist(),
    "detected_both_sensors": detected.tolist(),
    "detected_both_taxel_arrays": taxel_detected.tolist(),
    "latest_sample": samples[-1].to_dict(include_taxels=False) if samples else None,
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
with RESULT_PATH.open("w", encoding="utf-8") as result_file:
    json.dump(result, result_file, indent=2)

print(json.dumps(result, indent=2))
print("saved:", RESULT_PATH)

model.close()
app.close()
if not passed:
    raise SystemExit(1)
