"""Validate and export the 52 measured taxels against the sensor CAD."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene_config import (
    TACTILE_SENSOR_CAD_MAIN_MESH_RELATIVE_PATH,
    TACTILE_SENSOR_CAD_PRIM_PATH,
    TACTILE_SENSOR_CAD_SECONDARY_MESH_RELATIVE_PATH,
    TACTILE_SENSOR_CAD_USD_PATH,
)
from tactile_taxels import load_taxel_positions_mm


CSV_PATH = PROJECT_ROOT / "assets" / "tactile" / "taxel_coordinates.csv"
REPORT_PATH = (
    PROJECT_ROOT / "tactile_logs" / "taxel_geometry_validation.json"
)
SURFACE_TOLERANCE_MM = 0.01


def triangulated_mesh(stage, path):
    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    vertices = np.asarray(
        [transform.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()],
        dtype=np.float64,
    )
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    triangles = []
    cursor = 0
    for count in counts:
        face = [int(index) for index in indices[cursor : cursor + count]]
        cursor += count
        for offset in range(1, count - 1):
            triangles.append((face[0], face[offset], face[offset + 1]))
    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)


def main():
    taxels = np.asarray(load_taxel_positions_mm(), dtype=np.float64)
    stage = Usd.Stage.Open(str(TACTILE_SENSOR_CAD_USD_PATH))
    component_path = (
        TACTILE_SENSOR_CAD_PRIM_PATH + "/tn__20151000_27_1_1_1_bP5"
    )
    meshes = [
        triangulated_mesh(
            stage,
            TACTILE_SENSOR_CAD_PRIM_PATH
            + "/"
            + TACTILE_SENSOR_CAD_MAIN_MESH_RELATIVE_PATH,
        ),
        triangulated_mesh(
            stage,
            TACTILE_SENSOR_CAD_PRIM_PATH
            + "/"
            + TACTILE_SENSOR_CAD_SECONDARY_MESH_RELATIVE_PATH,
        ),
    ]
    cad = trimesh.util.concatenate(meshes)
    _, distances, _ = trimesh.proximity.closest_point_naive(cad, taxels)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("taxel_id", "x_mm", "y_mm", "z_mm", "cad_distance_mm"))
        for taxel_id, (position, distance) in enumerate(
            zip(taxels, distances), start=1
        ):
            writer.writerow((taxel_id, *position, float(distance)))

    report = {
        "passed": bool(np.max(distances) <= SURFACE_TOLERANCE_MM),
        "coordinate_source": str(
            PROJECT_ROOT / "assets" / "tactile" / "array.xlsx"
        ),
        "cad_source": str(TACTILE_SENSOR_CAD_USD_PATH),
        "sensor_count": 2,
        "mapping": {
            "sensor1": "fixed/gripper-side",
            "sensor2": "moving/wrist-roll-side",
            "local_coordinates_shared_by_both_sensors": True,
        },
        "taxel_count_per_sensor": int(len(taxels)),
        "units": "millimetres in native sensor/CAD frame",
        "coordinate_min_mm": np.min(taxels, axis=0).tolist(),
        "coordinate_max_mm": np.max(taxels, axis=0).tolist(),
        "cad_surface_tolerance_mm": SURFACE_TOLERANCE_MM,
        "cad_surface_distance_mm": {
            "min": float(np.min(distances)),
            "median": float(np.median(distances)),
            "max": float(np.max(distances)),
            "rms": float(np.sqrt(np.mean(np.square(distances)))),
            "max_distance_taxel_id": int(np.argmax(distances) + 1),
        },
        "coordinate_csv": str(CSV_PATH),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
