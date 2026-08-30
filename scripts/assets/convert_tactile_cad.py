"""Convert the tactile gripper STEP assemblies to self-contained USD assets.

Run inside the existing Isaac Sim conda environment::

    conda run -n isaacsim python scripts/assets/convert_tactile_cad.py

The source STEP files are never modified.  The HOOPS CAD converter is asked
for a monolithic, non-instanced USD so the resulting assembly can be inspected
and selectively referenced by the scene builder without hidden dependencies.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from isaacsim import SimulationApp


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "tactile" / "usd"
DEFAULT_INPUTS = (
    PROJECT_ROOT / "assets" / "tactile" / "Wrist_Roll_Follower_SO101_v2.step",
    PROJECT_ROOT / "assets" / "tactile" / "Moving_Jaw_SO101_v4_split.step",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help="STEP files to convert (defaults to both tactile gripper assemblies)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for converted USD assets",
    )
    parser.add_argument(
        "--lod",
        type=int,
        choices=range(5),
        default=4,
        metavar="0..4",
        help="HOOPS tessellation LOD; 4 is the high-fidelity baseline",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing derived USD output",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index used to initialize Kit (conversion itself is CPU-side)",
    )
    return parser.parse_args()


def validate_output(path: Path) -> dict[str, object]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Converted USD cannot be opened: {path}")

    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    points = sum(len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []) for prim in meshes)
    faces = sum(
        len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or [])
        for prim in meshes
    )
    return {
        "default_prim": str(stage.GetDefaultPrim().GetPath())
        if stage.GetDefaultPrim()
        else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "mesh_count": len(meshes),
        "point_count": points,
        "face_count": faces,
    }


def main() -> int:
    args = parse_args()
    inputs = [path.expanduser().resolve() for path in args.inputs]
    output_dir = args.output_dir.expanduser().resolve()

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".step", ".stp"}:
            raise ValueError(f"Expected STEP input, got: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{path.stem}.usd" for path in inputs]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Derived USD already exists; pass --force to replace it:\n  " + joined
        )

    print("inputs:", [str(path) for path in inputs], flush=True)
    print("output directory:", output_dir, flush=True)

    app = SimulationApp(
        {
            "headless": True,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "multi_gpu": False,
        }
    )
    exit_code = 0
    try:
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        for extension_name in (
            "omni.kit.converter.common",
            "omni.kit.converter.hoops_core",
        ):
            enabled = extension_manager.set_extension_enabled_immediate(
                extension_name, True
            )
            print(f"enable {extension_name}: {enabled}", flush=True)
        for _ in range(10):
            app.update()

        import omni.converter.hoops
        from omni.kit.converter.hoops_core import HoopsOptions, get_instance

        if get_instance() is None:
            raise RuntimeError("omni.kit.converter.hoops_core did not start")

        options = HoopsOptions()
        options.set_instancing_style_from_index(0)
        options.set_composition_style_from_index(0)
        options.filterStyle = omni.converter.hoops.FilterStyle.eOmit
        options.tessLOD = args.lod
        options.accurateSurfaceCurvatures = True
        options.accurateTessellation = False
        options.useMaterials = True
        options.useNormals = True
        options.convertMetadata = True
        options.convertCurves = False

        converter = omni.converter.hoops.Converter(options)
        for input_path, output_path in zip(inputs, outputs, strict=True):
            print(f"converting: {input_path}", flush=True)
            print(f"        to: {output_path}", flush=True)
            error_code, error_message = converter.convert(
                str(input_path), str(output_path), {}
            )
            if error_code != 0:
                raise RuntimeError(
                    f"HOOPS conversion failed ({error_code}): {error_message}"
                )
            if not output_path.is_file():
                raise RuntimeError(f"Converter reported success but wrote no file: {output_path}")
            print("validated:", validate_output(output_path), flush=True)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
