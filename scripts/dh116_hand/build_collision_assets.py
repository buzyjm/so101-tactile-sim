"""Build separate self-collision-enabled assets, preserving the original baselines.

Uses USD core only; run in the isaacsim Python environment. Visual meshes and
materials are copied unchanged. Collider offsets are authored in the shared
instances layer, since instance proxies cannot be edited by runtime overrides.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from pxr import Sdf, Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]


def build(robot, profile):
    asset = 'dh116_hand' if robot == 'hand' else 'xarm5_dh116'
    if profile == 'manual_2026_04':
        source_name = 'usd_manual_2026_04'
        target_name = 'usd_manual_2026_04_self_collision'
    else:
        source_name = 'usd'
        target_name = 'usd_self_collision'
    source = ROOT / 'assets' / asset / source_name / asset
    target = ROOT / 'assets' / asset / target_name / asset
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite an existing asset: {target}')
    shutil.copytree(source, target)
    instances = target / 'payloads' / 'instances.usda'
    stage = Usd.Stage.Open(str(instances))
    changed = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.AddAppliedSchema('PhysxCollisionAPI')
            prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
            prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
            changed.append(str(prim.GetPath()))
    if not changed:
        raise RuntimeError('No colliders were found in the shared instances layer')
    stage.GetRootLayer().Save()
    # The arm-mounted hand needs more positional iterations than the fixed-base
    # hand to reject the thumb/index transient under the added wrist dynamics.
    position_iterations = 32 if robot == 'hand' else 64
    root = target / f'{asset}.usda'
    stage = Usd.Stage.Open(str(root))
    roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.AddAppliedSchema('PhysxArticulationAPI')
            prim.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute('physxArticulation:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(position_iterations)
            prim.CreateAttribute('physxArticulation:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(8)
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.AddAppliedSchema('PhysxRigidBodyAPI')
            prim.CreateAttribute('physxRigidBody:maxDepenetrationVelocity', Sdf.ValueTypeNames.Float).Set(1.)
    stage.GetRootLayer().Save()
    manifest = {'source': str(source.relative_to(ROOT)), 'asset': str(root.relative_to(ROOT)),
                'colliders': changed, 'roots': roots, 'filtered_pairs': [],
                'parameters': {'contact_offset_m':.001, 'rest_offset_m':0.,
                               'position_iterations':position_iterations,'velocity_iterations':8,'max_depenetration_velocity_m_s':1.},
                'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(source.rglob('*')) if p.is_file()}}
    (target / 'collision_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(root, f'{len(changed)} collider meshes, {len(roots)} articulations; no extra collision filters')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['hand','arm'], required=True)
    p.add_argument('--profile', choices=['manual_2026_04','legacy'], default='manual_2026_04')
    a = p.parse_args()
    build(a.robot, a.profile)
