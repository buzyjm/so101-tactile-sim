"""Regression checks on the composed USD assets, including instance proxies."""
import hashlib
import json
from pathlib import Path
import unittest
from pxr import Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]


class CollisionAssetTests(unittest.TestCase):
    def test_instanced_colliders_have_effective_offsets(self):
        # Runtime recursive overrides used to skip these proxies entirely.
        for asset, expected in [('dh116_hand',17),('xarm5_dh116',24)]:
            with self.subTest(asset=asset):
                folder = ROOT/'assets'/asset/'usd_manual_2026_04_self_collision'/asset
                stage = Usd.Stage.Open(str(folder/f'{asset}.usda'))
                meshes = [p for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
                          if p.HasAPI(UsdPhysics.CollisionAPI)]
                self.assertEqual(len(meshes), expected)
                for p in meshes:
                    self.assertAlmostEqual(p.GetAttribute('physxCollision:contactOffset').Get(), .001, places=7)
                    self.assertEqual(p.GetAttribute('physxCollision:restOffset').Get(), 0.)
                roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
                self.assertEqual(len(roots),1)
                self.assertTrue(roots[0].GetAttribute('physxArticulation:enabledSelfCollisions').Get())
                expected_iterations = 32 if asset == 'dh116_hand' else 64
                self.assertEqual(
                    roots[0].GetAttribute('physxArticulation:solverPositionIterationCount').Get(),
                    expected_iterations,
                )

    def test_visuals_and_legacy_assets_preserved(self):
        for asset in ['dh116_hand','xarm5_dh116']:
            with self.subTest(asset=asset):
                folder = ROOT/'assets'/asset/'usd_manual_2026_04_self_collision'/asset
                manifest=json.loads((folder/'collision_manifest.json').read_text())
                source=ROOT/manifest['source']
                for relative, digest in manifest['source_sha256'].items():
                    self.assertEqual(hashlib.sha256((source/relative).read_bytes()).hexdigest(),digest)
                for relative in ('payloads/geometries.usd','payloads/materials.usda'):
                    self.assertEqual((source/relative).read_bytes(),(folder/relative).read_bytes())


if __name__=='__main__':
    unittest.main()
