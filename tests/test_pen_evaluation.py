import unittest
import numpy as np

from dh116_hand_lab.evaluation import summarize_rollout


class EvaluationTests(unittest.TestCase):
    def rollout(self, rate=4.0, drop=None, vertical=False):
        dt, steps = 1 / 60, 360
        angle = np.arange(1, steps + 1) * dt * rate
        axes = np.stack([np.cos(angle), np.sin(angle), np.zeros(steps)], axis=-1)[:, None]
        initial = np.array([[1., 0., 0.]])
        if vertical:
            axes[:] = (0, 0, 1)
            initial[:] = (0, 0, 1)
        on_palm = np.ones((steps, 1), dtype=bool)
        if drop is not None:
            # Simulate exactly one terminal step followed by auto-reset and motion.
            on_palm[drop] = False
        return summarize_rollout(np.full((steps, 1), rate), axes, np.zeros((steps, 1, 3)),
                                 on_palm, initial, [0.], [0., 0., 0.], dt, 1., 1)

    def test_known_rotations(self):
        report = self.rollout()
        r = report["trials"][0]
        self.assertTrue(r["success"])
        self.assertAlmostEqual(r["projected_rate_rad_s"], 4.)
        self.assertAlmostEqual(r["projected_turns"], 24 / (2 * np.pi))
        self.assertEqual(r["startup_s"], 1.)
        self.assertEqual(report["summary"]["quarter_turn_rate"], 1.)
        self.assertEqual(report["summary"]["half_turn_rate"], 1.)

    def test_reset_cannot_erase_drop_or_add_credit(self):
        r = self.rollout(drop=0)["trials"][0]
        self.assertTrue(r["dropped"])
        self.assertFalse(r["success"])
        self.assertEqual(r["projected_turns"], 0.)
        self.assertIsNone(r["startup_s"])

    def test_late_drop_cancels_success(self):
        r = self.rollout(drop=300)["trials"][0]
        self.assertFalse(r["success"])
        self.assertIsNotNone(r["startup_s"])
        self.assertAlmostEqual(r["projected_turns"], 20 / (2 * np.pi))

    def test_vertical_pen_cannot_earn_projected_spin(self):
        r = self.rollout(vertical=True)["trials"][0]
        self.assertFalse(r["success"])
        self.assertEqual(r["projected_turns"], 0.)
        self.assertEqual(r["projection_invalid_fraction"], 1.)

    def test_wrong_direction_does_not_start(self):
        r = self.rollout(rate=-4.)["trials"][0]
        self.assertFalse(r["success"])
        self.assertIsNone(r["startup_s"])


if __name__ == "__main__":
    unittest.main()
