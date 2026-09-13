import math
import unittest
import torch
from dh116_hand_lab.rewards import projected_spin_rate, update_best_progress


class RewardGeometryTests(unittest.TestCase):
    def test_axial_rolling_earns_nothing(self):
        # Rotation about this tilted long axis leaves its direction unchanged.
        axis = torch.tensor([[math.cos(.4), 0., math.sin(.4)]])
        self.assertEqual(projected_spin_rate(axis, axis, 1 / 60, -1.).item(), 0.)

    def test_crossing_angle_boundary(self):
        a, b = math.pi - .03, -math.pi + .03
        previous = torch.tensor([[math.cos(a), math.sin(a), 0.]])
        axis = torch.tensor([[math.cos(b), math.sin(b), 0.]])
        self.assertAlmostEqual(projected_spin_rate(previous, axis, .02, 1.).item(), 3., places=5)

    def test_back_and_forth_cancels(self):
        a = torch.tensor([[1., 0., 0.]])
        b = torch.tensor([[math.cos(.08), math.sin(.08), 0.]])
        forward = projected_spin_rate(a, b, .02, -1.).clamp(-10, 10)
        backward = projected_spin_rate(b, a, .02, -1.).clamp(-10, 10)
        self.assertEqual((forward + backward).item(), 0.)

    def test_vertical_axis_is_unobservable(self):
        a = torch.tensor([[0., 0., 1.]])
        b = torch.tensor([[0., 1., 0.]])
        self.assertEqual(projected_spin_rate(a, b, .02, -1.).item(), 0.)

    def test_best_progress_cannot_be_reearned_by_oscillation(self):
        cumulative = torch.tensor([0.0])
        best = torch.tensor([0.0])
        earned = 0.0
        reached = False
        for delta in (0.2, 0.2, -0.3, 0.2, -0.1, 0.4):
            cumulative, best, reward, hit = update_best_progress(
                cumulative, best, torch.tensor([delta]), target=1.0)
            earned += reward.item()
            reached = reached or hit.item()
        self.assertAlmostEqual(cumulative.item(), 0.6, places=6)
        self.assertAlmostEqual(best.item(), 0.6, places=6)
        self.assertAlmostEqual(earned, 0.6, places=6)
        self.assertFalse(reached)


if __name__ == "__main__":
    unittest.main()
