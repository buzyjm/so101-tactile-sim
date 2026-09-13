"""Reward geometry without simulator dependencies (xyzw is handled by the caller)."""
import torch


def projected_spin_rate(previous_axis, axis, dt: float, sign: float):
    """Signed change of the pen long-axis azimuth, excluding axial rolling.

    Directions with XY norm <= 0.2 are unobservable and earn zero. atan2 of
    cross/dot handles the +/-pi boundary; sampling assumes < pi per control step.
    """
    cross = previous_axis[:, 0] * axis[:, 1] - previous_axis[:, 1] * axis[:, 0]
    dot = (previous_axis[:, :2] * axis[:, :2]).sum(dim=-1)
    observable = (previous_axis[:, :2].square().sum(-1) > 0.04) & (axis[:, :2].square().sum(-1) > 0.04)
    return torch.where(observable, sign * torch.atan2(cross, dot) / dt, 0.0)


def update_best_progress(cumulative, best, heading_delta, target: float):
    """Advance net heading and pay only movement beyond the prior maximum."""
    cumulative = cumulative + heading_delta
    new_best = torch.maximum(best, cumulative)
    increment = (new_best - best) / target
    reached = cumulative >= target
    return cumulative, new_best, increment, reached
