"""Gesture vocabulary and choreography for the DH116 hand.

A gesture is six numbers: normalized thumb side swing in [-1, 1] (-1 at
the vendor's 0-degree limit, +1 at 60 degrees) and five flexions in [0, 1].
``hand_targets`` expands the six commands to 11 joint angles with the
manufacturer's nonlinear passive-joint curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dh116_joint_model import finger_passive_angle, thumb_passive_angle

from xarm5_dh116_lab.robot_cfg import (
    ALL_JOINTS,
    ARM_PRESENT,
    FINGER_MCP_MAX,
    FINGER_PIP_MAX,
    HAND_JOINTS,
    THUMB_ABDUCTION_MAX,
    THUMB_FLEXION_MAX,
    THUMB_IP_MAX,
)

#            abd  thumb index middle ring pinky
GESTURES: dict[str, tuple[float, ...]] = {
    "open":      (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "paper":     (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "rock":      (0.6, 1.0, 1.0, 1.0, 1.0, 1.0),
    "scissors":  (0.6, 1.0, 0.0, 0.0, 1.0, 1.0),
    "one":       (0.6, 1.0, 0.0, 1.0, 1.0, 1.0),
    "two":       (0.6, 1.0, 0.0, 0.0, 1.0, 1.0),
    "three":     (0.6, 1.0, 0.0, 0.0, 0.0, 1.0),
    "four":      (0.6, 1.0, 0.0, 0.0, 0.0, 0.0),
    "five":      (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumbs_up": (-1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
}
RPS = ("rock", "paper", "scissors")
COUNT = ("one", "two", "three", "four", "five")


def hand_targets(gesture: str | tuple[float, ...]) -> np.ndarray:
    """Joint targets (rad) in HAND_JOINTS order for a gesture."""
    abd, thumb, index, middle, ring, pinky = (
        GESTURES[gesture] if isinstance(gesture, str) else gesture)
    thumb_flex = thumb * THUMB_FLEXION_MAX
    thumb_side = 0.5 * (abd + 1.0) * THUMB_ABDUCTION_MAX
    out = [thumb_side, thumb_flex,
           min(thumb_passive_angle(thumb_flex), THUMB_IP_MAX)]
    for frac in (index, middle, ring, pinky):
        mcp = frac * FINGER_MCP_MAX
        out += [mcp, min(finger_passive_angle(mcp), FINGER_PIP_MAX)]
    assert len(out) == len(HAND_JOINTS)
    return np.array(out, dtype=np.float32)


def arm_targets(pose: dict[str, float] = ARM_PRESENT) -> np.ndarray:
    return np.array([pose[name] for name in ALL_JOINTS[:5]], dtype=np.float32)


def quintic(phase: float) -> float:
    """Smooth 0->1 blend with zero velocity and acceleration at both ends."""
    p = min(max(phase, 0.0), 1.0)
    return p * p * p * (10.0 - 15.0 * p + 6.0 * p * p)


@dataclass
class Segment:
    label: str
    gesture: str
    duration_s: float
    blend_s: float = 0.35
    # optional oscillation added to one arm joint: (joint name, amplitude
    # rad, frequency Hz).  A bob on joint4 nods the fist; a wave on joint5
    # rolls the open hand about the finger axis.
    wobble: tuple[str, float, float] | None = None
    arm: dict[str, float] = field(default_factory=lambda: dict(ARM_PRESENT))


def choreography(rng: np.random.Generator) -> list[Segment]:
    """Idle -> count to five -> three rounds of rock-paper-scissors (with a
    fist shake before each throw) -> wave -> thumbs up -> idle."""
    segments = [Segment("idle", "open", 1.0)]
    counting = list(COUNT)
    if rng.random() < 0.3:
        counting.reverse()
    for name in counting:
        segments.append(Segment(f"count {name}", name, 0.8))
    segments.append(Segment("idle", "open", 0.6))
    for round_index in range(3):
        segments.append(Segment(
            f"shake {round_index + 1}", "rock", 1.35,
            wobble=("joint4", 0.22, 2.2)))
        throw = str(rng.choice(RPS))
        segments.append(Segment(f"throw {throw}", throw, 1.2))
    segments.append(Segment("wave", "open", 2.4, wobble=("joint5", 0.35, 1.25)))
    # Thumbs up needs the thumb on top: fingers towards the camera (joint4 = 0
    # makes the pitch sum -pi/2) and the palm turned to the viewer's right
    # (joint5 = 3pi/2), measured in record_gestures.py --probe.
    segments.append(Segment("thumbs up", "thumbs_up", 1.8, blend_s=0.6,
                            arm={**ARM_PRESENT, "joint4": 0.0,
                                 "joint5": 1.5 * math.pi}))
    segments.append(Segment("idle", "open", 1.0, blend_s=0.6))
    return segments


def build_timeline(segments: list[Segment], hz: float
                   ) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Dense joint targets, shape (steps, 16) in ALL_JOINTS order, plus the
    (start step, label) list."""
    rows = []
    labels = []
    previous = np.concatenate([arm_targets(), hand_targets("open")])
    for segment in segments:
        steps = max(1, int(round(segment.duration_s * hz)))
        goal = np.concatenate([arm_targets(segment.arm),
                               hand_targets(segment.gesture)])
        blend_steps = max(1, int(round(segment.blend_s * hz)))
        # the caption changes half-way through the blend into the gesture
        labels.append((len(rows) + blend_steps // 2, segment.label))
        for k in range(steps):
            alpha = quintic(k / blend_steps)
            target = previous + alpha * (goal - previous)
            if segment.wobble is not None:
                name, amplitude, freq = segment.wobble
                t = k / hz
                # fade the wobble in and out over the segment
                envelope = math.sin(math.pi * min(k / steps, 1.0)) ** 0.5
                target = target.copy()
                target[ALL_JOINTS.index(name)] += (
                    amplitude * envelope * math.sin(2.0 * math.pi * freq * t))
            rows.append(target)
        previous = goal
    return np.array(rows, dtype=np.float32), labels


def label_at(labels: list[tuple[int, str]], step: int) -> str:
    current = labels[0][1]
    for start, label in labels:
        if step >= start:
            current = label
    return current
