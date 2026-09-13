"""Forward kinematics of the merged xArm5 + DH116 URDF, in plain numpy.

Used to choose arm poses offline (no simulator needed) and to cross-check
the simulated link poses in ``record_gestures.py --probe``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = PROJECT_ROOT / "assets" / "xarm5_dh116" / "xarm5_dh116.urdf"


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


class UrdfChain:
    """Every joint of the URDF with its parent, origin, axis and limits."""

    def __init__(self, path: Path = URDF_PATH) -> None:
        root = ET.parse(path).getroot()
        self.joints: dict[str, dict] = {}
        self.child_of: dict[str, str] = {}
        for joint in root.iter("joint"):
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0")
                                                if origin is not None
                                                else "0 0 0").split()])
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0")
                                      if origin is not None
                                      else "0 0 0").split()]
            axis_el = joint.find("axis")
            axis = np.array([float(v) for v in (axis_el.get("xyz")
                                                 if axis_el is not None
                                                 else "0 0 1").split()])
            limit = joint.find("limit")
            self.joints[joint.get("name")] = {
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": xyz,
                "rot": _rpy(*rpy),
                "axis": axis,
                "lower": float(limit.get("lower", 0.0)) if limit is not None else 0.0,
                "upper": float(limit.get("upper", 0.0)) if limit is not None else 0.0,
            }
            self.child_of[joint.find("child").get("link")] = joint.get("name")

    def limits(self, name: str) -> tuple[float, float]:
        return self.joints[name]["lower"], self.joints[name]["upper"]

    def link_pose(self, link: str, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """World pose (position, rotation matrix) of ``link`` for joints ``q``.

        The URDF root link is the world frame."""
        chain = []
        while link in self.child_of:
            joint = self.child_of[link]
            chain.append(joint)
            link = self.joints[joint]["parent"]
        pos = np.zeros(3)
        rot = np.eye(3)
        for joint in reversed(chain):
            spec = self.joints[joint]
            pos = pos + rot @ spec["xyz"]
            rot = rot @ spec["rot"]
            if spec["type"] in ("revolute", "continuous"):
                rot = rot @ _axis_angle(spec["axis"], q.get(joint, 0.0))
        return pos, rot


def solve_present_pose(chain: UrdfChain, reach: float, height: float,
                       palm_dir: np.ndarray = np.array([1.0, 0.0, 0.0])) -> dict[str, float]:
    """Arm joints that hold the hand upright (fingers +Z) with the palm facing
    ``palm_dir``, the hand base near (reach, 0, height) in the base frame.

    joints 2-4 share a parallel axis, so their sum fixes the hand pitch:
    sum = pi points the fingers up.  joint5 (roll about the finger axis)
    turns the palm, joint1 is left at zero.  The remaining two degrees of
    freedom are searched on a grid for the closest hand-base position.
    """
    best = None
    lo2, hi2 = chain.limits("joint2")
    lo3, hi3 = chain.limits("joint3")
    lo4, hi4 = chain.limits("joint4")
    for j2 in np.arange(lo2, hi2, 0.02):
        for j3 in np.arange(lo3, hi3, 0.02):
            j4 = math.pi - j2 - j3
            if not lo4 <= j4 <= hi4:
                continue
            q = {"joint1": 0.0, "joint2": float(j2), "joint3": float(j3), "joint4": float(j4), "joint5": 0.0}
            pos, rot = chain.link_pose("hand_base_link", q)
            # elbow must stay above the base and in front of it
            elbow, _ = chain.link_pose("link3", q)
            if elbow[2] < 0.15:
                continue
            err = math.hypot(pos[0] - reach, pos[2] - height)
            if best is None or err < best[0]:
                best = (err, q, pos, rot)
    err, q, pos, rot = best
    # roll so that the palm normal (+X of the hand base) faces palm_dir
    x_axis = rot[:, 0]
    z_axis = rot[:, 2]
    y_axis = rot[:, 1]
    target = palm_dir - np.dot(palm_dir, z_axis) * z_axis
    target /= np.linalg.norm(target)
    roll = math.atan2(np.dot(np.cross(x_axis, target), z_axis), np.dot(x_axis, target))
    # joint5 turns about the hand's own +Z (finger axis): positive joint5 is
    # a positive rotation about z_axis in the world.
    q["joint5"] = float(roll)
    return q


if __name__ == "__main__":
    chain = UrdfChain()
    q = solve_present_pose(chain, reach=0.32, height=0.55)
    pos, rot = chain.link_pose("hand_base_link", q)
    print("present pose:", {k: round(v, 4) for k, v in q.items()})
    print("hand base pos:", np.round(pos, 3))
    print("palm normal (+x):", np.round(rot[:, 0], 3), " fingers (+z):", np.round(rot[:, 2], 3))
    for link in ("link2", "link3", "link4", "link5"):
        p, _ = chain.link_pose(link, q)
        print(f"  {link}: {np.round(p, 3)}")
    for name in ("hand_finger21_link", "hand_finger11_link", "hand_finger51_link"):
        p, r = chain.link_pose(name, q)
        print(f"  {name}: {np.round(p, 3)}")
