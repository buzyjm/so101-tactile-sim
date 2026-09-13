"""Checks for the Leadshine 2026-04-09 DH116 calibration."""

import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from dh116_joint_model import (
    FINGER_FLEXION_MAX, FINGER_PASSIVE_MAX, FINGER_PASSIVE_MIN,
    NOMINAL_HAND_MASS_KG, THUMB_ABDUCTION_MAX, THUMB_FLEXION_MAX,
    THUMB_PASSIVE_MAX, THUMB_PASSIVE_MIN, finger_passive_angle,
    thumb_passive_angle,
)

ROOT = Path(__file__).resolve().parents[1]


class JointModelTests(unittest.TestCase):
    def test_vendor_curve_endpoints(self):
        self.assertAlmostEqual(math.degrees(THUMB_PASSIVE_MIN), 0.2631, places=4)
        self.assertAlmostEqual(math.degrees(THUMB_PASSIVE_MAX), 34.0671, places=4)
        self.assertAlmostEqual(math.degrees(FINGER_PASSIVE_MIN), 0.6344, places=4)
        self.assertAlmostEqual(math.degrees(FINGER_PASSIVE_MAX), 96.6824, places=4)
        self.assertAlmostEqual(thumb_passive_angle(THUMB_FLEXION_MAX), THUMB_PASSIVE_MAX)
        self.assertAlmostEqual(finger_passive_angle(FINGER_FLEXION_MAX), FINGER_PASSIVE_MAX)

    def test_generated_hand_urdf(self):
        root = ET.parse(ROOT / "assets/dh116_hand/dh116_hand.urdf").getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        expected = {
            "hand_finger11": (0.0, THUMB_ABDUCTION_MAX),
            "hand_finger12": (0.0, THUMB_FLEXION_MAX),
            "hand_finger13": (THUMB_PASSIVE_MIN, THUMB_PASSIVE_MAX),
        }
        for finger in range(2, 6):
            expected[f"hand_finger{finger}1"] = (0.0, FINGER_FLEXION_MAX)
            expected[f"hand_finger{finger}2"] = (FINGER_PASSIVE_MIN, FINGER_PASSIVE_MAX)
        for name, (lower, upper) in expected.items():
            limit = joints[name].find("limit")
            self.assertAlmostEqual(float(limit.get("lower")), lower, places=7)
            self.assertAlmostEqual(float(limit.get("upper")), upper, places=7)
            self.assertIsNone(joints[name].find("mimic"))
        mass = sum(float(link.find("inertial/mass").get("value"))
                   for link in root.findall("link"))
        self.assertAlmostEqual(mass, NOMINAL_HAND_MASS_KG, places=7)

        arm_root = ET.parse(ROOT / "assets/xarm5_dh116/xarm5_dh116.urdf").getroot()
        installed_mass = sum(
            float(link.find("inertial/mass").get("value"))
            for link in arm_root.findall("link")
            if link.get("name", "").startswith("hand_")
        )
        self.assertAlmostEqual(installed_mass, NOMINAL_HAND_MASS_KG, places=7)


if __name__ == "__main__":
    unittest.main()
