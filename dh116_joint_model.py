"""Vendor DH116 joint limits and nonlinear passive-joint coupling.

The coefficients are from Leadshine's 2026-04-09 DH116 URDF/MJCF release.
Inputs and outputs of the helpers are radians.  The original vendor curves
were fitted in degrees; the quadratic coefficients below include the unit
conversion needed to evaluate them in radians.
"""

from __future__ import annotations

import math

VENDOR_RELEASE = "2026-04-09"
VENDOR_DOWNLOAD = "https://store.leadshine.com/pages/download-center"

THUMB_ABDUCTION_MIN = 0.0
THUMB_ABDUCTION_MAX = math.radians(60.0)
THUMB_FLEXION_MIN = 0.0
THUMB_FLEXION_MAX = math.radians(30.0)
FINGER_FLEXION_MIN = 0.0
FINGER_FLEXION_MAX = math.radians(80.0)

# q_passive = offset + linear*q_active + quadratic*q_active**2, radians.
THUMB_COUPLING = (math.radians(0.2631), 0.8808, 0.0082 * 180.0 / math.pi)
FINGER_COUPLING = (math.radians(0.6344), 0.9206, 0.0035 * 180.0 / math.pi)


def coupled_angle(active_angle, coefficients):
    """Evaluate a vendor passive-joint curve for a scalar or array-like value."""
    offset, linear, quadratic = coefficients
    return offset + linear * active_angle + quadratic * active_angle * active_angle


def thumb_passive_angle(active_angle):
    return coupled_angle(active_angle, THUMB_COUPLING)


def finger_passive_angle(active_angle):
    return coupled_angle(active_angle, FINGER_COUPLING)


THUMB_PASSIVE_MIN = thumb_passive_angle(THUMB_FLEXION_MIN)
THUMB_PASSIVE_MAX = thumb_passive_angle(THUMB_FLEXION_MAX)
FINGER_PASSIVE_MIN = finger_passive_angle(FINGER_FLEXION_MIN)
FINGER_PASSIVE_MAX = finger_passive_angle(FINGER_FLEXION_MAX)

# The published URDF omits motors/electronics and sums to 237.67046 g.  The
# manual specifies 545 +/- 10 g for the complete hand.  Generated assets keep
# the published moving-link inertias and place the missing mass on the palm.
NOMINAL_HAND_MASS_KG = 0.545
VENDOR_URDF_MASS_KG = 0.23767046

