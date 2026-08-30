"""Isaac Lab sensor implementations for the tactile gripper."""

from .tactile_contact_sensor import (
    DPS2015ContactSensor,
    DPS2015ContactSensorCfg,
)

__all__ = ["DPS2015ContactSensor", "DPS2015ContactSensorCfg"]
