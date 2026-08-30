"""Radial tool-pose geometry for the SO-101 ball task.

Pure numpy, no Isaac imports, so the pick-and-place task and the offline IK
probes share exactly one definition of the grasp pose.  The SO-101 is a 5-DOF
arm, so the tool frame is always built around the radial direction from the
base to the object; that is the orientation family the arm can actually reach.
"""

from __future__ import annotations

import math

import numpy as np

# This side-entry pose was established against the current SO-101 Lula model.
# The downward pitch lets the fingers clear the table.  The lateral correction
# accounts for the asymmetric moving jaw.
DEFAULT_TOOL_TILT_DEG = 15.0
TCP_OFFSET_M = 0.025
LATERAL_OFFSET_M = 0.015
GRASP_HEIGHT_OFFSET_M = 0.015


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3 x 3 rotation matrix to a wxyz quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / math.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / scale,
                (rotation[2, 1] - rotation[1, 2]) * scale,
                (rotation[0, 2] - rotation[2, 0]) * scale,
                (rotation[1, 0] - rotation[0, 1]) * scale,
            ]
        )

    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        scale = 2.0 * math.sqrt(
            1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
        )
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ]
        )
    if index == 1:
        scale = 2.0 * math.sqrt(
            1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
        )
        return np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ]
        )

    scale = 2.0 * math.sqrt(
        1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
    )
    return np.array(
        [
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        ]
    )


class RadialToolGeometry:
    """Tool poses on the radial approach line from ``base_xy`` to an object."""

    def __init__(self, base_xy) -> None:
        self.base_xy = np.asarray(base_xy, dtype=float)[:2]

    def motion_yaw(self, object_center) -> float:
        direction = np.asarray(object_center, dtype=float)[:2] - self.base_xy
        return math.atan2(direction[1], direction[0])

    @staticmethod
    def motion_rotation(motion_yaw: float) -> np.ndarray:
        return np.array(
            [
                [math.cos(motion_yaw), -math.sin(motion_yaw), 0.0],
                [math.sin(motion_yaw), math.cos(motion_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def tool_pose(
        self, object_center, tilt_deg: float = DEFAULT_TOOL_TILT_DEG
    ) -> dict:
        """Frame position, orientation, tool rotation, and insertion axis."""
        object_center = np.asarray(object_center, dtype=float)
        world_from_motion = self.motion_rotation(self.motion_yaw(object_center))

        tilt = math.radians(tilt_deg)
        tool_axis_motion = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
        tool_x_motion = np.cross(np.array([0.0, 1.0, 0.0]), tool_axis_motion)
        tool_x_motion /= np.linalg.norm(tool_x_motion)
        tool_y_motion = np.cross(tool_axis_motion, tool_x_motion)
        tool_rotation_motion = np.column_stack(
            [tool_x_motion, tool_y_motion, tool_axis_motion]
        )
        # Roll 90 degrees so the fingers close laterally across the ball.
        gripper_roll = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )

        tool_rotation_world = (
            world_from_motion @ tool_rotation_motion @ gripper_roll
        )
        tool_axis_world = world_from_motion @ tool_axis_motion
        frame_position = (
            object_center
            + world_from_motion
            @ np.array([0.0, LATERAL_OFFSET_M, GRASP_HEIGHT_OFFSET_M])
            + TCP_OFFSET_M * tool_axis_world
        )
        return {
            "frame_position": frame_position,
            "orientation": rotation_to_quaternion(tool_rotation_world),
            "tool_rotation": tool_rotation_world,
            "insertion_axis": tool_axis_world,
        }

    def held_offset_tool(
        self,
        object_center,
        held_world_position,
        tilt_deg: float = DEFAULT_TOOL_TILT_DEG,
    ) -> np.ndarray:
        """Where a held object sits relative to the IK frame, in tool axes.

        The grasped ball is rigid with respect to the gripper, so this offset is
        the invariant: it stays constant as the approach yaw and the tool tilt
        change, which the motion-frame offset does not.
        """
        pose = self.tool_pose(object_center, tilt_deg)
        residual = (
            np.asarray(held_world_position, dtype=float)
            - pose["frame_position"]
        )
        return pose["tool_rotation"].T @ residual

    def object_for_held_target(
        self,
        held_target,
        held_offset_tool,
        tilt_deg: float = DEFAULT_TOOL_TILT_DEG,
        iterations: int = 6,
    ) -> np.ndarray:
        """Object argument that puts a held object on ``held_target``.

        The tool frame yaws toward its object argument, so an object held at a
        fixed tool-frame offset sweeps through world space as that yaw changes.
        The yaw depends on the argument being solved for, so iterate the fixed
        point; the offset is a few centimetres and this converges immediately.
        """
        held_target = np.asarray(held_target, dtype=float)
        object_center = held_target.copy()
        for _ in range(iterations):
            pose = self.tool_pose(object_center, tilt_deg)
            held_world = (
                pose["frame_position"] + pose["tool_rotation"] @ held_offset_tool
            )
            object_center = object_center + (held_target - held_world)
        return object_center
