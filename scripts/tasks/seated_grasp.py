"""Solve and calibrate a grasp that seats the ball on both tactile pads.

The scripted task originally swept the gripper in sideways and closed hard.
That lifted the ball but pinched it at the jaw tips, so the moving pad read
nothing and the fixed pad saturated.  Four things are needed instead, each
measured rather than assumed:

* the tool pose is solved so the ball rests on the fixed pad's centroid;
* the tool orientation and its position must share one base-yaw plane, since
  the SO-101 has five joints -- orienting toward the ball is 6.6 deg off and
  makes the pose infeasible;
* the approach descends vertically (collision-free at 15 deg tilt), because a
  lateral sweep shoves the ball to the fingertips before the jaws close;
* the commanded joint angles are NOT where the arm ends up.  The position
  drives are limited to 3.35 N m, so gravity leaves roughly 9 mm of
  steady-state error.  The grasp frame is therefore calibrated against the pad
  pose the simulation actually reaches.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, UsdGeom


def _matrix(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def _rotation(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(3)] for i in range(3)]).T


def seated_ball_in_pad_frame(taxels: np.ndarray, radius: float) -> np.ndarray:
    """Ball centre resting on the taxel centroid, in the pad's own frame.

    This is the ball CENTRE, not the contact point: seating the centre on the
    array centroid lands the resulting contact near y = +2 mm, which is already
    close to the +1.5 mm measured on hardware.  Aiming the centre itself at the
    measured contact centroid pushes the contact another 7 mm down into the
    fingertips and reproduces the 25 N crush the seated grasp exists to avoid.
    """
    taxels = np.asarray(taxels, dtype=float)
    centroid = taxels.mean(axis=0)
    for offset in np.linspace(0.5 * radius, 2.0 * radius, 4001):
        candidate = centroid + np.array([0.0, 0.0, offset])
        if abs(np.linalg.norm(taxels - candidate, axis=1).min() - radius) < 2e-5:
            return candidate
    raise RuntimeError("could not solve the seated ball centre")


def seated_ball_in_gripper_link(
    stage, taxels, radius, pad_prim_path, gripper_link_prim_path
) -> np.ndarray:
    """The same point, expressed in the gripper link's axes."""
    world = lambda path: UsdGeom.Xformable(
        stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    pad_in_link = world(pad_prim_path) * world(gripper_link_prim_path).GetInverse()
    seated = seated_ball_in_pad_frame(taxels, radius)
    return (np.append(seated, 1.0) @ _matrix(pad_in_link))[:3]


def seated_grasp_frame(
    stage, geometry, ball_world, tilt_deg, taxels, radius,
    pad_prim_path, gripper_link_prim_path, gripper_frame_prim_path,
    iterations: int = 6,
):
    """IK frame pose whose jaw seats the ball, consistent with 5-DOF yaw."""
    world = lambda path: UsdGeom.Xformable(
        stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
    w_link = world(gripper_link_prim_path)
    w_frame = world(gripper_frame_prim_path)
    link_in_frame = w_link * w_frame.GetInverse()
    frame_origin_in_link = np.asarray(
        (w_frame * w_link.GetInverse()).ExtractTranslation())
    seated = seated_ball_in_gripper_link(
        stage, taxels, radius, pad_prim_path, gripper_link_prim_path)
    ball_world = np.asarray(ball_world, dtype=float)

    virtual_object = ball_world.copy()
    frame = orientation = None
    for _ in range(iterations):
        pose = geometry.tool_pose(virtual_object, tilt_deg)
        tool = np.asarray(pose["tool_rotation"])
        frame_rot = Gf.Matrix4d()
        frame_rot.SetRotate(Gf.Matrix3d(*tool.T.flatten()))
        link_rot = _rotation(link_in_frame * frame_rot)
        frame = link_rot @ frame_origin_in_link + (ball_world - link_rot @ seated)
        orientation = pose["orientation"]
        frame_yaw = math.atan2(frame[1], frame[0])
        if abs(frame_yaw - geometry.motion_yaw(virtual_object)) < 3.0e-4:
            break
        radius_xy = float(np.linalg.norm(virtual_object[:2]))
        virtual_object = np.array([radius_xy * math.cos(frame_yaw),
                                   radius_xy * math.sin(frame_yaw),
                                   virtual_object[2]])
    return frame, orientation


def calibrate_grasp_frame(
    stage, grasp_frame, ball_target, taxels, radius, pad_prim_path,
    settle_at, attempts: int = 5, tolerance: float = 0.0015, report=None,
):
    """Shift the grasp frame until the pad's *measured* seat hits the ball.

    ``settle_at(frame)`` must drive the arm to that frame with the gripper open
    and step physics until it stops moving; it returns nothing.  Everything is
    read back from the stage afterwards, so no kinematic model is trusted.
    """
    seated = seated_ball_in_pad_frame(taxels, radius)
    ball_target = np.asarray(ball_target, dtype=float)
    grasp_frame = np.asarray(grasp_frame, dtype=float).copy()
    for attempt in range(attempts):
        settle_at(grasp_frame)
        pad_to_world = UsdGeom.Xformable(
            stage.GetPrimAtPath(pad_prim_path)).ComputeLocalToWorldTransform(0)
        seat_world = np.asarray(
            pad_to_world.Transform(Gf.Vec3d(*seated)), dtype=float)
        residual = ball_target - seat_world
        if report is not None:
            report(attempt, seat_world, residual)
        if float(np.linalg.norm(residual)) < tolerance:
            return grasp_frame, float(np.linalg.norm(residual))
        grasp_frame = grasp_frame + residual
    return grasp_frame, float(np.linalg.norm(residual))
