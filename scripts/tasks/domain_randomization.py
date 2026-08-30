"""Per-episode domain randomisation for the SO-101 tactile task.

Randomisation covers parameters we cannot measure, so a policy never comes to
depend on any single value.  It does NOT cover model structure: the simulated
contact patch is 4-6 taxels against 16-23 on hardware, and no amount of gain or
noise jitter turns a point contact into a distributed one.  That gap has to be
fixed in the contact model, not here.

Ranges are grounded where hardware data allows:

* friction is the widest band on purpose -- 1.6 was tuned until the grasp held,
  the specification itself says "unknown; tune or measure experimentally", and
  nothing has ever measured the plush ball against the pads;
* ball mass is measured (4.81 g) so it barely moves;
* drive force spans the gravity droop the position drives already exhibit;
* contact offset stays inside the band that keeps taxels reporting at all --
  scripts/experiments/test_seated_grasp.py showed the signal dies from 5 mm up.

This deliberately leaves object poses alone.  Moving the ball invalidates the
runtime grasp calibration, which then has to rerun per episode; that is a
performance problem to solve before pose randomisation is worth switching on.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class RandomizationRanges:
    """Inclusive sampling bounds, one entry per randomised parameter."""

    static_friction: tuple[float, float] = (1.30, 2.00)
    ball_mass_kg: tuple[float, float] = (0.0043, 0.0053)
    joint_max_force: tuple[float, float] = (2.70, 4.00)
    contact_offset_m: tuple[float, float] = (0.0010, 0.0035)
    # Only used by --tactile-close.  The grasp itself is not yet robust enough
    # to randomise: a rigid sphere between two flat pads holds across roughly
    # two degrees of closure, and closing angle, friction and even closing
    # *duration* all move that window.  Randomising the grip is pointless until
    # the contact model spreads the load the way the hardware does.
    target_grip_force_n: tuple[float, float] = (4.0, 5.5)
    # Device clock: the recorded sidecars sit at a median 11.00 ms with a p99
    # of 15.5 ms, so allow a little jitter around the nominal rate.
    sample_period_scale: tuple[float, float] = (0.98, 1.06)


@dataclass(frozen=True)
class EpisodeRandomization:
    seed: int
    static_friction: float
    dynamic_friction: float
    ball_mass_kg: float
    joint_max_force: float
    contact_offset_m: float
    target_grip_force_n: float
    sample_period_scale: float

    def as_dict(self) -> dict:
        return asdict(self)


def sample_episode(seed: int, ranges: RandomizationRanges | None = None
                   ) -> EpisodeRandomization:
    """Draw one episode's parameters; the seed alone reproduces the episode."""
    ranges = ranges or RandomizationRanges()
    rng = np.random.default_rng(seed)

    def uniform(bounds: tuple[float, float]) -> float:
        return float(rng.uniform(*bounds))

    static_friction = uniform(ranges.static_friction)
    return EpisodeRandomization(
        seed=seed,
        static_friction=static_friction,
        # PhysX combines per-material values; keep the pair physically ordered.
        dynamic_friction=static_friction * float(rng.uniform(0.80, 0.92)),
        ball_mass_kg=uniform(ranges.ball_mass_kg),
        joint_max_force=uniform(ranges.joint_max_force),
        contact_offset_m=uniform(ranges.contact_offset_m),
        target_grip_force_n=uniform(ranges.target_grip_force_n),
        sample_period_scale=uniform(ranges.sample_period_scale),
    )


def apply(stage, episode: EpisodeRandomization, ball_path: str,
          contact_prim_paths, joint_root: str, joint_names) -> None:
    """Write one episode's parameters onto the open stage."""
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/Randomized")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr().Set(episode.static_friction)
    physics.CreateDynamicFrictionAttr().Set(episode.dynamic_friction)
    physics.CreateRestitutionAttr().Set(0.0)
    for path in (ball_path, *contact_prim_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"randomisation target missing: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose="physics")
        # Rest offset first: PhysxCollisionAPI creates each attribute at its
        # schema default, and PhysX validates the pair on every write.
        for descendant in Usd.PrimRange(prim):
            if descendant.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(descendant)
                collision.CreateRestOffsetAttr().Set(0.0)
                collision.CreateContactOffsetAttr().Set(
                    episode.contact_offset_m)

    mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(ball_path))
    mass.CreateMassAttr().Set(episode.ball_mass_kg)

    for name in joint_names:
        drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{joint_root}/{name}"), "angular")
        if drive:
            drive.CreateMaxForceAttr().Set(episode.joint_max_force)
