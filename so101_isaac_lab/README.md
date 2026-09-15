# SO-101 tactile Isaac Lab interface

This layer wires the existing Isaac Sim scene and the 52-taxel fusion chain into
Isaac Lab 3.0's `ManagerBasedRLEnv`.  Isaac Sim still owns USD, PhysX contact
and rendering; Isaac Lab owns multi-environment cloning, the
action/observation/reward/termination managers, and the training tensors.

Verified against Python 3.12, Isaac Sim `6.0.1.0`, Isaac Lab `3.0.0b2.post1`
and Torch `2.11.0` (CUDA).  The `isaacsim` conda environment has these versions
installed and passes `pip check`.

## Tensor contract

The default environment `SO101TactileEnvCfg` runs 240 Hz physics with
`decimation=4`, so the policy steps at 60 Hz.  The two tactile sensors still
integrate independently at 83.3 Hz with sample-and-hold.

| Interface | Shape | Meaning |
| --- | --- | --- |
| `actions` | `(N, 6)` | Normalised position commands for the six joints |
| `obs["policy"]` | `(N, 312)` | Normalised tactile input, flattened as `[finger, taxel, xyz]` |
| `obs["tactile_debug"]["taxel_forces"]` | `(N, 2, 52, 3)` | Unnormalised three-axis force components, N |
| `obs["tactile_debug"]["force_magnitudes"]` | `(N, 2, 52)` | Per-taxel force magnitude, N |
| `obs["tactile_debug"]["total_forces"]` | `(N, 2, 3)` | Independent resultant-force channel, N |
| `rewards` | `(N,)` | Sum of the current generic tactile rewards |

`total_forces` is an independent channel, not a naive sum over the 52 quantised
points.  The unquantised spatial distribution conserves the resultant force
exactly; once each taxel independently applies its 0.1 N threshold and
quantisation, the two totals are allowed to differ by that quantisation.

The current rewards are an interface baseline only: bilateral contact `1.0`,
normal-force balance `0.1`, and an overload term `-0.1` past 20 N.  These are
not calibrated weights for a final grasping task.

## Key files

- `tactile_tensor.py` -- pure Torch, batched, runs on CPU or CUDA; the 52-point
  mapping and the output clock.
- `sensors/tactile_contact_sensor.py` -- one PhysX ContactSensor per finger,
  reading contact points, normal forces and friction forces, and transforming
  them into the real CAD sensor frame.
- `scene_cfg.py` -- robot, probe and two-sensor scenes, all clonable through
  `{ENV_REGEX_NS}`.
- `env_cfg.py` -- the production action, observation, reward and termination
  configuration.
- `assets/isaac_lab/so101_tactile.usda` -- a lightweight reference to the
  current `lab_scene.usda/World/Robot`; it does not copy the robot asset.

`InteractiveSceneCfg.lazy_sensor_update` must stay `False`.  Otherwise the
physics substeps that no observation reads never reach the 240 Hz tactile
integration.  The configuration pins this already.

## Regression commands

Run from the project root:

```bash
# Pure Torch: CPU + CUDA, shapes, rejection, quantisation, output clock, conservation
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/test_tactile_tensor.py

# Real PhysX dynamic contact across two cloned environments
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_tactile_observation.py \
  --viz none --num_envs 2 --steps 120 --contact

# Production ManagerBasedRLEnv: the zero-contact interface at the default decimation=4
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_manager_env.py \
  --viz none --num_envs 2 --steps 3

# Production ManagerBasedRLEnv: the non-zero contact path, driven by a scripted probe
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_manager_env.py \
  --viz none --num_envs 2 --steps 30 --contact
```

The small probe in contact mode resets every physics step, so that test
temporarily uses `decimation=1`; the production configuration stays at
`decimation=4`.  It is a deterministic regression stimulus, not training-scene
dynamics.

The current CAD sub-references make PhysX print a handful of `Failed to find
articulation at .../MillimetersToMeters/Geometry` warnings at startup.  They
come from nested CAD references that exist purely for appearance; the robot
articulation, both sensors' contact, and every tensor regression are fine.
Flattening the visual USD would remove the log noise later, and does not affect
using this interface.

## Presentation renders

`TactilePresentationSceneCfg` is a separate Isaac Lab presentation scene: it
keeps the full lab background and the real gripper, gives the left and right
fingertips their own dynamic probes, and renders 104 taxel heat points at their
true spatial positions through a Lab `CameraCfg`.  The probes are a repeatable
contact stimulus; they do not represent calibrated rubber dynamics.

```bash
conda run --no-capture-output -n isaacsim python \
  scripts/isaac_lab/render_tactile_presentation.py \
  --viz none --device cuda:0 --steps 180 --quality preview
```

Outputs:

- `renders/isaac_lab_tactile_scene_raw.png` -- the raw RTX camera image.
- `renders/isaac_lab_tactile_presentation.png` -- the presentation figure with
  the bilateral tactile and policy HUD.
- `tactile_logs/isaac_lab_tactile_presentation.json` -- the full `(2,52,3)`
  values for this contact, resultants, taxel world coordinates, peaks and
  conservation error.

On an idle GPU, `--quality final` raises the camera's accumulated frame count.

### Standalone 104-taxel force vector field

Vector-field mode draws no taxel heat spheres and hides both contact-stimulus
probes in the final render.  Taxels above `0.1 N` become arrows: the arrow's
origin is the taxel's world position, its direction the three-axis force
transformed into world coordinates, and its displayed length maps the force
magnitude onto 6-32 mm.  Unloaded taxels get no arrow, but all 104 positions
and their zero forces still go into the JSON.

```bash
conda run --no-capture-output -n isaacsim python \
  scripts/isaac_lab/render_tactile_vector_field.py \
  --viz none --device cuda:0 --steps 180 --quality preview
```

Its own outputs:

- `renders/isaac_lab_taxel_vector_field.png` -- the result with the vector
  legend HUD.
- `renders/isaac_lab_taxel_vector_field_raw.png` -- the RTX image without HUD.
- `tactile_logs/isaac_lab_taxel_vector_field.json` -- the `(2,52,3)` forces in
  both the sensor and world frames, true taxel positions, display start/end
  points, arrow lengths, and the indices of the active taxels.

## Newton backend (port started 2026-09-13)

Isaac Lab 3.0 selects the physics engine through `SimulationCfg.physics`.
`physics_presets.make_physics_cfg(name)` builds the concrete config:

| Preset | Solver | Contacts | Status |
| --- | --- | --- | --- |
| `physx` | PhysX 5 | PhysX | the calibrated reference |
| `newton_mjwarp` | MuJoCo-warp | MuJoCo's own detector | arm/drives verified |
| `newton_pipeline` | MuJoCo-warp | Newton `CollisionPipeline`, rigid | per-contact data available |
| `newton_hydro` | MuJoCo-warp | Newton pipeline + SDF hydroelastic | contact patch works, forces not yet calibrated |
| `newton_kamino*` | Kamino (P-ADMM) | internal / Newton pipeline | forces now read from its P-ADMM multipliers, but the arm's links land 124-433 mm away from where PhysX/MuJoCo put them at identical joint angles, so nothing is ever grasped |
| `newton_featherstone_hydro` | Featherstone | Newton pipeline + hydro | contact sensors unsupported (`update_contacts` not implemented) |
| `newton_xpbd*` | XPBD | Newton pipeline (+ hydro) | joint position drives inert in this release (arm never leaves its start pose even with gains x1000); unusable for the arm |

Everything below the config layer is backend-dispatched: `DPS2015ContactSensor`
is now a factory returning `DPS2015PhysXContactSensor` or
`DPS2015NewtonContactSensor`, both subclasses of `DPS2015TactileCore` (device
frame, taxel mapper, 83.3 Hz clock).  The Newton class keeps Isaac Lab's
Newton `ContactSensor` for registration and net forces and reads per-contact
positions, normals and solved forces straight from
`NewtonManager.get_contacts()` -- Isaac Lab's wrapper hides them
(`contact_pos_w` raises).  `newton_hooks.py` holds opt-in `MODEL_INIT` hooks
that edit the `ModelBuilder` before finalize: contact stiffness / MuJoCo
impedance, hydroelastic flags + SDFs, collision disabling.

`scripts/isaac_lab/probe_backend_pose.py` is the A/B harness: it replays the
nominal trajectory (`--until grip_hold`) on any preset and reports gripper
and jaw poses, joint errors, ball motion, per-phase taxel activity
(`--dps_sensors`), force-bearing contact pairs every 0.05 s
(`--trace_contacts`), the hydroelastic contact patch (`--hydro`), and the
geometric pad-to-ball gap of the simulated hulls.

### What was verified

- Articulation, drives and the xyzw root-quaternion compensation carry over:
  gripper and jaw poses agree with PhysX to 0.2 mm at every waypoint, joint
  tracking errors match.  Body ordering differs; look bodies up by name.
- The Newton tactile chain works end to end: at the moment the moving pad
  engages, `DPS2015NewtonContactSensor` reports the same force the raw
  contact buffer holds.  The PhysX sensor is unchanged by the refactor
  (close phase: fixed 5 taxels / 4.43 N, moving 4 / 4.42 N, as before).
- Simulated Newton collision hulls coincide with the USD geometry (gap check
  within 0.01 mm), so the differences below are contact-model differences,
  not import errors.

### Contact-model comparison

Force-controlled squeeze (ball mass raised to 2 kg so it stays put, gripper
effort limited to 0.3 N·m, adapters deactivated on both backends via
`assets/isaac_lab/so101_tactile_noadapter.usda` -- they never touch the ball
on PhysX either), env 0, `--taxel_threshold_n 0.1`:

| Backend / contact model | Pad engaged | Active taxels | Pad force | Patch |
| --- | --- | --- | --- | --- |
| PhysX rigid (reference) | both | 4 / 4 | 3.5-4.0 N | 2-3 points |
| Newton rigid (MuJoCo, solimp 0.99) | moving only, intermittent | 0-5 | 0-15 N spikes | 1 point |
| Newton hydro kh=1e7, first touch | moving | 18 | 11 N | 32 contacts |
| Newton hydro kh=1e6, first touch | moving | 23 | 8.4 N | 18 contacts |
| Newton hydro kh=1e7, close end | moving | 50-51 | 360-410 N (unstable) | 299 mm², 16x15 mm |
| Newton hydro kh=1e5, default MuJoCo impedance, first touch | moving | 15 | 7.6 N | 267 mm², 16x13 mm |
| Newton hydro kh=1e6, default MuJoCo impedance, first touch | moving | 7 | 2.3 N | -- |
| Real DP-S2015 (sotac dataset) | both | 14-20 | 6.2-8.2 N | -- |

(`solimp 0.99` rows use `--newton_ke 1e4 --newton_kd 200 --newton_solimp
0.99 0.999 0.001 0.5 2`; the default-impedance rows keep Newton's defaults.
With default impedance the forces stay bounded but the ball creeps 1-2.5 cm
under the one-sided push, so neither setting yet gives a settled two-pad
squeeze on MuJoCo.)

The hydroelastic generator delivers what the README's open problem asked
for: an area contact of 150-300 mm² (a 13-16 mm patch, about the pad's
width) that lights 18-50 taxels instead of the 4-5 that any rigid point
contact can reach, and at first touch the taxel count / force pair
(18-23 taxels at 8-11 N) sits right next to the hardware distribution.

Solver choice on this release is forced, not a preference: MuJoCo-warp is
the only Newton solver that both tracks the arm (0.2 mm vs PhysX) and reports
contact forces.  XPBD's joint drives are inert, Featherstone reports no
forces, Kamino mis-builds the kinematic chain (details under known issues).

What is not solved: MuJoCo-warp turns the hydroelastic stiffness into a
constraint impedance, and under a closing jaw the resulting forces run away
(hundreds of newtons, 1-3 cm ball displacement) instead of settling at the
drive's torque limit.  The same solver already fails the *rigid* pinch that
PhysX holds: its soft, mass-scaled contacts let a 0.2 N pad touch push the
4.8 g ball 1-2 cm into the table, and the calibrated 44° close (pads above
the equator) squeezes the ball out sideways.  Kamino holds the pinch with
hard contacts (ball settles exactly where PhysX puts it) but the Isaac Lab
integration never fills `contacts.force`, so no solver on this Newton
release gives both a stable grasp and sensor forces.

### Known issues (Newton 1.2.1 / isaaclab_newton 0.13.6)

- `finalize()` attaches a mesh SDF only for `GeoType.MESH`; a
  convexHull-approximated pad is `CONVEX_MESH` and silently gets none, so
  hydroelastic pairs produce nothing.  `register_hydroelastic` demotes the
  shape to MESH and builds a scale-baked SDF on the simulation device.
- The USD importer runs CoACD at threshold 0.5 for `convexDecomposition`,
  collapsing each concave tactile adapter to one hull that fills the pad's
  pocket.  Re-decomposing at MODEL_INIT adds shapes after replication and
  breaks MuJoCo's per-world geom layout; bake hulls into the USD instead or
  use the no-adapter asset.
- Newton's `ContactSensorCfg.from_base_cfg` rejects extra dataclass fields;
  the Newton sensor passes a stripped base cfg through.
- With `--enable_cameras` the tactile-mount visual meshes under the
  `MillimetersToMeters` xform render 1000x too large (Fabric world matrices
  from `body_q` ignore prim scale) and balls in envs 1-3 fell through the
  table in that mode only.  Not investigated further.
- Kamino needs `rigid_contact_max` above its own contact count (65536 in the
  preset) and reports articulation body poses at the wrong index.
- XPBD's revolute joint-target constraint (`solvers/xpbd/kernels.py`, marked
  `# TODO fix`) uses `1/ke` as compliance without the dt^2 scaling and ignores
  `kd`; the SO-101 does not move under it at all.
- Kamino's `update_contacts` converts geometry only; the Newton sensor reads
  its P-ADMM multipliers directly (`_contact_forces_on_shape0`).
- Kamino's `ModelKamino.from_newton` rewrites the articulation in place and
  the result no longer matches the asset.  Measured with an identity base
  orientation (`--init_rot 0 0 0 1`) and converged, identical joint angles
  (agreement 1e-5), against the MuJoCo pipeline on the same USD:
  `joint_X_p` is unchanged for the root and the first joint but rewritten for
  every joint below it, following `X_p_kamino[j] = X_p_kamino[parent(j)] o
  X_p_newton[j]` -- each joint inherits its parent's rotation a second time,
  compounding down the chain (`joint_X_c` is identity before and after, so
  the documented "absorb non-identity child frames" pass had nothing to
  absorb).  Link positions then diverge from `upper_arm_link` onward by
  124 mm, growing to 433 mm at the gripper frame, with orientation errors of
  90-180°, and the collision shapes move with them (pad hulls end up 0.2-0.5 m
  from where PhysX places them), so the contacts Kamino solves are not the
  contacts of this robot.  Baking the base yaw into the USD does not help --
  the chain is already wrong with no rotation at all; a yawed root
  additionally destabilises the solve (joint errors of 2.5-3.6 rad).  Kamino
  also converges much more slowly than MuJoCo: the arm needs over 1 s to
  reach a held target that MuJoCo reaches within 1 s.  Do not call `eval_fk`
  on `state_0` while Kamino steps -- it corrupts the solve.

### Next steps

1. Calibrate the hydro force scale: sweep `kh` and `--hydro_layer_mm`
   against the drive torque, or wait for a Newton release whose Featherstone
   / Kamino solvers report contact forces (penalty contacts would use kh as
   a real spring).
2. Read the taxel map from the hydro contact surface directly
   (`hydro_surface_summary` already extracts the iso-surface) instead of the
   reduced contact set, then rerun `analyze_feature_transfer.py` against
   the hardware distribution.
3. Bake a proper adapter decomposition into the asset and fix the Fabric
   render scale before any parallel video on Newton.
