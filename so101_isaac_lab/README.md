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
