# SO-101 tactile simulation

An Isaac Sim 6.0 / Isaac Lab reconstruction of a lab SO-101 arm with DP-S2015
Elite tactile fingertips.  The goal is to **train tactile policies in simulation
and deploy them zero-shot to the real arm**.

## Status

**Working**

- Tactile geometry: 52 taxels per pad, coordinates taken from the measured
  `array.xlsx`, with a maximum CAD surface error of 0.00412 mm.  One Isaac
  `ContactSensor` each for the fixed and moving pad, fused into `(2, 52, 3)`,
  alongside the resultant `(2, 3)` and magnitudes `(2, 52)`.
- Isaac Lab interface: the policy observation is a normalised `(N, 312)`; the
  debug observation keeps the raw dimensions.
- **The scripted pick-and-place task runs end to end** on the nominal,
  hand-tuned configuration: lift +99 mm, ball landing 18.3 mm from the bowl
  centre (threshold 28.5 mm), peak 4.9 N with 0.0% clipped.
- It is **not yet robust**.  Across ten randomised episodes
  (`tactile_logs/randomized/`) it succeeds 7/10, and both pads stay in contact
  for anywhere between 0.0% and 88.5% of the carry, median 31.1%.  The three
  failures throw the ball 277-912 mm from the bowl.  Peak force barely moves
  (4.74-4.90 N) because it is set by the fixed closing angle, not by the
  randomised friction, mass or torque.
- Parallel scene: `BallPickPlaceSceneCfg` clones the table, ball, bowl and
  tactile sensors across environments; the ball settles with 0.0 mm error.
- Real-data alignment: the sample rate is corrected to 90.9 Hz from the raw
  sidecars in `Jingyi-Z/sotac` (the datasheet claims 83.3 Hz), and the
  observation distribution over 21 red-ball episodes is characterised.

**Open problem -- the contact model**

Simulation gives 2.6 active taxels and 1.8-2.4 N per pad; the real sensor gives
14.3-20.1 taxels and 6.2-8.2 N.  Per-taxel force is comparable (0.7-0.85 N in
sim, about 0.35 N real).  What differs is **how many taxels participate**.
Ruled out, with data:

| Attempt | Result |
| --- | --- |
| Contact-offset calibration | No effect, and 2 mm breaks the carry outright (both-pad contact 87% -> 39%) |
| Compliant-contact stiffness sweep (0 to 1e7, four decades) | Active taxels stay at 6/4; no change |
| Domain randomisation (10 episodes) | Distribution overlap improves by at most 0.04 |

A rigid sphere between two flat pads produces a contact count that the collision
algorithm decides; material parameters cannot move it.  The remaining directions
are a soft-body/FEM ball, or a TacSL-style SDF field with lateral load coupling.

**Measured on hardware, not yet fed back into sim**

- `wrist_flex` tops out at +79.3° on the real arm; the URDF says +95°.
- Command latency is 100-133 ms and is not modelled.
- Holding error is 0.2-0.7° on hardware; the simulation's position drives leave
  roughly 9 mm of gravity droop at their force limit.

## Layout

```text
.
├── build_scene.py              # Generates the scene USD
├── scene_config.py             # Dimensions, mounting poses, object layout, physics materials
├── so101_scene_spec.yaml       # Scene and sensor spec -- the source of the measured values
├── so101_descriptor.yaml       # Lula kinematics description
├── tactile_*.py                # Taxel geometry, fusion, SDF contact model
├── assets/                     # SO-101 USD, sensor CAD, taxel coordinates
├── calibration/                # Camera intrinsics/extrinsics data and results
├── scripts/
│   ├── tasks/                  # Scripted pick-and-place, grasp geometry, domain randomisation
│   ├── tactile/                # Tactile analysis, real-vs-sim comparison, visualisation
│   ├── isaac_lab/              # Parallel scene, recording, ManagerBasedRLEnv regressions
│   ├── calibration/            # Camera calibration workflow
│   ├── diagnostics/            # Joint, gripper, collision and reachability probes
│   ├── experiments/            # Historical experiments and parameter sweeps; not stable entry points
│   ├── rendering/              # Single-frame and video rendering
│   ├── streaming/              # WebRTC browser streaming
│   └── teleop/                 # Keyboard teleoperation
├── so101_isaac_lab/            # Tactile sensor, MDP terms, scene and environment configs
└── official_so101_workshop/    # NVIDIA's official workshop (its own repository, not tracked here)
```

## Environment

Requires Isaac Sim 6.0 and Isaac Lab in a conda environment named `isaacsim`.
Run every command from the project root.

The SO-101 URDF is expected next door at
`../SO-ARM100/Simulation/SO101/so101_new_calib.urdf`; override with
`SO101_URDF_PATH` if it lives elsewhere.

## Building the scene

The scene USD is not tracked.  Generate it with:

```bash
# Reference layout: aligned to real_scene.jpg, ball sitting in the bowl
conda run -n isaacsim python build_scene.py

# Task initial state: ball outside the bowl, for the pick-and-place task
SO101_SCENE_OUT=lab_scene_task.usda SO101_SCENE_LAYOUT=task_ready \
  conda run -n isaacsim python build_scene.py
```

Tactile geometry defaults to `cad + high_fidelity`.  To fall back:

```bash
SO101_TACTILE_GEOMETRY=mvp conda run -n isaacsim python build_scene.py
SO101_TACTILE_COLLISION=fast conda run -n isaacsim python build_scene.py
```

## Entry points

```bash
# Scripted pick-and-place (a deterministic expert, for physics validation)
conda run --no-capture-output -n isaacsim \
  python scripts/tasks/scripted_ball_pick_place.py

# The same, recorded to renders/pick_place.mp4
conda run --no-capture-output -n isaacsim \
  python scripts/tasks/scripted_ball_pick_place.py --video

# Tactile geometry regression
conda run -n isaacsim python scripts/tactile/validate_taxel_geometry.py

# Isaac Lab end-to-end regression
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/smoke_manager_env.py --viz none --num_envs 2

# Parallel scene recorded to renders/parallel_pick_place.mp4
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/record_parallel_scene.py --num_envs 9
```

`grasp_v2.py` and `pick_place_policy.py` are legacy entry points from when the
target was a cube; `scripts/tasks/scripted_ball_pick_place.py` supersedes them.
The tail of `scene_config.py` keeps `BASKET_* = BOWL_*` and
`TARGET_CUBE_* = TARGET_BALL_*` aliases so they still import, but they are not
maintained as stable entry points.

For the tactile implementation, its regression method and the uncalibrated
items, see `scripts/tactile/README.md`.  For the Isaac Lab tensor interface and
the training environment, see `so101_isaac_lab/README.md`.

## Keyboard teleoperation

Drives the arm from the keyboard inside the Isaac Sim GUI, writing
`UsdPhysics.DriveAPI` angle targets:

```bash
conda run --no-capture-output -n isaacsim \
  python scripts/teleop/keyboard_teleop.py
```

`TAB` switches between joint jogging and Cartesian translation, `1`–`6` select a
joint, `↑`/`↓` jog it, `W/S A/D Q/E` translate the gripper, `O`/`C` open and
close it, and `ESC` exits printing the final joint targets.  Full key table in
`scripts/teleop/README.md`.

Key events come from the Kit app window, so this cannot run headless.

## Browser streaming

The repository carries a local WebRTC client generated by NVIDIA's
`@nvidia/create-ov-web-rtc-app` (`web_client/`; restore its dependencies with
`npm install`):

```bash
./scripts/streaming/start_browser_viewer.sh
```

Then open `http://localhost:5173`.  The host firewall needs TCP 49100 and
UDP 47998 open.

**Note**: the media channel is UDP, so plain SSH port forwarding is not enough.
Signalling connects and the page then sits on "waiting for stream" forever.  To
watch remotely, record off-screen instead, e.g. with
`scripts/isaac_lab/record_parallel_scene.py`.

This stream has no authentication and no transport encryption.  Do not expose
5173, 49100 or 47998 to the public internet.

## Scene coordinate frame

- X runs along the length of the table.
- Y points from the arm's mounting edge into the table.
- Z is up, and **the tabletop is z = 0**.

That last point matters: in the Isaac Lab parallel scene the ground plane
therefore sits at z = -10 mm, the underside of the table slab.  At z = 0 it
would be coplanar with every tabletop in the grid and z-fight.

## Next steps

1. Grasp robustness.  7/10 under randomisation is not a base for RL: the ball
   is held across roughly two degrees of gripper closure, so anything that
   shifts the contact point drops a pad.  This gates everything below it.
2. Contact model: a soft-body ball or an SDF field with lateral coupling, to
   raise the number of active taxels.
3. Task rewards and terminations for the parallel environments.  Right now
   there are only the tactile terms `bilateral_contact` / `force_balance` /
   `overload` -- nothing expresses "put the ball in the bowl".
4. Feed the measured hardware calibration back into sim: tighten the
   `wrist_flex` limit, model command latency.
5. Randomise the ball and bowl poses.
6. Split the recording camera out of the training scene.  The `CameraCfg`
   currently lives on `BallPickPlaceSceneCfg`, which forces every consumer to
   pass `--enable_cameras`.
