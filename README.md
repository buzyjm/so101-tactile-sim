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
  The scripted pick-and-place replays across 25 staggered environments with
  per-environment layouts (`--vary_deg`): each clone's ball and bowl are
  rotated about the measured shoulder-pan axis by their own random azimuths
  (bowl -15..+5 deg, ball a further -5..+25 deg away from it, so the two
  never overlap), with matching pan offsets on that environment's pick- and
  place-phase waypoints -- radius stays nominal, so the grasp geometry is
  exactly equivalent.  The bowl is split into a kinematic physics twin
  (accepts per-env pose writes, never renders) and a collision-free static
  render twin (follows per-env USD edits); see `assets/isaac_lab/
  bowl_phys.usda` / `bowl_scenery.usda`.  25/25 with ~131 mm lift and every
  ball settling on its own bowl's floor (`renders/parallel_pick_place_expert
  .mp4`, closeup in `renders/pick_place_closeup.mp4`).  The transport hold uses a kinematic
  attach -- see the carry-divergence note below; grasp, squeeze and the final
  drop into the bowl are unmodified physics.  A pure-physics grasp-cycle
  variant (`--choreo grasp_cycle`, two-stage 4.5 N / 10.3 N hold, 25/25) is
  in `renders/parallel_grasp_cycle.mp4`.
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

**Open problem -- the carry diverges in the cloned Isaac Lab pipeline**

The scripted trajectory replays in `BallPickPlaceSceneCfg` and matches the
standalone run to 0.1 mm in pad-apex positions and 0.1 N in pad forces through
approach, descend and close -- yet the ball detaches the moment the lift
starts, where the standalone carries it.  Invariant to command rate (60 vs
240 Hz), drive gains, solver iterations, CPU vs GPU physics, contact
compliance (rigid to 1500 N/m), close depth (4.5 to 10 N), wrist path, and
removing the tactile sensors entirely; ball-table friction behaves correctly
in isolation (slide-to-roll transition as expected).  Until this is resolved
the parallel pick-and-place video bridges the carry with a kinematic attach
(`--attach`): once both pads report >= 1 N at the end of close, the ball is
pose-coupled to `gripper_link` until the jaw opens past the ball diameter
during release, then dropped.  Approach, close, the squeeze forces (the pads
still press 3.5-4 N against the coupled ball throughout the carry) and the
drop-and-settle into the bowl are unmodified physics.  A fully pure-physics
alternative is the grasp-cycle choreography (`--choreo grasp_cycle`), also
verified 25/25.

Three pipeline traps found while calibrating, all fixed in-tree:

- The root-pose write consumes the `init_state.rot` quaternion shifted by one
  component (effectively expecting xyzw where the config documents wxyz), so
  the documented ordering pitched every clone 90 degrees onto its side --
  the broken "mount angle" in the first parallel render.  Compensated in
  `_TASK_INIT_STATE`, calibrated against the standalone gripper pose to
  sub-millimetre agreement.
- Physics-material bindings that target scene-level paths
  (`/World/PhysicsMaterials/...`) die when the robot or bowl is referenced
  into a clone; the pads silently fell back to friction 0.5.  The Isaac Lab
  assets now declare their materials inside the referenced subtree.
- The asset's `root_joint` is consumed as pure fixed-base topology; its
  authored frames are ignored, so base orientation cannot be corrected there.
- An empty `Material` prim (physics-only, no surface shader) renders pitch
  black wherever Kit's binding resolution picks it up -- the fingertip pads
  went black when the friction rebind was first added.  The Isaac Lab
  gripper-pad material now carries a real PreviewSurface, and the tactile
  mounts get their two-tone look back via inherited `displayColor` primvars.
- Static colliders ignore post-parse USD transform edits (the render follows,
  the physics stays put), and kinematic rigid bodies take tensor pose writes
  but never render.  Per-environment placement of scenery therefore needs
  the split-twin pattern used for the bowl.
- RTX streams static meshes in lazily: short probe renders (about a second)
  can show a mesh as missing that appears fine two seconds into a real
  recording.  The recorder holds an idle 3 s lead-in for that reason; judge
  rendering from mid-video frames, never from short probes.

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
├── xarm5_dh116_lab/            # xArm5 + DH116 Isaac Lab package: robot/scene cfg, FK, gestures
├── dh116_hand_lab/             # DH116 hand alone: pen-spinning Direct RL env + PPO config
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

# Parallel scripted pick-and-place, per-env layouts, verified and recorded
conda run --no-capture-output -n isaacsim \
  python scripts/isaac_lab/record_parallel_pick_place.py \
  --num_envs 25 --env_spacing 1.4 --attach --vary_deg 12 --stagger_s 3.0 --record
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
./scripts/streaming/start_browser_viewer.sh            # lab_scene.usda
./scripts/streaming/start_browser_viewer.sh path/to/scene.usda
```

The script starts three things: the headless Isaac Sim streaming app pinned
to one GPU (`STREAM_GPU`, default 0), a TURN relay (`coturn` in Docker on the
host network, TCP/UDP 3478) and the Vite dev server for the client.  Then:

- **On the machine or its LAN**: open `http://<lan-ip>:5173`.  The media goes
  straight over UDP 47998; the TURN relay is only a fallback.
- **Remote over SSH or VS Code**: forward three TCP ports and open
  `http://localhost:5173`:

  ```bash
  ssh -L 5173:localhost:5173 -L 49100:localhost:49100 -L 3478:localhost:3478 <user>@<host>
  ```

  The WebRTC media channel is UDP and an SSH tunnel only carries TCP, which is
  why earlier attempts sat on "waiting for stream" after the signalling
  connected.  When the page is opened as `localhost` the client now sends the
  media through the TURN relay over TCP (`iceTransportPolicy: relay`); the
  relay forwards it to Isaac Sim over UDP on the host.  Verified from a
  headless Chrome that could only reach `localhost`: H.264 1920x1080 at
  60 fps, selected candidate pair `relay(tcp) -> host udp :47998`.

Query parameters on the viewer: `?ice=relay|all|none` (default `relay` on
localhost, `all` elsewhere), `?turn=host:port`, `?turnuser=`, `?turnpass=`,
`?server=` and `?signal=` for the signalling endpoint.

The same script can stream an Isaac Lab script instead of the stock
streaming app, e.g. the DH116 gesture show live (the viewer can orbit the
viewport with the mouse):

```bash
STREAM_APP_CMD="python scripts/xarm5_dh116/record_gestures.py --stream --num_envs 9 --env_spacing 1.3" \
    ./scripts/streaming/start_browser_viewer.sh
```

The launcher exports `STREAM_KIT_ARGS` (GPU pinning, media address) and
`PUBLIC_IP` for the app, which enables `--livestream 1` itself.  A second
stream next to a running one takes other ports (`SIGNAL_PORT=49110
MEDIA_PORT=47999 WEB_PORT=5174`; then open `http://localhost:5174/?signal=49110`)
and shares the TURN container.  Note for other Isaac Lab scripts: Lab 3.0
only pumps the Kit app loop through a Kit visualizer, which the headless
livestream launch leaves inert -- the browser then connects and the encoder
times out "waiting for frame".  `record_gestures.py --stream` pumps
`simulation_app.update()` itself once per control step (with
`/app/player/playSimulations` off, as Lab's own visualizer does).  `STREAM_DEBUG=1`
turns on StreamSDK logging (ICE, encoder, QoS) in the Kit log; the Kit log
lives under `~/.nvidia-omniverse/logs/Kit/Isaac-Sim Streaming/6.0/`.

Two more things the script fixes that also broke streaming before: the
advertised media address is pinned to the LAN IP (StreamSDK otherwise picks
an interface itself, on this machine possibly a Docker bridge), and rendering
is restricted to one GPU.  With both GPUs enabled the app also initialised
CUDA/OptiX on the second card and, when that card was busy, failed with
`cudaErrorMemoryAllocation` before encoding a single frame.

This stream and the TURN relay have no authentication and no transport
encryption (fixed credentials `so101`/`so101`).  Do not expose 5173, 49100,
47998 or 3478 to the public internet.

## Scene coordinate frame

- X runs along the length of the table.
- Y points from the arm's mounting edge into the table.
- Z is up, and **the tabletop is z = 0**.

That last point matters: in the Isaac Lab parallel scene the ground plane
sits at the measured lab-floor height z = -0.735 m and the cloned tables
stand on their pedestals, same as the standalone scene.

## xArm5 + DH116 showcase

A second, independent robot: a UFactory xArm5 carrying a Leadshine DH116
11-DoF dexterous hand.  `--env lab` (default) stands it on the table of
Isaac Sim's `Simple_Room` with a single prop; `--env warehouse` is the
busier `Simple_Warehouse` + packing-table variant.  Sources, licences and
the changes made to the vendor URDFs are documented in
[assets/xarm5_dh116/README.md](assets/xarm5_dh116/README.md).

```bash
# 1. (once) merge the vendor URDFs + flange adapter into assets/xarm5_dh116/xarm5_dh116.urdf
python scripts/xarm5_dh116/build_urdf.py --xarm-urdf <xarm5 expanded urdf> \
    --xarm-meshes <xarm_ros/xarm_description/meshes> --dh116-pkg <DH116-R000-A1>
# 2. (once) URDF -> USD with the Isaac Sim 6.0 importer
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/convert_assets.py
# 3. build the scene, settle physics, save xarm5_dh116_<env>.usda and render
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/build_scene.py --preview          # fast look
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/build_scene.py                    # 4K RTX final
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/build_scene.py --env warehouse    # warehouse variant
```

`build_scene.py` searches the shoulder/elbow angles that put the flange at a
chosen reach and height above the table (`hover` per env), so the hand hovers
over the prop instead of touching it; `--views wrist` renders a close-up of
the flange adapter.  The default renderer is RTX real-time (`--renderer rt`,
12 x 16 subframes): it is noticeably sharper than the path tracer's denoised
output for these white, low-detail meshes.

## DH116 gesture show (Isaac Lab)

The xArm5 + DH116 also has its own Isaac Lab package, `xarm5_dh116_lab/`,
independent of the SO-101 code: an `ArticulationCfg` for the merged asset
(`robot_cfg.py`), a clone-safe scene with a pedestal and one camera per
environment (`scene_cfg.py`), a numpy forward-kinematics of the URDF for
choosing arm poses offline (`kinematics.py`) and a gesture vocabulary for
the 11-DoF hand (`gestures.py`).  A gesture is six numbers -- thumb
side swing and five flexions -- and the passive PIP/IP joints follow the
quadratic coupling curves in Leadshine's 2026-04-09 MJCF release.

```bash
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/record_gestures.py --probe
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/record_gestures.py --num_envs 25
conda run --no-capture-output -n isaacsim python scripts/xarm5_dh116/record_gestures.py --num_envs 4 --closeup
```

Every environment holds the hand up to the camera and runs its own random
take: count to five, three rounds of rock-paper-scissors with a fist shake,
a wave, a thumbs up (`renders/dh116_gestures.mp4`, captioned close-up of
env 0 in `renders/dh116_gestures_closeup.mp4`).  It is joint-target
kinematics under physics -- nothing is grasped. The historical recordings
and 0.7-degree tracking result used disabled self-collisions. Current defaults
enable self-collisions: contact blocks infeasible finger targets, so those
tracking numbers no longer describe the current model.

Two things found on the way, both checked with `--probe` (simulated link
positions against the FK):

- This asset hits the same quaternion trap as the SO-101 one: Isaac Lab
  3.0's root-pose write path and its data API use (x, y, z, w), whatever
  the `InitialStateCfg` docstring says.  The documented identity
  `(1, 0, 0, 0)` hung the arm upside down under the pedestal; every link
  came back at exactly -z of the FK.  `robot_cfg.py` writes `(0, 0, 0, 1)`.
- The "presenting" pose is joints 2-4 summing to -pi (fingers up) with the
  forearm horizontal and the wrist bent 90 degrees; joint5 = pi turns the
  palm to the camera.  Thumbs up needs the fingers pointed at the camera
  (joint4 = 0) and the palm turned sideways (joint5 = 3pi/2): the thumb
  only spreads in the palm plane.

## DH116 self-collision correction

Both the hand-only and xArm5 + DH116 configurations now use the derived
`assets/<robot>/usd_manual_2026_04_self_collision/<robot>/` assets.
Self-collisions are enabled, with 32 position iterations for hand-only and
64 for the arm-mounted hand, plus 8 velocity solver iterations, articulation
contacts solved last, and 1 m/s maximum hand-link depenetration velocity.
The shared instanced colliders explicitly use a 1 mm contact offset and
zero rest offset; runtime recursive overrides skipped these instance proxies.
Visual geometry and materials are unchanged. No extra collision-pair filters
were added; the physics engine's adjacent-link exclusions remain.

Seven deterministic gesture phases at 120 Hz passed the local acceptance
checks for both robots after applying the 2026-04 vendor joint model. Maximum
sampled contact penetration was 0.081 mm (hand) / 0.406 mm (arm + hand), with
less than 1 degree transient joint-limit violation. This checks the tested
contact shapes and poses, not every possible visual-mesh overlap or real-hand
behavior. See the [manual calibration report](reports/dh116_manual_calibration_2026-09-10.md),
the [earlier collision report](reports/dh116_self_collision_2026-09-08.md) and
[corrected hand gesture video](renders/dh116_self_collision_hand.mp4).

```bash
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/audit_self_collision.py \
  --robot hand --device cuda:0 --output logs/collision_audit/hand_repeat.json
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/audit_self_collision.py \
  --robot arm --device cuda:0 --output logs/collision_audit/arm_repeat.json
```

Training and playback default to `--collision_profile current`, including
when loading an old checkpoint. Evaluation defaults to `--collision_profile
saved` to reproduce the checkpoint's physics; explicitly select `current`
to evaluate migration to the corrected model. A newly trained checkpoint
saves the corrected configuration, so its `saved` evaluation also enables
self-collisions. The scripts print the selected profile and asset path.
Historical checkpoints evaluated with disabled self-collisions are not
evidence of performance under the corrected physics.

## DH116 pen spinning (Isaac Lab, RL)

The hand on its own, palm up, targeting pen rotation about the palm normal
on a 6-DoF-actuated hand. This is a DH116 adaptation in progress, not a
completed reproduction of the Eureka demo.  Package
`dh116_hand_lab/` (independent of the arm package): `hand_cfg.py`
(stand-alone hand articulation, palm-up base pose, pen spawn/home points),
`pen_spin_env_cfg.py` / `pen_spin_env.py` (Direct RL env: 41-d observation,
6 actions expanded to 11 targets with the vendor's nonlinear passive-joint curves, reward = signed
pen angular velocity about world Z clipped to [-2, 10] rad/s, plus an
on-hand bonus, a fall penalty and action-rate costs) and `agents.py`
(rsl_rl PPO).  Asset: `assets/dh116_hand/` cut from the merged URDF by
`scripts/dh116_hand/build_hand_urdf.py` and converted by `convert_hand.py`.

```bash
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/flick_test.py --envs_per_pattern 8     # scripted sanity check
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/train_pen_spin.py --num_envs 4096 --max_iterations 3000
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/play_pen_spin.py --checkpoint logs/rsl_rl/dh116_pen_spin/<run>/model_2999.pt --num_envs 8 --record
```

Needs `rsl-rl-lib==5.0.1` (the version `isaaclab_rl` pins; 5.5 changed the
model constructor).  `isaaclab_rl` still serialises the pre-5.0 model keys
(`stochastic`, `init_noise_std`, ...) as MISSING, which rsl_rl 5.x rejects;
the train/play scripts strip them (`clean_agent_cfg`).

**v1 result corrected on 2026-09-07:** the old report of median 6.7 rad/s
and 20 turns integrated the angular velocity's world-Z component. The pen
is tilted and mostly rolls around its own long axis; this gives a large
world-Z angular velocity without its long axis going around the palm.
The old drop counter also read state after automatic reset and could miss
falls. Historical videos have these old labels and selected a best case.

The corrected evaluation loads the checkpoint's saved configuration,
snapshots state before reset, and scores only the first episode of each
environment. It measures turns from the pen long axis's projected azimuth.
For `model_2999.pt`, two seeds (11, 29), 256 environments per seed, 16 fixed
yaw bins and 32 seconds produced:

| Metric | Seed 11 | Seed 29 |
| --- | ---: | ---: |
| Old signed omega-Z median, rad/s | 6.39 | 6.64 |
| Actual projected azimuth mean, rad/s | -0.0013 | 0.0022 |
| Trials that dropped | 58/256 | 52/256 |
| Sustained palm-spin successes | 0/256 | 0/256 |

Success requires a 1-second window averaging at least 3 rad/s by 5 seconds,
no drop over the whole trial, and mean projected rate after the first second
of at least 3 rad/s. Drops use the existing position envelope, not contact
sensing. This is a new, longer, fixed-start protocol; the historical 32-env
figures are not directly comparable. Full evidence is described in
[the progress report](reports/dh116_pen_spin_2026-09-07.md).

```bash
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/eval_pen_spin.py --info \
  --checkpoint logs/rsl_rl/dh116_pen_spin/2026-09-06_02-13-16_v1/model_2999.pt \
  --num_envs 256 --seconds 32 --seed 11 --yaw_bins 16 \
  --output logs/evaluations/dh116_v1/seed11_repeat.json
```

`play_pen_spin.py` restores the saved task configuration and applies the
current collision profile by default. It displays projected
turns, and records the explicitly selected `--record_env` (default 0),
without automatically selecting the most successful environment. It stops
crediting a trial after the first drop, even if the simulator resets it.

**Training without an LLM:** `run_reward_experiments.py` runs explicitly
written reward variants through train -> checkpoint -> evaluation -> report.
It needs no model server or API. It is not the automatic LLM reward search
from Eureka. The historical paired pilot used the same v1 checkpoint, training seed,
4096 environments, 300 additional iterations, 16-second episodes and PPO
entropy in both branches. `omega_z_control` keeps v1's reward;
`heading_v2` rewards measured long-axis azimuth change with symmetric
clipping so a forward/backward swing cancels. Tilt cost stays zero in this
comparison. The ranking uses fixed evaluation metrics, never reward totals.
That September 7 pilot used the original physics with self-collisions off.
The command below now trains with the corrected physics and evaluates each
candidate's saved configuration; it is a new experiment, not an exact
reproduction of the historical pilot.

```bash
conda run --no-capture-output -n isaacsim python scripts/dh116_hand/run_reward_experiments.py \
  --checkpoint logs/rsl_rl/dh116_pen_spin/2026-09-06_02-13-16_v1/model_2999.pt \
  --output logs/reward_experiments/dh116_heading_self_collision \
  --devices cuda:0 cuda:1 --iterations 300 --num_envs 4096
```

Each output directory must be new. Training saves source hashes, source
copies, config YAMLs and a final-checkpoint manifest. The experiment folder
contains evaluation JSONs, reward-component curves, logs and `summary.json`.
No candidate is promoted when all candidates have zero success. This pilot
has one training seed and two evaluation seeds; it is not a multi-training-seed
performance comparison. Baseline source was preserved before edits under
`logs/baselines/dh116_v1_source_2026-09-07/`.

The 300-iteration paired pilot completed: both variants scored 0/512
sustained-spin successes. The heading reward largely stopped axial rolling
and reduced mean tilt from 23.94 to 12.16 degrees, but did not produce palm
spinning; drop rates were 21.29% (control) and 25.00% (heading v2).
No candidate was promoted. The next training step is a reachable target-pose
curriculum, not a claim that the pen-spinning task is solved.

Things that mattered:

- Where the pen rests.  The base_link mesh's x = 0.028 m is the thumb
  mount, not the palm; a 5 mm capsule settles with its centre at x = 0.019
  and, because the palm slopes towards the fingers, rolls on to the finger
  bases (hand z ~ 0.12).  Spawning it at the knuckles put it inside the
  covers and launched it at 1.7 m/s; it now spawns 1.4 cm above the palm
  near the wrist and drops into place.
- The thumb sweeps 3-5 cm above the palm plane (FK of the vendor
  geometry) and only touches a pen at full abduction, where it knocks it
  off; the reset therefore leaves the thumb raised.
- The fall check is relative to the centre of the palm/finger area with a
  10 cm / 6 cm tolerance; a tight box around the assumed rest point
  terminated calm episodes at 0.2 s and made every early metric noise.

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
