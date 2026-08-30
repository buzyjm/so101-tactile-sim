# Keyboard teleoperation

`keyboard_teleop.py` drives the SO-101 from the keyboard inside the Isaac Sim
GUI.  It writes `UsdPhysics.DriveAPI` angle targets on
`/World/Robot/Physics/<joint>` -- the same interface `grasp_v2.py` uses, so the
two are interchangeable.

```bash
conda run --no-capture-output -n isaacsim \
  python scripts/teleop/keyboard_teleop.py
```

Common options:

```text
--usd PATH              scene to open, default lab_scene_task.usda
--mode joint|cartesian  starting mode, default joint
--joint-speed DEG_S     joint jog speed, default 45 deg/s
--linear-speed M_S      Cartesian translation speed, default 0.08 m/s
--gripper-speed DEG_S   gripper open/close speed, default 90 deg/s
--self-test STEPS       run both modes for STEPS frames on synthetic key
                        events instead of waiting for a keyboard, then exit
```

`--self-test` regresses the control chain with nobody sitting at the window: it
jogs all six joints in turn, closes the gripper once, then translates along
±X/±Y/±Z in Cartesian mode, printing the joint angles and TCP position that
actually changed on each step.  It still needs a window, because the keyboard
subscription hangs off the app window.

```bash
conda run --no-capture-output -n isaacsim \
  python -u scripts/teleop/keyboard_teleop.py --self-test 30
```

## Keys

| Key | Action |
| --- | --- |
| `TAB` | Switch between joint and Cartesian mode |
| `1`–`6` | Select joint (joint mode) |
| `↑` / `↓` | Jog the selected joint (joint mode) |
| `W`/`S` `A`/`D` `Q`/`E` | Translate the gripper along world X / Y / Z (Cartesian mode) |
| `Z` / `X` | Jog `wrist_roll` (Cartesian mode) |
| `O` / `C` | Open / close the gripper (both modes) |
| `R` | Return to the pose the scene loaded with |
| `H` | Print help |
| `ESC` | Exit and print the final joint targets |

## The two modes

**joint** writes a single joint target.  No IK, and no calibration beyond the
URDF, which makes it the right tool for chasing down limits, drive gains and
contact.

**cartesian** solves for `gripper_frame_link` with the same
`LulaKinematicsSolver` that `so101_descriptor.yaml` and `grasp_v2.py` use,
warm-started from the current joint targets.  Entering the mode locks the
current tool orientation and translates only from there, which stops the solver
jumping to a different IK branch near the tabletop.  A step whose IK fails is
not committed, so the Cartesian target never accumulates drift, and the
unreachable position is printed to the terminal.

## Limitations

- A real window is required.  Key events come from the Kit app window's carb
  input interface, so this cannot run headless, nor over the `--no-window`
  stream that `start_browser_viewer.sh` sets up.
- Joint limits are read at runtime from the USD `physics:lowerLimit` /
  `physics:upperLimit`.  There is no self-collision check --
  `scripts/diagnostics/check_collision.py` still has to be run separately.
- Teleoperation records no trajectory.  Collecting data needs
  `isaacsim.replicator.episode_recorder` wired up separately.
