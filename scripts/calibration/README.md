# SO-101 real-camera calibration

All transforms use the explicit convention `T_A_from_B`: the matrix maps a
point expressed in frame B into frame A. Real camera images and joint samples
are calibration data; the scripts never write results into `scene_config.py`
automatically.

## What to do in the lab

Generate and print the board before going to the robot. Intrinsic calibration
can be captured first and solved later from lossless frames taken by the camera
being calibrated at its final 640x480 setting--not by a phone. For the fixed
top-camera extrinsic, also measure the printed O/X/Y board pose relative to the
SO-101 base. Wrist hand-eye photos must be captured with
`capture_wrist_handeye.py`: each stopped-pose image needs the matching joint
encoder values, so unrelated photos cannot be synchronized afterward.

## 1. Generate and print the board

```bash
conda run -n isaacsim python scripts/calibration/generate_charuco_board.py
```

Print `calibration/board/charuco_a4_8x6_25mm.pdf` on A4 landscape paper at
**Actual size / 100%**. Disable Fit to page. Use matte paper and mount it flat
on a 3-5 mm foamboard, PVC sheet, acrylic sheet, or another rigid backing. Do
not use glossy lamination, curved cardboard, fabric, or a backing that stretches
the paper. Measure the full 8-square width: it should be 200 mm. If its physical
size differs, update `square_length_m` and scale `marker_length_m` by the same
ratio before solving.

The board origin `O` is the **outer top-left corner of the printed pattern**;
the PDF includes an arrow pointing to that exact corner. Printed +X points
right and +Y points down the sheet. The board +Z direction is `+X cross +Y`.

## 2. Capture and solve intrinsics

Identify cameras first inside the official container:

```bash
lerobot-find-cameras opencv
```

Capture 30-50 diverse views from each camera. SPACE saves and Q exits:

```bash
conda run -n isaacsim python scripts/calibration/capture_intrinsics.py \
  --camera 0 --name top

conda run -n isaacsim python scripts/calibration/capture_intrinsics.py \
  --camera 2 --name wrist
```

Replace the indices with the detected devices. Then solve:

```bash
conda run -n isaacsim python scripts/calibration/calibrate_intrinsics.py --name top
conda run -n isaacsim python scripts/calibration/calibrate_intrinsics.py --name wrist
```

Results are written to `calibration/results/`; visual axis overlays are written
to `calibration/reports/`.

## 3. Fixed top-camera extrinsic

Do not move the camera after intrinsic capture. Put the board at one fixed,
measured pose. The printed O/X/Y frame is the board frame. Measure O in the
SO-101 base frame and record board roll/pitch/yaw.

Capture several still frames without moving either camera or board:

```bash
conda run -n isaacsim python scripts/calibration/capture_intrinsics.py \
  --camera 0 --name top --purpose extrinsic --max-images 10
```

Example only—the numbers below must be replaced by measurements:

```bash
conda run -n isaacsim python scripts/calibration/calibrate_top_extrinsic.py \
  --board-x 0.20 --board-y 0.15 --board-z 0.0 \
  --board-roll-deg 0 --board-pitch-deg 0 --board-yaw-deg 0
```

## 4. Wrist hand-eye capture

Photos alone are insufficient: every image must be paired with the joint
positions from the same stopped pose. The board must remain fixed for the whole
sequence. The capture script owns the serial port; stop all other robot-control
programs first.

Add the following mount to the official `docker run` command, then start
`teleop-docker` as usual:

```bash
-v /home/buzyjm/isaac_work:/workspace/isaac_work
```

Inside the container run:

```bash
python /workspace/isaac_work/scripts/calibration/capture_wrist_handeye.py \
  --camera "$CAMERA_GRIPPER" \
  --robot-port "$ROBOT_PORT" \
  --robot-id "$ROBOT_ID" \
  --passive
```

Passive mode disables torque only after an explicit confirmation. Support the
arm before confirming and keep fingers clear of pinch points. Collect 20-30
poses with varied translation and rotations about multiple axes.

Solve on the host with Isaac Sim/Lula FK:

```bash
conda run -n isaacsim python scripts/calibration/solve_wrist_handeye.py
```

The solver compares TSAI, PARK, HORAUD, ANDREFF and DANIILIDIS methods, then
selects the result that makes the fixed board pose most consistent across all
robot poses.

## 5. Visual verification

```bash
conda run -n isaacsim python scripts/calibration/verify_calibration.py \
  --images calibration/raw/top_intrinsics \
  --intrinsics calibration/results/top_intrinsics.yaml \
  --output calibration/reports/top_verify
```

Do not import a calibration into Isaac Sim until detection overlays and numeric
errors have been reviewed. OpenCV camera axes are +X right, +Y down, +Z forward;
USD camera axes differ and require a later axis conversion during scene import.
