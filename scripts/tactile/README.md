# DP-S2015-Elite tactile CAD integration

The default scene uses the supplied STEP-derived CAD for both modified gripper
parts and both DP-S2015-Elite sensors. Each sensor keeps the occurrence
transform from its STEP assembly, including the approximately 12-degree mount
angle. Sensor 1 is fixed/gripper-side and sensor 2 is moving/wrist-roll-side.

The 52 measured taxel positions are loaded directly from
`assets/tactile/array.xlsx`. Both sensors share that local coordinate table.
The positions define a closed convex physical contact shell and are exported,
with their CAD surface distances, to
`assets/tactile/taxel_coordinates.csv`. Runtime uses one Isaac ContactSensor
per fingertip, not 52 sensors. Each PhysX contact position and impulse is
transformed into the CAD sensor frame and distributed over the six nearest
taxels by a normalized spatial kernel. The public output now contains both the
backwards-compatible aggregate `(2, 3)` force and the full fixed/moving
`(2, 52, 3)` taxel tensor.

In the native CAD frame the active surface faces local `+Z`. The output model
converts the physical compressive force to positive device `Fz`.

The hardware-inspired aggregate output model applies:

- 83.3 Hz sample-and-hold output;
- Fx/Fy clipping to [-10, 10] N;
- Fz clipping to [0, 25] N;
- 0.1 N minimum detectable component and 0.1 N/LSB quantization;
- exact pre-quantization vector-force conservation across the 52 taxels;
- a 1.5 mm Gaussian spatial width and 4 mm active-surface gate, both explicitly
  provisional until paired contact-location calibration is available.

Build and run the independent geometry and contact checks:

```bash
conda run -n isaacsim python build_scene.py
conda run -n isaacsim python scripts/tactile/validate_taxel_geometry.py
conda run -n isaacsim python scripts/tactile/test_taxel_fusion.py
conda run -n isaacsim python scripts/tactile/run_contact_mvp.py
```

The builder defaults to real CAD rendering and high-fidelity collision. The
legacy procedural appearance and fast collision modes remain available:

```bash
SO101_TACTILE_GEOMETRY=mvp conda run -n isaacsim python build_scene.py
SO101_TACTILE_COLLISION=fast conda run -n isaacsim python build_scene.py
```

The contact check uses 240 Hz physics and a gravity-disabled dynamic probe for
each active surface. Each probe is driven inward along that sensor-local
normal, so both world-space orientations receive the same solver-generated
contact. The test writes `tactile_logs/contact_mvp_result.json` and requires
both sensor outputs to cross the modeled 0.1 N detection threshold.

Render CAD inspection views with:

```bash
conda run -n isaacsim python scripts/rendering/render_tactile.py
```

Generate an annotated contact view driven by the same live ContactSensor data:

```bash
conda run -n isaacsim python scripts/tactile/visualize_contact_mvp.py
```

This writes `renders/tactile_contact_active.png` and the displayed sample to
`tactile_logs/contact_visualization.json`. The isolated dome pair is a clean
display proxy; forces are read from the two sensors attached to the robot.

Render all 104 CAD-aligned taxels on the complete gripper, colored by the
live per-taxel force magnitude:

```bash
conda run -n isaacsim python scripts/tactile/render_taxel_heatmap.py
```

This writes `renders/tactile_taxel_heatmap_gripper.png` and the exact displayed
`(2, 52, 3)` sample to `tactile_logs/taxel_heatmap_gripper.json`.

Generate a third-person grasp presentation with a minimal force-and-direction
overlay. Two axonometric coordinate frames sit beside the scene rather than on
the robot. Each shows the sensor-local X/Y/Z axes, its live force vector, and
the numeric `(Fx, Fy, Fz)` value:

```bash
SO101_TACTILE_VIDEO_GPU=1 \
conda run -n isaacsim python scripts/tactile/render_third_person_tactile_grasp.py

ffmpeg -y -i renders/tactile_third_person_grasp_raw.mp4 \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  renders/tactile_third_person_grasp.mp4
```

This writes the raw video, a poster frame, and
`tactile_logs/third_person_grasp.json`. The HUD displays live Isaac
ContactSensor output: two invisible 4 mm kinematic contact patches gently load
the CAD-aligned tactile collision shells through a compliant contact material,
and the corresponding PhysX contact impulses pass through the 83.3 Hz
DP-S2015 range, threshold, and 0.1 N/LSB quantization model. The approach/close/lift motion is scripted and the visible
ball follows the measured midpoint of the fingertip bodies during lift. It is
therefore a truthful contact-sensor visualization, but not evidence of a stable
free-body grasp or calibrated hardware force fidelity. Use `run_contact_mvp.py`
for the independent Isaac ContactSensor regression.

This is a contact-path regression, not a force-calibration test. Probe speed
and compliant stiffness are deterministic regression excitation parameters,
not estimates of real contact dynamics.

Real-data reference and current limitations:

- [SoTac](https://huggingface.co/datasets/Jingyi-Z/sotac) uses the same two
  DP-S2015-Elite pads and confirms the target `(2, 52, 3)` fixed/moving layout,
  Newton-scaled 0.1 N quantization, and raw approximately 91 Hz sidecars. Its
  SDK coordinate asset matches `array.xlsx` at all 52 indices exactly:
  dataset `p_00` maps to local taxel ID 1 through `p_51` to ID 52.
- SoTac provides measured distributed forces and a hardware resultant, but no
  paired ground-truth contact positions. It can calibrate noise, bias, temporal
  response, spatial covariance, clipping, and domain randomization; it cannot
  uniquely identify the spatial contact kernel by itself.
- The current spatial kernel is therefore a force-conserving baseline, not a
  claim that rubber deformation, hysteresis, drift, or crosstalk are calibrated.
- Runtime integrates every 240 Hz PhysX substep before emitting an 83.3 Hz
  product-specification sample-and-hold value. Switching the output clock to
  the measured SoTac approximately 91 Hz stream remains an explicit
  temporal-calibration choice rather than an assumed replacement.
- Analyze downloaded SoTac raw sidecars without modifying simulator parameters:

  ```bash
  python3 scripts/tactile/analyze_sotac.py \
    --sensor1 /path/to/sensor1.csv \
    --sensor2 /path/to/sensor2.csv \
    --sdk-coordinates /path/to/pxsr_stddp03g_points.json \
    --output tactile_logs/sotac_calibration_report.json
  ```

  The report measures raw rate/jitter, quantization, baseline per-taxel bias
  and noise, active-taxel statistics, timestamp alignment, SDK coordinate
  identity, and the difference between the independent hardware resultant and
  the 52-vector sum. Episode 0 confirms that those two force quantities are not
  interchangeable.
- The Isaac Lab wrapper is implemented in `so101_isaac_lab/`. It exposes
  aggregate `(N, 2, 3)`, taxels `(N, 2, 52, 3)`, per-taxel magnitudes
  `(N, 2, 52)`, and normalized policy features `(N, 312)` without changing
  numbering. See `so101_isaac_lab/README.md` for the manager-based environment
  contract and regression commands.
