# Leadshine DH116 hand-only asset

This asset is cut from the merged xArm5 + DH116 URDF by
`scripts/dh116_hand/build_hand_urdf.py`.

- `dh116_hand.urdf`: current generated model: 11 joints, 6 commands, Leadshine
  2026-04-09 nonlinear follower ranges and 0.545 kg nominal total mass.
- `usd_manual_2026_04/`: Isaac Sim 6.0 import, retained as the calibrated
  no-self-collision source.
- `usd_manual_2026_04_self_collision/`: runtime/training asset with 1 mm
  contact offset, zero rest offset and 32/8 solver iterations.
- `usd/` and `usd_self_collision/`: historical pre-manual-calibration assets,
  retained so old results remain reproducible.

The vendor gives only total hand mass. The generator preserves moving-link
inertias and assigns the unmodelled motor/electronics mass to the base link.
PhysX cannot represent the published quadratic linkage with a native mimic
joint, so runtime code computes follower targets and applies them through a
compliant drive.

See `reports/dh116_manual_calibration_2026-09-10.md` for provenance, equations,
validation results and remaining physical calibration gaps.
