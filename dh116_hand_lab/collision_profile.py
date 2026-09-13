"""Apply the verified self-collision profile explicitly when migrating a policy."""

def apply_collision_profile(env_cfg, profile):
    if profile == "current":
        from dh116_hand_lab.hand_cfg import DH116_HAND_CFG, HAND_USD
        if not HAND_USD.is_file():
            raise FileNotFoundError(f"Build corrected collision assets first: {HAND_USD}")
        spawn = env_cfg.robot_cfg.spawn
        spawn.usd_path = str(HAND_USD)
        spawn.rigid_props.max_depenetration_velocity = 1.0
        spawn.articulation_props.enabled_self_collisions = True
        spawn.articulation_props.solver_position_iteration_count = 32
        spawn.articulation_props.solver_velocity_iteration_count = 8
        env_cfg.sim.physics.solve_articulation_contact_last = True
        # A historical checkpoint may contain the former one-actuator config.
        # Restore current active/coupling drive groups together with the asset.
        env_cfg.robot_cfg.actuators = {
            name: actuator.copy() for name, actuator in DH116_HAND_CFG.actuators.items()}
        env_cfg.robot_cfg.init_state.joint_pos = dict(DH116_HAND_CFG.init_state.joint_pos)
        env_cfg.robot_cfg.soft_joint_pos_limit_factor = 1.0
    elif profile != "saved":
        raise ValueError(f"Unknown collision profile: {profile}")
    enabled = env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions
    print(f"Collision profile: {profile}; self-collisions={enabled}; asset={env_cfg.robot_cfg.spawn.usd_path}", flush=True)
    if not enabled:
        print("Historical physics: self-collisions are disabled. Use --collision_profile current for the corrected model.", flush=True)
