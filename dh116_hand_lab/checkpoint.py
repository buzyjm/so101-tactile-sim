"""Restore the configuration saved with a policy instead of today's defaults."""
from pathlib import Path


def load_checkpoint_configs(checkpoint):
    import yaml
    from dh116_hand_lab.pen_spin_env_cfg import DH116PenSpinEnvCfg

    folder = Path(checkpoint).resolve().parent
    env_cfg = DH116PenSpinEnvCfg()
    saved = yaml.full_load((folder / "env_cfg.yaml").read_text())
    # Configclass updates mappings strictly by key.  Older checkpoints used
    # one catch-all joint initializer and one actuator group, while the
    # calibrated model splits active and follower drives.  Shape these two
    # mappings like the saved config first; ``collision_profile=current``
    # replaces them with today's model after the historical config is loaded.
    saved_robot = saved["robot_cfg"]
    env_cfg.robot_cfg.init_state.joint_pos = {
        key: 0.0 for key in saved_robot["init_state"]["joint_pos"]
    }
    actuator_template = next(iter(env_cfg.robot_cfg.actuators.values()))
    env_cfg.robot_cfg.actuators = {
        key: actuator_template.copy() for key in saved_robot["actuators"]
    }
    # Isaac Lab's strict update rejects int -> default None for optional seed.
    env_cfg.seed = saved["seed"]
    env_cfg.from_dict(saved)
    agent_cfg = yaml.full_load((folder / "agent_cfg.yaml").read_text())
    return env_cfg, clean_agent_cfg(agent_cfg)


def clean_agent_cfg(cfg):
    for key in ("actor", "critic"):
        for name in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            cfg[key].pop(name, None)
    return cfg
