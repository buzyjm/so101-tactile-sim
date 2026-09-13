"""Clone-safe scene for the DH116 gesture show."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from xarm5_dh116_lab.robot_cfg import XARM5_DH116_CFG

PEDESTAL_HEIGHT = 0.60
PEDESTAL_RADIUS = 0.16


@configclass
class GestureSceneCfg(InteractiveSceneCfg):
    """Ground, a pedestal per environment, the robot on top, one camera."""

    lazy_sensor_update = False

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Visual only: the robot is fixed-base, nothing can touch the pedestal.
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CylinderCfg(
            radius=PEDESTAL_RADIUS,
            height=PEDESTAL_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.22, 0.24, 0.27), roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, PEDESTAL_HEIGHT / 2.0)),
    )

    robot: ArticulationCfg = XARM5_DH116_CFG.replace(
        init_state=XARM5_DH116_CFG.init_state.replace(
            pos=(0.0, 0.0, PEDESTAL_HEIGHT)))

    # One per environment (Isaac Lab sizes the buffers by env count); all
    # are posed identically and only env 0 is read.
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.5),
        init_state=AssetBaseCfg.InitialStateCfg(
            # tilted so the light comes from the camera side and above
            rot=(0.9239, 0.0, 0.3827, 0.0)),
    )
