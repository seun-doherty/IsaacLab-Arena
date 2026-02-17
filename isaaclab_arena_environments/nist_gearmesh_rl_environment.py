# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""RL-trainable gear mesh environment (Forge-style).

This is a separate environment class from NISTGearMeshEnvironment to avoid
modifying the existing IL/teleoperation workflow. It uses GearMeshTaskRL
with a full Forge-style reset: each episode the arm is IK-positioned above
the gear base, the gear is placed in the closed gripper, and the agent only
needs to learn the fine insertion motion.
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# Forge reset joints — starting pose for the IK servo loop.
# The forge_style_gear_reset event will IK from here to above the gear base.
# From Factory/Forge: CtrlCfg.reset_joints
FORGE_RESET_JOINTS = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]

# Gripper width for medium gear (diameter 0.03m).
# Forge: held_asset_cfg.diameter / 2 * 1.25 = 0.01875
GEAR_GRIP_WIDTH = 0.01875


class NISTGearMeshRLEnvironment(ExampleEnvironmentBase):
    """RL-trainable gear mesh assembly environment (Forge-style).

    Follows Forge's design: at each reset the arm is IK-servoed above the
    gear base, the held gear is placed in the gripper using Forge's
    transform chain, and the gripper closes around it. The agent only
    needs to learn the fine insertion motion.
    """

    name: str = "nist_gear_mesh_rl"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.gear_mesh_task import GearMeshTaskRL
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        # Get assets from registry
        background = self.asset_registry.get_asset_by_name("table")()
        gear_base = self.asset_registry.get_asset_by_name("nist_gear_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
        small_gear = self.asset_registry.get_asset_by_name("small_nist_gear")()
        large_gear = self.asset_registry.get_asset_by_name("large_nist_gear")()

        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        # Embodiment — concatenate observations for RL
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(concatenate_observation_terms=True)
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Forge-style initial arm pose (hand forward, ready for insertion)
        # and gripper closed to medium gear width.
        embodiment.set_initial_joint_pose(
            FORGE_RESET_JOINTS + [GEAR_GRIP_WIDTH, GEAR_GRIP_WIDTH]
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set initial poses
        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))

        # Gear base at table center (same as Forge: fixed asset at (0.6, 0, 0.05))
        # Arena's table surface is at z≈0.02 for the assembly config.
        gear_base.set_initial_pose(
            Pose(position_xyz=(0.6, 0.0, 0.02), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )
        # Medium gear initial pose — overridden by forge_style_gear_reset event
        # on each reset, but needs a valid default for scene construction.
        medium_gear.set_initial_pose(
            Pose(position_xyz=(0.62, 0.0, 0.07), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )
        # Flanking gears — will be placed at gear base position by the reset event.
        small_gear.set_initial_pose(
            Pose(position_xyz=(0.6, 0.0, 0.02), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )
        large_gear.set_initial_pose(
            Pose(position_xyz=(0.6, 0.0, 0.02), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # Scene
        scene = Scene(assets=[background, gear_base, medium_gear, small_gear, large_gear, light])

        # RL task — Forge-style: IK + gear in gripper + keypoint rewards
        task = GearMeshTaskRL(
            fixed_asset=gear_base,
            held_asset=medium_gear,
            auxiliary_asset_list=[small_gear, large_gear],
            background_scene=background,
            embodiment=embodiment,
            episode_length_s=20.0,
            rl_training_mode=args_cli.rl_training_mode,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for the RL gear mesh environment."""
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )
        parser.add_argument(
            "--rl_training_mode",
            type=bool,
            default=True,
            help="If True, disables success termination during training.",
        )
