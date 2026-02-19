# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST assembled board gear-mesh environment.

Uses a single composite USD (nist_assembled_wgears) that contains the board,
gear base, small gear, and large gear. The only separate dynamic asset is
the medium gear, which the robot must pick up and insert.
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class NISTAssembledGearMeshEnvironment(ExampleEnvironmentBase):
    """Gear insertion environment using the pre-assembled NIST board.

    The assembled board USD already has the gear base and flanking gears baked in.
    The robot's task is to pick up the medium gear and insert it onto the peg.
    """

    name: str = "nist_assembled_gear_mesh"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.nist_gear_insertion_task import NistGearInsertionTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        # Assets
        background = self.asset_registry.get_asset_by_name("table")()
        background2 = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        assembled_board = self.asset_registry.get_asset_by_name("nist_assembled_board")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()

        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        # Embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # Teleop
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Poses
        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        background2.set_initial_pose(Pose(position_xyz=(-3.98, 2.88, -1.01), rotation_wxyz=(0.707, 0, 0, 0.707)))

        assembled_board.set_initial_pose(
            Pose(
                position_xyz=(0.71, -0.005, 0.05),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        medium_gear.set_initial_pose(
            Pose(
                position_xyz=(0.37, 0.08, 0.05),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Scene
        scene = Scene(assets=[background, background2, assembled_board, medium_gear, light])

        # Task
        task = NistGearInsertionTask(
            assembled_board=assembled_board,
            held_gear=medium_gear,
            background_scene=background,
            peg_offset_from_board=[-0.12462, -0.04432, 0.05906],
            success_z_fraction=0.95,
            xy_threshold=0.015,
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
        """Add CLI arguments for the assembled gear mesh environment."""
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )
