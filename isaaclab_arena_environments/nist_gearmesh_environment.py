# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class NISTGearMeshEnvironment(ExampleEnvironmentBase):
    """
    Gear mesh assembly environment with 4 gears:
    - gear_base: gear base (to be meshed with)
    - medium_gear: medium gear (to be picked and assembled)
    - small_gear: small reference gear
    - large_gear: large reference gear
    """

    name: str = "nist_gear_mesh"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.assembly_task import AssemblyTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        # Get assets from registry
        background = self.asset_registry.get_asset_by_name("table")()
        background2 = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        gear_base = self.asset_registry.get_asset_by_name("nist_gear_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
        small_gear = self.asset_registry.get_asset_by_name("small_nist_gear")()
        large_gear = self.asset_registry.get_asset_by_name("large_nist_gear")()
        board = self.asset_registry.get_asset_by_name("nist_board")()

        kit_tray = self.asset_registry.get_asset_by_name("kit_tray")()
        bnc_plug = self.asset_registry.get_asset_by_name("bnc_plug")()
        dsub_plug = self.asset_registry.get_asset_by_name("dsub_plug")()
        factory_peg_8mm = self.asset_registry.get_asset_by_name("factory_peg_8mm")()
        rj45_plug = self.asset_registry.get_asset_by_name("rj45_plug")()
        usb_a_plug = self.asset_registry.get_asset_by_name("usb_a_plug")()
        waterproof_plug = self.asset_registry.get_asset_by_name("waterproof_plug")()

        assembled_nist_board = self.asset_registry.get_asset_by_name("nist_assembled_board")()

        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        background2.set_initial_pose(Pose(position_xyz=(-3.98, 2.88, -1.01), rotation_wxyz=(0.707, 0, 0, 0.707)))
        # Set initial poses for all 4 gears
        gear_base.set_initial_pose(
            Pose(
                position_xyz=(0.40, -0.10, 0.02),  # Gear base position
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        medium_gear.set_initial_pose(
            Pose(
                position_xyz=(0.37, 0.08, 0.05),  # Medium gear — raised above table to avoid interpenetration at spawn
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        small_gear.set_initial_pose(
            Pose(
                position_xyz=(0.40, -0.10, 0.02),  # Small reference gear
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        large_gear.set_initial_pose(
            Pose(
                position_xyz=(0.40, -0.10, 0.02),  # Large reference gear
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        assembled_nist_board.set_initial_pose(
            Pose(
                position_xyz=(0.71, -0.005, 0.05),   # NIST board position
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Create scene with all 4 gears and background
        scene = Scene(assets=[background, background2, gear_base, medium_gear, small_gear, large_gear, assembled_nist_board, light])

        # Create gear mesh task
        task = AssemblyTask(
            fixed_asset=gear_base,
            held_asset=medium_gear,
            auxiliary_asset_list=[small_gear, large_gear],
            background_scene=background,
            pose_range={"x": (0.25, 0.35), "y": (-0.06, 0.06), "z": (0.07, 0.07), "yaw": (0.0, 0.0)},
            min_separation=0.15,
            randomization_mode="held_fixed_and_auxiliary",
        )

        # Override success termination: require 70% gear insertion instead of simple proximity
        from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
        from isaaclab_arena.tasks.terminations import gear_mesh_insertion_success

        task.termination_cfg.success = TerminationTermCfg(
            func=gear_mesh_insertion_success,
            params={
                "held_object_cfg": SceneEntityCfg(medium_gear.name),
                "fixed_object_cfg": SceneEntityCfg(gear_base.name),
                "success_z_fraction": 0.30,
            },
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
        """Add CLI arguments for gear mesh environment."""
        parser.add_argument("--background", type=str, default="table", help="Background scene (table)")
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )