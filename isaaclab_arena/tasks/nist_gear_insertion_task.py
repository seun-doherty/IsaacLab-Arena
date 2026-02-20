# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple gear insertion task for the assembled NIST board.

The assembled board USD contains the gear base, small gear, and large gear
as a single composite asset. Only the medium gear is a separate object that
the robot must pick up and insert onto the peg.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import gear_mesh_insertion_success
from isaaclab_arena.terms.events import place_gear_in_gripper
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class NistGearInsertionTask(TaskBase):
    """Gear insertion task using the assembled NIST board.

    The assembled board already contains the gear base and flanking gears.
    The task is simply: insert the medium gear onto the peg on the board.
    No randomization -- assets stay at their initial poses.
    """

    def __init__(
        self,
        assembled_board: Asset,
        held_gear: Asset,
        background_scene: Asset,
        peg_offset_from_board: list[float] | None = None,
        gear_peg_height: float = 0.02,
        success_z_fraction: float = 0.30,
        xy_threshold: float = 0.0025,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        start_in_gripper: bool = False,
        num_arm_joints: int = 7,
        hand_grasp_width: float = 0.03,
        hand_close_width: float = 0.0,
        gripper_joint_setter_func: Callable[
            [torch.Tensor, Sequence[int], Sequence[int], float], None
        ] | None = None,
        end_effector_body_name: str = "panda_hand",
        grasp_rot_offset: list[float] | None = None,
        grasp_offset: list[float] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.assembled_board = assembled_board
        self.held_gear = held_gear
        self.background_scene = background_scene
        self.peg_offset_from_board = peg_offset_from_board or [2.025e-2, 0.0, 0.0]
        self.gear_peg_height = gear_peg_height
        self.success_z_fraction = success_z_fraction
        self.xy_threshold = xy_threshold
        self.start_in_gripper = start_in_gripper
        self.num_arm_joints = num_arm_joints
        self.hand_grasp_width = hand_grasp_width
        self.hand_close_width = hand_close_width
        self.gripper_joint_setter_func = gripper_joint_setter_func
        self.end_effector_body_name = end_effector_body_name
        self.grasp_rot_offset = grasp_rot_offset or [1.0, 0.0, 0.0, 0.0]
        self.grasp_offset = grasp_offset or [0.0, 0.0, 0.0]
        self.task_description = (
            f"Insert the {held_gear.name} onto the gear base on the {assembled_board.name}"
            if task_description is None
            else task_description
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=gear_mesh_insertion_success,
            params={
                "held_object_cfg": SceneEntityCfg(self.held_gear.name),
                "fixed_object_cfg": SceneEntityCfg(self.assembled_board.name),
                "gear_base_offset": self.peg_offset_from_board,
                "gear_peg_height": self.gear_peg_height,
                "success_z_fraction": self.success_z_fraction,
                "xy_threshold": self.xy_threshold,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_gear.name),
            },
        )
        return _TerminationsCfg(success=success, object_dropped=object_dropped)

    def get_events_cfg(self):
        cfg = _EventsCfg()
        if self.start_in_gripper and self.gripper_joint_setter_func is not None:
            cfg.place_gear = EventTermCfg(
                func=place_gear_in_gripper,
                mode="reset",
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "num_arm_joints": self.num_arm_joints,
                    "hand_grasp_width": self.hand_grasp_width,
                    "hand_close_width": self.hand_close_width,
                    "gripper_joint_setter_func": self.gripper_joint_setter_func,
                    "end_effector_body_name": self.end_effector_body_name,
                    "grasp_rot_offset": self.grasp_rot_offset,
                    "grasp_offset": self.grasp_offset,
                },
            )
        return cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_gear,
            offset=np.array([1.5, -0.5, 1.0]),
        )


@configclass
class _TerminationsCfg:
    """Termination terms for the gear insertion task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class _EventsCfg:
    """Events: just reset everything to default poses."""

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    place_gear: EventTermCfg | None = None
