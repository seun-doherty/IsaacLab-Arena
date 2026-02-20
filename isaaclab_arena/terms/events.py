# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_tasks.direct.automate import factory_control as fc

from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class place_gear_in_gripper(ManagerTermBase):
    """IK the robot to the gear's position, then close the gripper.

    Replicates the deploy gear-assembly ``set_robot_to_grasp_pose`` pattern:
      * Single EE body Jacobian with full DOF solving.
      * Explicit ``grasp_rot_offset`` quaternion applied to the gear
        orientation to get the desired EE orientation.
      * Explicit ``grasp_offset`` position offset (in the rotated gear frame)
        from the gear centre to the desired EE body origin.
      * Two-step gripper close (physical grasp width then PD close width).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[robot_cfg.name]

        gear_cfg: SceneEntityCfg = cfg.params["gear_cfg"]
        self.gear = env.scene[gear_cfg.name]

        self.num_arm_joints: int = cast(int, cfg.params["num_arm_joints"])
        self.hand_grasp_width: float = cast(float, cfg.params["hand_grasp_width"])
        self.hand_close_width: float = cast(float, cfg.params["hand_close_width"])
        self.gripper_joint_setter_func: Callable[..., Any] = cast(
            Callable[..., Any], cfg.params["gripper_joint_setter_func"],
        )

        # Grasp rotation offset (quaternion [w, x, y, z]) applied to gear quat
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
        )

        # Grasp position offset in the rotated gear frame [x, y, z]
        grasp_offset = cfg.params["grasp_offset"]
        self.grasp_offset_tensor = torch.tensor(
            grasp_offset, device=env.device, dtype=torch.float32
        )

        # -- EE body index --
        ee_name: str = cast(str, cfg.params.get("end_effector_body_name", "panda_hand"))
        eef_indices, _ = self.robot.find_bodies([ee_name])
        if not eef_indices:
            raise ValueError(f"End-effector body '{ee_name}' not found in robot")
        self.eef_idx: int = eef_indices[0]
        self.jacobi_body_idx: int = self.eef_idx - 1

        # -- joints --
        all_joints, _ = self.robot.find_joints([".*"])
        self.all_joints = all_joints
        self.finger_joints = all_joints[self.num_arm_joints:]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg | None = None,
        gear_cfg: SceneEntityCfg | None = None,
        num_arm_joints: int | None = None,
        hand_grasp_width: float | None = None,
        hand_close_width: float | None = None,
        gripper_joint_setter_func: Callable | None = None,
        end_effector_body_name: str | None = None,
        grasp_rot_offset: list | None = None,
        grasp_offset: list | None = None,
        max_ik_iterations: int = 10,
        pos_threshold: float = 1e-6,
        rot_threshold: float = 1e-6,
    ) -> None:
        n = len(env_ids)
        device = env.device

        grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]
        grasp_offset_batch = self.grasp_offset_tensor.unsqueeze(0).expand(n, -1)

        for _ in range(max_ik_iterations):
            joint_pos = self.robot.data.joint_pos[env_ids].clone()
            joint_vel = self.robot.data.joint_vel[env_ids].clone()

            # Target pose: gear position + offsets
            gear_pos_w = self.gear.data.root_link_pos_w[env_ids].clone()
            gear_quat_w = self.gear.data.root_link_quat_w[env_ids].clone()

            target_quat = math_utils.quat_mul(gear_quat_w, grasp_rot_offset_tensor)
            target_pos = gear_pos_w + math_utils.quat_apply(
                target_quat, grasp_offset_batch
            )

            # Current EE pose
            eef_pos = self.robot.data.body_pos_w[env_ids, self.eef_idx]
            eef_quat = self.robot.data.body_quat_w[env_ids, self.eef_idx]

            pos_error, aa_error = fc.get_pose_error(
                fingertip_midpoint_pos=eef_pos,
                fingertip_midpoint_quat=eef_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            delta_hand_pose = torch.cat((pos_error, aa_error), dim=-1)

            if (torch.norm(pos_error, dim=-1).max() < pos_threshold
                    and torch.norm(aa_error, dim=-1).max() < rot_threshold):
                break

            # Full Jacobian for the EE body (all DOFs)
            jacobians = self.robot.root_physx_view.get_jacobians().clone()
            jacobian = jacobians[env_ids, self.jacobi_body_idx, :, :]

            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=jacobian,
                device=device,
            )

            joint_pos = joint_pos + delta_dof_pos
            joint_vel = torch.zeros_like(joint_pos)

            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self.robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # --- Gripper close (two-step, matching deploy env) ---
        joint_pos = self.robot.data.joint_pos[env_ids].clone()

        # Step 1: physical state at grasp width
        for row_idx in range(n):
            self.gripper_joint_setter_func(
                joint_pos, [row_idx], self.finger_joints, self.hand_grasp_width,
            )

        self.robot.set_joint_position_target(
            joint_pos, joint_ids=self.all_joints, env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Step 2: PD target at close width for grip force
        for row_idx in range(n):
            self.gripper_joint_setter_func(
                joint_pos, [row_idx], self.finger_joints, self.hand_close_width,
            )

        self.robot.set_joint_position_target(
            joint_pos, joint_ids=self.all_joints, env_ids=env_ids,
        )


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame
    pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_wxyz[:, :3] += env.scene.env_origins[env_ids]
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    # Set the objects pose in each environment independently
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        # Convert the pose to the env frame
        pose = pose_list[cur_env]
        pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device)
        pose_t_xyz_q_wxyz[:3] += env.scene.env_origins[cur_env, :].squeeze()
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=torch.tensor([cur_env], device=env.device))
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )
