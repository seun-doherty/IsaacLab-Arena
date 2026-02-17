# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gear mesh RL task for Isaac Lab Arena.

Extends the existing AssemblyTask with RL-specific rewards, observations,
and termination config so that an RL agent can learn gear meshing.

Follows Forge's design: at each reset the arm is IK-servoed above the gear
base, the gear is placed in the closed gripper, and the agent only needs to
learn the fine insertion motion.
"""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

import isaaclab_arena_environments.mdp as mdp
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.tasks.assembly_task import AssemblyTask
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.rewards import gear_mesh_rewards


class GearMeshTaskRL(AssemblyTask):
    """RL-trainable gear mesh task (Forge-style).

    Each episode the arm is IK-positioned above the gear base, the gear
    is placed in the closed gripper, and the agent learns the fine
    insertion motion. Keypoint-based rewards from Factory/Forge guide
    the learning.
    """

    # Gear geometry constants from Factory/Forge GearBase config.
    MEDIUM_GEAR_BASE_OFFSET: list[float] = [2.025e-2, 0.0, 0.0]
    GEAR_PEG_HEIGHT: float = 0.02  # GearBase.height
    GEAR_BASE_HEIGHT: float = 0.005  # GearBase.base_height
    SUCCESS_XY_THRESHOLD: float = 0.0025  # 2.5 mm radial centering
    SUCCESS_Z_FRACTION: float = 0.05  # gear must be within 5% of peg height

    # Factory MediumGear dimensions (used for gripper-relative placement).
    GEAR_HEIGHT: float = 0.03  # MediumGear.height
    GEAR_DIAMETER: float = 0.03  # MediumGear.diameter

    def __init__(
        self,
        fixed_asset: Asset,
        held_asset: Asset,
        auxiliary_asset_list: list[Asset],
        background_scene: Asset,
        embodiment: EmbodimentBase,
        episode_length_s: float = 20.0,
        max_x_separation: float = 0.020,
        max_y_separation: float = 0.020,
        max_z_separation: float = 0.020,
        task_description: str | None = None,
        rl_training_mode: bool = True,
    ):
        self.embodiment = embodiment
        self.rl_training_mode = rl_training_mode

        super().__init__(
            fixed_asset=fixed_asset,
            held_asset=held_asset,
            auxiliary_asset_list=auxiliary_asset_list,
            background_scene=background_scene,
            episode_length_s=episode_length_s,
            max_x_separation=max_x_separation,
            max_y_separation=max_y_separation,
            max_z_separation=max_z_separation,
            task_description=task_description,
            pose_range={},
            min_separation=0.0,
            randomization_mode="held_and_fixed_only",
        )

        # Build RL-specific configs
        robot_name = self.embodiment.get_embodiment_name_in_scene()

        self._observation_cfg = GearMeshObservationsCfg(
            held_asset=self.held_asset,
            fixed_asset=self.fixed_asset,
            robot_name=robot_name,
        )
        self._rewards_cfg = GearMeshRewardCfg(
            held_asset=self.held_asset,
            fixed_asset=self.fixed_asset,
        )

        # Override termination to support RL training mode (Forge-style success)
        self.termination_cfg = self._make_rl_termination_cfg()

        # Override events: replace assembly randomization with Forge-style reset
        self.events_cfg = self._make_forge_events_cfg()

    def _make_forge_events_cfg(self):
        """Create Forge-style events: reset to default, then IK + gear in gripper."""
        robot_name = self.embodiment.get_embodiment_name_in_scene()
        return GearMeshEventsCfg(
            robot_cfg=SceneEntityCfg(robot_name),
            held_asset_cfg=SceneEntityCfg(self.held_asset.name),
            fixed_asset_cfg=SceneEntityCfg(self.fixed_asset.name),
            auxiliary_asset_cfgs=[SceneEntityCfg(a.name) for a in self.auxiliary_asset_list],
            gear_base_offset=self.MEDIUM_GEAR_BASE_OFFSET,
            peg_height=self.GEAR_PEG_HEIGHT,
            base_height=self.GEAR_BASE_HEIGHT,
        )

    def _make_rl_termination_cfg(self):
        """Create RL-aware termination config using Forge-style geometric success."""
        success = TerminationTermCfg(
            func=gear_mesh_rl_success,
            params={
                "held_object_cfg": SceneEntityCfg(self.held_asset.name),
                "fixed_object_cfg": SceneEntityCfg(self.fixed_asset.name),
                "gear_base_offset": self.MEDIUM_GEAR_BASE_OFFSET,
                "gear_peg_height": self.GEAR_PEG_HEIGHT,
                "xy_threshold": self.SUCCESS_XY_THRESHOLD,
                "z_success_fraction": self.SUCCESS_Z_FRACTION,
                "rl_training": self.rl_training_mode,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_asset.name),
            },
        )
        return GearMeshTerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_observation_cfg(self):
        return self._observation_cfg

    def get_rewards_cfg(self):
        return self._rewards_cfg


# ---------------------------------------------------------------------------
# Events (Forge-style reset)
# ---------------------------------------------------------------------------

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_conjugate,
    quat_mul,
)


def _step_sim_no_action(env: ManagerBasedEnv) -> None:
    """Step the simulation without processing actions. Used during reset only.

    Mirrors Forge's ``step_sim_no_action``: write buffered data to the
    simulation, advance physics by one step, and read the new state back.
    """
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.sim.get_physics_dt())


def forge_style_gear_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("nist_gear_base"),
    auxiliary_asset_cfgs: list[SceneEntityCfg] | None = None,
    gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    peg_height: float = 0.02,
    base_height: float = 0.005,
    hand_init_pos_z: float = 0.035,
    gear_height: float = 0.03,
    ee_body_name: str = "panda_hand",
    ee_offset: list[float] = [0.0, 0.0, 0.107],
    ik_duration: float = 0.5,
    grasp_duration: float = 0.25,
) -> None:
    """Forge-style reset: IK arm above peg, place gear in gripper, close gripper.

    Replicates Factory/Forge ``randomize_initial_state`` for gear_mesh:

    1. Servo the arm via DLS IK so the EE is above the target peg.
    2. Place the held gear relative to the EE using Forge's z-flip +
       inverse-transform chain.
    3. Close the gripper around the gear.

    This function steps the simulation during reset (same as Forge).
    It runs AFTER ``reset_scene_to_default``.

    Args:
        robot_cfg: Robot articulation scene entity.
        held_object_cfg: Held gear scene entity.
        fixed_object_cfg: Gear base (fixed asset) scene entity.
        auxiliary_asset_cfgs: Flanking gear scene entities.
        gear_base_offset: Offset of the medium peg on the gear base [m].
        peg_height: Height of the peg on the gear base [m].
        base_height: Height of the gear base platform [m].
        hand_init_pos_z: Height above peg tip to position the hand [m].
        gear_height: Height of the held gear [m] (Factory MediumGear: 0.03).
        ee_body_name: Body used for IK (typically ``panda_hand``).
        ee_offset: Offset from *ee_body* to the EE point (fingertip centre).
        ik_duration: Duration of the IK servo loop [s].
        grasp_duration: Duration of the gripper-closing phase [s].
    """
    if env_ids is None:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    fixed: RigidObject = env.scene[fixed_object_cfg.name]
    held: RigidObject = env.scene[held_object_cfg.name]

    device = env.device
    n = env.num_envs
    dt = env.sim.get_physics_dt()
    n_arm_dofs = 7

    ee_offset_t = torch.tensor(ee_offset, device=device)
    ee_body_idx = robot.find_bodies(ee_body_name)[0][0]

    # ------------------------------------------------------------------
    # 1. Step sim to get FK from the reset joints
    # ------------------------------------------------------------------
    _step_sim_no_action(env)

    # ------------------------------------------------------------------
    # 2. Compute target EE position (above the peg tip on the gear base)
    # ------------------------------------------------------------------
    fixed_pos_w = fixed.data.root_pos_w
    fixed_quat_w = fixed.data.root_quat_w

    peg_tip_local = torch.zeros((n, 3), device=device)
    peg_tip_local[:, 0] = gear_base_offset[0]
    peg_tip_local[:, 2] = base_height + peg_height
    peg_tip_w = fixed_pos_w + _quat_rotate(fixed_quat_w, peg_tip_local)

    target_ee_pos_w = peg_tip_w.clone()
    target_ee_pos_w[:, 2] += hand_init_pos_z

    # Pointing straight down: roll = pi  ->  quat (wxyz) = [0, 1, 0, 0]
    target_ee_quat_w = torch.zeros((n, 4), device=device)
    target_ee_quat_w[:, 1] = 1.0

    # ------------------------------------------------------------------
    # 3. IK servo loop (DLS, mirrors Forge's set_pos_inverse_kinematics)
    # ------------------------------------------------------------------
    ik_time = 0.0
    while ik_time < ik_duration:
        hand_pos_w = robot.data.body_pos_w[:, ee_body_idx]
        hand_quat_w = robot.data.body_quat_w[:, ee_body_idx]
        offset_w = _quat_rotate(hand_quat_w, ee_offset_t.unsqueeze(0).expand(n, -1))
        ee_pos_w = hand_pos_w + offset_w
        ee_quat_w = hand_quat_w

        # Pose error (world frame)
        pos_error = target_ee_pos_w - ee_pos_w
        q_err = quat_mul(target_ee_quat_w, quat_conjugate(ee_quat_w))
        q_err = torch.where(q_err[:, 0:1] < 0, -q_err, q_err)
        rot_error = axis_angle_from_quat(q_err)
        delta_pose = torch.cat([pos_error, rot_error], dim=-1)

        # Jacobian for ee_body, with offset correction
        jacobians = robot.root_physx_view.get_jacobians()
        J = jacobians[:, ee_body_idx - 1, :, :n_arm_dofs].clone()
        J[:, 0:3] += torch.cross(
            offset_w.unsqueeze(-1).expand(-1, -1, n_arm_dofs),
            J[:, 3:],
            dim=1,
        )

        # DLS solve
        lam = 0.05
        JJT = J @ J.transpose(1, 2)
        damped = JJT + lam**2 * torch.eye(6, device=device)
        delta_q = J.transpose(1, 2) @ torch.linalg.solve(
            damped, delta_pose.unsqueeze(-1)
        )
        delta_q = delta_q.squeeze(-1)

        joint_pos = robot.data.joint_pos.clone()
        joint_pos[:, :n_arm_dofs] += delta_q
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)

        _step_sim_no_action(env)
        ik_time += dt

    # ------------------------------------------------------------------
    # 4. Place flanking gears at the gear base (before held gear so the
    #    arm doesn't collide with them)
    # ------------------------------------------------------------------
    if auxiliary_asset_cfgs:
        for aux_cfg in auxiliary_asset_cfgs:
            aux: RigidObject = env.scene[aux_cfg.name]
            aux_state = aux.data.default_root_state.clone()
            aux_state[:, 0:3] = fixed_pos_w
            aux_state[:, 3:7] = fixed_quat_w
            aux_state[:, 7:] = 0.0
            aux.write_root_pose_to_sim(aux_state[:, 0:7])
            aux.write_root_velocity_to_sim(aux_state[:, 7:])

    # ------------------------------------------------------------------
    # 5. Place gear in gripper (Forge transform chain)
    # ------------------------------------------------------------------
    _step_sim_no_action(env)
    hand_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    hand_quat_w = robot.data.body_quat_w[:, ee_body_idx]
    offset_w = _quat_rotate(hand_quat_w, ee_offset_t.unsqueeze(0).expand(n, -1))
    ee_pos_w = hand_pos_w + offset_w
    ee_quat_w = hand_quat_w

    # Flip z orientation of the EE frame (Forge convention)
    flip_z = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).expand(n, -1)
    flipped_pos, flipped_quat = combine_frame_transforms(
        ee_pos_w, ee_quat_w,
        torch.zeros((n, 3), device=device), flip_z,
    )

    # Gear-in-fingertip offset (Forge get_handheld_asset_relative_pose)
    held_rel_pos = torch.zeros((n, 3), device=device)
    held_rel_pos[:, 0] = gear_base_offset[0]
    held_rel_pos[:, 2] = gear_height / 2.0 * 1.1
    held_rel_quat = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=device
    ).unsqueeze(0).expand(n, -1)

    inv_quat = quat_conjugate(held_rel_quat)
    inv_pos = _quat_rotate(inv_quat, -held_rel_pos)

    gear_pos_w, gear_quat_w = combine_frame_transforms(
        flipped_pos, flipped_quat,
        inv_pos, inv_quat,
    )

    gear_state = held.data.default_root_state.clone()
    gear_state[:, 0:3] = gear_pos_w
    gear_state[:, 3:7] = gear_quat_w
    gear_state[:, 7:] = 0.0
    held.write_root_pose_to_sim(gear_state[:, 0:7])
    held.write_root_velocity_to_sim(gear_state[:, 7:])

    _step_sim_no_action(env)

    # ------------------------------------------------------------------
    # 6. Close gripper around the gear
    # ------------------------------------------------------------------
    grasp_time = 0.0
    while grasp_time < grasp_duration:
        joint_target = robot.data.joint_pos.clone()
        joint_target[:, n_arm_dofs:] = 0.0
        robot.set_joint_position_target(joint_target)
        _step_sim_no_action(env)
        grasp_time += dt


@configclass
class GearMeshEventsCfg:
    """Events for the gear mesh RL task (Forge-style reset).

    1. reset_all: resets robot and objects to default poses.
    2. place_gear_in_gripper: IK arm above peg, place gear in gripper,
       close gripper (steps the simulation, same as Forge).
    """

    reset_all: EventTermCfg = MISSING
    place_gear_in_gripper: EventTermCfg = MISSING

    def __init__(
        self,
        robot_cfg: SceneEntityCfg,
        held_asset_cfg: SceneEntityCfg,
        fixed_asset_cfg: SceneEntityCfg,
        auxiliary_asset_cfgs: list[SceneEntityCfg],
        gear_base_offset: list[float],
        peg_height: float,
        base_height: float,
    ):
        self.reset_all = EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )
        self.place_gear_in_gripper = EventTermCfg(
            func=forge_style_gear_reset,
            mode="reset",
            params={
                "robot_cfg": robot_cfg,
                "held_object_cfg": held_asset_cfg,
                "fixed_object_cfg": fixed_asset_cfg,
                "auxiliary_asset_cfgs": auxiliary_asset_cfgs,
                "gear_base_offset": gear_base_offset,
                "peg_height": peg_height,
                "base_height": base_height,
            },
        )


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


def gear_mesh_rl_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg,
    fixed_object_cfg: SceneEntityCfg,
    gear_base_offset: list[float],
    gear_peg_height: float,
    xy_threshold: float = 0.0025,
    z_success_fraction: float = 0.05,
    rl_training: bool = False,
) -> torch.Tensor:
    """Forge-style geometric success check for gear meshing.

    Adapted from Factory/Forge _get_curr_successes(). Checks:
      1. XY centering — radial distance between held gear base and
         target peg location must be < xy_threshold (default 2.5 mm).
      2. Z insertion — held gear base must be within
         gear_peg_height * z_success_fraction above the target.

    During RL training, always returns False to prevent early termination.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    held_object: RigidObject = env.scene[held_object_cfg.name]
    fixed_object: RigidObject = env.scene[fixed_object_cfg.name]

    held_pos = held_object.data.root_pos_w - env.scene.env_origins
    held_quat = held_object.data.root_quat_w
    fixed_pos = fixed_object.data.root_pos_w - env.scene.env_origins
    fixed_quat = fixed_object.data.root_quat_w

    offset = torch.tensor(gear_base_offset, device=env.device)

    held_base_pos = held_pos + _quat_rotate(held_quat, offset.unsqueeze(0).expand(env.num_envs, -1))
    target_pos = fixed_pos + _quat_rotate(fixed_quat, offset.unsqueeze(0).expand(env.num_envs, -1))

    xy_dist = torch.linalg.vector_norm(target_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
    is_centered = xy_dist < xy_threshold

    z_disp = held_base_pos[:, 2] - target_pos[:, 2]
    height_threshold = gear_peg_height * z_success_fraction
    is_inserted = z_disp < height_threshold

    return torch.logical_and(is_centered, is_inserted)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (wxyz convention)."""
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


@configclass
class GearMeshTerminationsCfg:
    """Termination terms for the gear mesh RL task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class GearMeshObservationsCfg:
    """Observations for the gear mesh RL task.

    Provides the robot with the positions of the held gear and the
    fixed gear base, both expressed in the robot's root frame.
    """

    task_obs: ObsGroup = MISSING

    def __init__(self, held_asset: Asset, fixed_asset: Asset, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            held_object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(held_asset.name),
                },
            )
            fixed_object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(fixed_asset.name),
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class GearMeshRewardCfg:
    """Reward terms for the gear mesh RL task (Forge-style).

    Since the gear starts above the target peg, no reaching or carrying
    rewards are needed. Uses Forge's multi-scale keypoint rewards plus
    success and engagement bonuses.
    """

    keypoint_baseline: RewardTermCfg = MISSING
    keypoint_coarse: RewardTermCfg = MISSING
    keypoint_fine: RewardTermCfg = MISSING
    engaged_bonus: RewardTermCfg = MISSING
    success_bonus: RewardTermCfg = MISSING
    action_penalty: RewardTermCfg = MISSING

    def __init__(self, held_asset: Asset, fixed_asset: Asset):
        self.keypoint_baseline = RewardTermCfg(
            func=gear_mesh_rewards.keypoint_reward_coarse,
            params={
                "held_object_cfg": SceneEntityCfg(held_asset.name),
                "fixed_object_cfg": SceneEntityCfg(fixed_asset.name),
                "num_keypoints": 4,
                "keypoint_scale": 0.15,
                "a": 5.0,
                "b": 4.0,
            },
            weight=1.0,
        )
        self.keypoint_coarse = RewardTermCfg(
            func=gear_mesh_rewards.keypoint_reward_coarse,
            params={
                "held_object_cfg": SceneEntityCfg(held_asset.name),
                "fixed_object_cfg": SceneEntityCfg(fixed_asset.name),
                "num_keypoints": 4,
                "keypoint_scale": 0.15,
                "a": 50.0,
                "b": 2.0,
            },
            weight=1.0,
        )
        self.keypoint_fine = RewardTermCfg(
            func=gear_mesh_rewards.keypoint_reward_fine,
            params={
                "held_object_cfg": SceneEntityCfg(held_asset.name),
                "fixed_object_cfg": SceneEntityCfg(fixed_asset.name),
                "num_keypoints": 4,
                "keypoint_scale": 0.15,
                "a": 100.0,
                "b": 0.0,
            },
            weight=1.0,
        )
        self.engaged_bonus = RewardTermCfg(
            func=gear_mesh_rewards.gear_engaged,
            params={
                "held_object_cfg": SceneEntityCfg(held_asset.name),
                "fixed_object_cfg": SceneEntityCfg(fixed_asset.name),
            },
            weight=1.0,
        )
        self.success_bonus = RewardTermCfg(
            func=gear_mesh_rewards.gear_success,
            params={
                "held_object_cfg": SceneEntityCfg(held_asset.name),
                "fixed_object_cfg": SceneEntityCfg(fixed_asset.name),
            },
            weight=1.0,
        )
        self.action_penalty = RewardTermCfg(
            func=gear_mesh_rewards.action_penalty,
            params={},
            weight=-0.01,
        )
