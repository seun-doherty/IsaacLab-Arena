# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward functions for the gear mesh RL task.

The keypoint reward is adapted from the Factory/Forge environments in IsaacLab.
Reference: Appendix B of https://arxiv.org/pdf/2408.04587
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _squashing_fn(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Bounded reward: r(x) = 1 / (exp(a*x) + b + exp(-a*x)).

    Controls:
        a – slope (how quickly the reward changes with distance)
        b – maximum value offset (b=0 gives max=0.5, b=2 gives max=0.25, b=4 gives max≈0.17)
    """
    return 1.0 / (torch.exp(a * x) + b + torch.exp(-a * x))


def keypoint_reward_coarse(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("nist_gear_base"),
    num_keypoints: int = 4,
    keypoint_scale: float = 0.15,
    a: float = 50.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Keypoint-based reward measuring alignment between held gear and target on gear base.

    Places keypoints along the z-axis of both the held object and its target location
    on the gear base, then computes mean distance and passes it through a squashing function.
    The 'coarse' scale (a=50, b=2) is appropriate for guiding initial alignment.
    """
    return _keypoint_distance_reward(env, held_object_cfg, fixed_object_cfg, num_keypoints, keypoint_scale, a, b)


def keypoint_reward_fine(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("nist_gear_base"),
    num_keypoints: int = 4,
    keypoint_scale: float = 0.15,
    a: float = 100.0,
    b: float = 0.0,
) -> torch.Tensor:
    """Keypoint-based reward for fine insertion.

    The 'fine' scale (a=100, b=0) gives a sharper reward that helps with the
    last-millimetre precision needed for gear meshing.
    """
    return _keypoint_distance_reward(env, held_object_cfg, fixed_object_cfg, num_keypoints, keypoint_scale, a, b)


def gear_engaged(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("nist_gear_base"),
    gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    gear_peg_height: float = 0.02,
    engage_threshold: float = 0.9,
) -> torch.Tensor:
    """Binary bonus when the gear is engaged with the peg (within 90% of insertion).

    Matches Forge's curr_engaged reward term.
    """
    return _gear_insertion_check(
        env, held_object_cfg, fixed_object_cfg, gear_base_offset, gear_peg_height,
        xy_threshold=0.0025, z_fraction=engage_threshold,
    ).float()


def gear_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("nist_gear_base"),
    gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    gear_peg_height: float = 0.02,
    success_threshold: float = 0.05,
) -> torch.Tensor:
    """Binary bonus when the gear is fully inserted (within 5% of peg height).

    Matches Forge's curr_success reward term.
    """
    return _gear_insertion_check(
        env, held_object_cfg, fixed_object_cfg, gear_base_offset, gear_peg_height,
        xy_threshold=0.0025, z_fraction=success_threshold,
    ).float()


def _gear_insertion_check(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg,
    fixed_object_cfg: SceneEntityCfg,
    gear_base_offset: list[float],
    gear_peg_height: float,
    xy_threshold: float,
    z_fraction: float,
) -> torch.Tensor:
    """Check if gear is centered (XY) and inserted (Z) to the given fraction."""
    held_object: RigidObject = env.scene[held_object_cfg.name]
    fixed_object: RigidObject = env.scene[fixed_object_cfg.name]

    held_pos = held_object.data.root_pos_w - env.scene.env_origins
    fixed_pos = fixed_object.data.root_pos_w - env.scene.env_origins

    offset = torch.tensor(gear_base_offset, device=env.device)

    # Simple position-based check (ignoring rotation for speed)
    target_pos = fixed_pos.clone()
    target_pos[:, 0] += offset[0]

    xy_dist = torch.linalg.vector_norm(target_pos[:, 0:2] - held_pos[:, 0:2], dim=1)
    is_centered = xy_dist < xy_threshold

    z_disp = held_pos[:, 2] - target_pos[:, 2]
    height_threshold = gear_peg_height * z_fraction
    is_close = z_disp < height_threshold

    return torch.logical_and(is_centered, is_close)


def action_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize large actions to encourage smooth, efficient motions."""
    return torch.norm(env.action_manager.action, p=2, dim=-1)


def _keypoint_distance_reward(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg,
    fixed_object_cfg: SceneEntityCfg,
    num_keypoints: int,
    keypoint_scale: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Core keypoint reward computation.

    Keypoints are placed along the z-axis of the held and target objects.
    The reward is the squashing function applied to the mean keypoint distance.
    """
    held_object: RigidObject = env.scene[held_object_cfg.name]
    fixed_object: RigidObject = env.scene[fixed_object_cfg.name]

    held_pos = held_object.data.root_pos_w - env.scene.env_origins
    fixed_pos = fixed_object.data.root_pos_w - env.scene.env_origins

    # Compute keypoint offsets along z-axis, centered at 0, scaled
    offsets = torch.zeros((num_keypoints, 3), device=env.device)
    offsets[:, 2] = torch.linspace(0.0, 1.0, num_keypoints, device=env.device) - 0.5
    offsets = offsets * keypoint_scale  # (K, 3)

    # Expand for batched computation: (num_envs, K, 3)
    held_keypoints = held_pos.unsqueeze(1) + offsets.unsqueeze(0)
    fixed_keypoints = fixed_pos.unsqueeze(1) + offsets.unsqueeze(0)

    # Mean keypoint distance across all keypoints
    kp_dist = torch.norm(held_keypoints - fixed_keypoints, p=2, dim=-1).mean(dim=-1)  # (num_envs,)

    return _squashing_fn(kp_dist, a, b)
