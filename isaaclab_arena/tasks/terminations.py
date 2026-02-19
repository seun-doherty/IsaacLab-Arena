# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.terminations import root_height_below_minimum
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor


# NOTE(alexmillane, 2025.09.15): The velocity threshold is set high because some stationary
# seem to generate a "small" velocity.
def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    sensor: ContactSensor = env.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(sensor.data.force_matrix_w.clone(), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = object.data.root_lin_vel_w
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)
    return condition_met


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Multi-object version of `object_on_destination`.

    Returns True only when ALL objects in the list satisfy the destination condition.
    See `object_on_destination` for details on the single-object logic.
    """
    condition_met = torch.ones((env.num_envs), device=env.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg in zip(object_cfg_list, contact_sensor_cfg_list):
        single_condition = object_on_destination(
            env=env,
            object_cfg=object_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met


def objects_in_proximity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns:
        Boolean tensor indicating when objects are within a certain proximity of each other.
    """
    # Get object entities from the scene
    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins

    # Get positions relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins
    target_object_pos = target_object.data.root_pos_w - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def gear_mesh_insertion_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    fixed_object_cfg: SceneEntityCfg = SceneEntityCfg("nist_gear_base"),
    gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
    gear_peg_height: float = 0.02,
    success_z_fraction: float = 0.80,
    xy_threshold: float = 0.0025,
) -> torch.Tensor:
    """Terminate when the gear is inserted onto the peg to the required depth.

    Checks that the held gear is centered on the peg (XY) and lowered past
    a fraction of the peg height (Z). A success_z_fraction of 0.30 means
    the gear must be 70% inserted.

    Args:
        held_object_cfg: Scene entity for the held gear.
        fixed_object_cfg: Scene entity for the gear base.
        gear_base_offset: XYZ offset from gear base origin to the target peg center.
        gear_peg_height: Height of the peg in meters.
        success_z_fraction: Remaining fraction of peg height that counts as success
            (0.30 = 70% inserted, 0.05 = 95% inserted).
        xy_threshold: Maximum radial distance from peg center in meters.
    """
    held_object: RigidObject = env.scene[held_object_cfg.name]
    fixed_object: RigidObject = env.scene[fixed_object_cfg.name]

    held_pos = held_object.data.root_pos_w - env.scene.env_origins
    fixed_pos = fixed_object.data.root_pos_w - env.scene.env_origins

    offset = torch.tensor(gear_base_offset, device=env.device)
    target_pos = fixed_pos.clone()
    target_pos += offset

    xy_dist = torch.linalg.vector_norm(target_pos[:, 0:2] - held_pos[:, 0:2], dim=1)
    is_centered = xy_dist < xy_threshold

    z_disp = held_pos[:, 2] - target_pos[:, 2]
    height_threshold = gear_peg_height * success_z_fraction
    is_inserted = z_disp < height_threshold

    return torch.logical_and(is_centered, is_inserted)


def object_at_fixed_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    target_position: list[float] = [0.0, 0.0, 0.0],
    tolerance: float = 0.005,
) -> torch.Tensor:
    """Terminate when the object is within a tolerance of a fixed world-space position.

    Args:
        object_cfg: Scene entity for the object to check.
        target_position: Fixed XYZ target position in the environment frame.
        tolerance: Maximum 3D distance in meters for success.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    target = torch.tensor(target_position, device=env.device).unsqueeze(0)
    distance = torch.norm(obj_pos - target, dim=-1)
    return distance < tolerance


def gear_contact_and_z_success(
    env: ManagerBasedRLEnv,
    held_object_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    board_cfg: SceneEntityCfg = SceneEntityCfg("nist_assembled_board"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("gear_contact_sensor"),
    peg_offset_from_board: list[float] = [0.0, 0.0, 0.0],
    z_threshold: float = 0.01,
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate when the gear contacts the board AND is inserted to sufficient depth.

    Two conditions must be met simultaneously:
      1. Contact: the contact sensor between the gear and board reports force
         above ``force_threshold`` while the gear velocity is below
         ``velocity_threshold`` (i.e. the gear has settled).
      2. Z-depth: the gear's Z position is below the peg top, computed as
         the board's *live* position + ``peg_offset_from_board``, plus a
         ``z_threshold`` margin. This keeps the check relative to the board
         so it stays valid even if the board shifts.

    Args:
        held_object_cfg: Scene entity for the held gear.
        board_cfg: Scene entity for the assembled board.
        contact_sensor_cfg: Scene entity for the contact sensor on the gear.
        peg_offset_from_board: XYZ offset from the board origin to the peg top.
        z_threshold: The gear must be within this distance above the peg-top Z
            to count as inserted (meters). A small positive value is forgiving.
        force_threshold: Minimum contact force (N) to consider contact established.
        velocity_threshold: Maximum gear velocity (m/s) to consider it settled.
    """
    held_object: RigidObject = env.scene[held_object_cfg.name]
    board: RigidObject = env.scene[board_cfg.name]
    sensor: ContactSensor = env.scene[contact_sensor_cfg.name]

    held_pos = held_object.data.root_pos_w - env.scene.env_origins
    board_pos = board.data.root_pos_w - env.scene.env_origins

    # Contact check: force above threshold + low velocity
    assert sensor.data.force_matrix_w.shape[1] == 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    force_norm = torch.norm(sensor.data.force_matrix_w.clone(), dim=-1).reshape(-1)
    has_contact = force_norm > force_threshold

    vel_norm = torch.norm(held_object.data.root_lin_vel_w, dim=-1)
    is_settled = vel_norm < velocity_threshold

    # Z-depth check: gear Z must be at or below peg top + threshold
    offset = torch.tensor(peg_offset_from_board, device=env.device)
    peg_top_z = board_pos[:, 2] + offset[2]
    is_deep_enough = held_pos[:, 2] < (peg_top_z + z_threshold)

    return has_contact & is_settled & is_deep_enough


def lift_object_il_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_position: tuple[float, float, float] | None = None,
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        goal_position: Fixed goal position [x, y, z] to use if command goal not available.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """

    object_instance: RigidObject = env.scene[object_cfg.name]
    object_pos = object_instance.data.root_pos_w

    goal_pos = torch.tensor([goal_position] * env.num_envs, device=env.device)

    # Check if object is within tolerance of goal
    distance = torch.norm(object_pos - goal_pos, dim=1)
    return distance < position_tolerance


def lift_object_rl_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rl_training: bool = False,
    command_name: str = "object_pose",
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Supports multiple modes:
    - RL training: Always returns False (no early termination)
    - RL evaluation: Uses goal from command manager

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        rl_training: If True, always returns False (disables success termination for RL training).
        command_name: The name of the command that is used to control the object.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """
    # During RL training, never terminate early
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_instance: RigidObject = env.scene[object_cfg.name]
    object_pos = object_instance.data.root_pos_w

    # Try to get goal position from command manager
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    goal_pos = command[:, :3]

    # Check if object is within tolerance of goal
    distance = torch.norm(object_pos - goal_pos, dim=1)
    return distance < position_tolerance


def goal_pose_task_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_x_range: tuple[float, float] | None = None,
    target_y_range: tuple[float, float] | None = None,
    target_z_range: tuple[float, float] | None = None,
    target_orientation_wxyz: tuple[float, float, float, float] | None = None,
    target_orientation_tolerance_rad: float = 0.1,
) -> torch.Tensor:
    """Terminate when the object's pose is within the thresholds (BBox + Orientation).

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        target_x_range: Success zone x-range [min, max] in meters.
        target_y_range: Success zone y-range [min, max] in meters.
        target_z_range: Success zone z-range [min, max] in meters.
        target_orientation_wxyz: Target quaternion [w, x, y, z].
        target_orientation_tolerance_rad: Angular tolerance in radians (default: 0.1).

    Returns:
        A boolean tensor of shape (num_envs, )
    """
    object_instance: RigidObject = env.scene[object_cfg.name]
    object_root_pos_w = object_instance.data.root_pos_w
    object_root_quat_w = object_instance.data.root_quat_w

    device = env.device
    num_envs = env.num_envs

    has_any_threshold = any([
        target_x_range is not None,
        target_y_range is not None,
        target_z_range is not None,
        target_orientation_wxyz is not None,
    ])

    if not has_any_threshold:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    success = torch.ones(num_envs, dtype=torch.bool, device=device)

    # Position range checks
    ranges = [target_x_range, target_y_range, target_z_range]
    for idx, range_val in enumerate(ranges):
        if range_val is not None:
            range_min, range_max = range_val
            in_range = (object_root_pos_w[:, idx] >= range_min) & (object_root_pos_w[:, idx] <= range_max)
            success &= in_range

    # Orientation check
    if target_orientation_wxyz is not None:
        target_quat = torch.tensor(target_orientation_wxyz, device=device, dtype=torch.float32).unsqueeze(0)

        # Formula: |<q1, q2>| > cos(tolerance / 2)
        quat_dot = torch.sum(object_root_quat_w * target_quat, dim=-1)
        abs_dot = torch.abs(quat_dot)
        min_cos = math.cos(target_orientation_tolerance_rad / 2.0)

        ori_success = abs_dot >= min_cos
        success &= ori_success

    return success


def root_height_below_minimum_multi_objects(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("robot")]
) -> torch.Tensor:
    """Terminate when any asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    outs = [
        root_height_below_minimum(env=env, minimum_height=minimum_height, asset_cfg=asset_cfg)
        for asset_cfg in asset_cfg_list
    ]
    outs_tensor = torch.stack(outs, dim=0)  # [X, N]
    terminated = outs_tensor.any(dim=0)  # [N], bool
    return terminated
