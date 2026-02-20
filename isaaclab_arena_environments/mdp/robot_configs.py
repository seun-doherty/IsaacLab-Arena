# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Task-specific robot configurations for manipulation tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


def franka_gripper_joint_setter(
    joint_pos: torch.Tensor,
    row_indices: Sequence[int],
    finger_joint_indices: Sequence[int],
    width: float,
) -> None:
    """Set Franka Panda finger joints to achieve a given total opening *width*.

    Each finger joint is set to ``width / 2`` (symmetric grasp).
    Signature matches the ``gripper_joint_setter_func`` expected by
    :class:`~isaaclab_arena.terms.events.place_gear_in_gripper`.
    """
    for jid in finger_joint_indices:
        joint_pos[row_indices, jid] = width / 2.0

# ===========================================================================================
# Franka Panda robot configuration optimized for assembly tasks (peg insert, gear mesh, etc.).
# ===========================================================================================

FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()

# Enable contact sensors for assembly tasks
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.activate_contact_sensors = True

# Disable gravity on the robot for better control stability
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True

# Increase stiffness and damping for precise positioning
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 30.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 150.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_forearm"].damping = 30.0
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_hand"].stiffness = 2e3
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.actuators["panda_hand"].damping = 1e2

# Set initial position for tabletop tasks
FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.init_state.pos = (0.0, 0.0, 0.0)
