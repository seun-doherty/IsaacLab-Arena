# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for SmolVLA closed-loop policy."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SmolVLAClosedloopPolicyConfig:
    """Configuration for the SmolVLA closed-loop inference policy.

    This config is loaded from a YAML file via ``create_config_from_yaml``.
    It mirrors the structure of ``Gr00tClosedloopPolicyConfig`` but drops all
    GR00T-specific fields (embodiment tags, modality configs, joint remapping)
    in favour of standard LeRobot / SmolVLA parameters.
    """

    # ── Model ────────────────────────────────────────────────────────────
    model_path: str = field(
        default="lerobot/smolvla_base",
        metadata={
            "description": (
                "Path to a local SmolVLA checkpoint directory **or** a HuggingFace "
                "repo id (e.g. 'lerobot/smolvla_base').  The LeRobot "
                "``SmolVLAPolicy.from_pretrained`` loader accepts both."
            )
        },
    )
    language_instruction: str = field(
        default="",
        metadata={"description": "Natural-language instruction passed to the policy at every step."},
    )

    # ── Action chunking ──────────────────────────────────────────────────
    action_horizon: int = field(
        default=10,
        metadata={"description": "Total number of future actions the model predicts per forward pass."},
    )
    action_chunk_length: int = field(
        default=10,
        metadata={
            "description": (
                "How many of those predicted actions to actually *execute* before "
                "querying the model again.  Must be <= action_horizon."
            )
        },
    )

    # ── Observation keys ─────────────────────────────────────────────────
    pov_cam_name_sim: str = field(
        default="wrist_cam_rgb",
        metadata={
            "description": (
                "Key inside ``observation['camera_obs']`` that holds the primary "
                "point-of-view RGB tensor from the simulation.  For Franka the "
                "camera config field is 'wrist_cam' and the data type is 'rgb', "
                "so the observation manager exposes it as 'wrist_cam_rgb'."
            )
        },
    )
    observation_image_key_lerobot: str = field(
        default="wrist",
        metadata={
            "description": (
                "Suffix used in the LeRobot frame dict, i.e. the frame will "
                "contain ``observation.images.<this_value>``."
            )
        },
    )
    state_obs_key: str = field(
        default="joint_pos",
        metadata={
            "description": (
                "Key inside ``observation['policy']`` that holds the robot joint "
                "state tensor.  Franka uses 'joint_pos'; GR1/G1 uses "
                "'robot_joint_pos'.  Must match the embodiment's ObservationsCfg."
            )
        },
    )

    # ── Image ────────────────────────────────────────────────────────────
    target_image_height: int = field(
        default=200,
        metadata={"description": "Height to resize camera frames to before feeding the model."},
    )
    target_image_width: int = field(
        default=200,
        metadata={"description": "Width to resize camera frames to before feeding the model."},
    )

    # ── Robot dimensions ─────────────────────────────────────────────────
    state_dim: int = field(
        default=9,
        metadata={"description": "Dimensionality of observation.state (e.g. 9 for Franka: 7 joints + 2 gripper)."},
    )
    action_dim: int = field(
        default=7,
        metadata={"description": "Dimensionality of the action vector (e.g. 7 for Franka: 6 EE delta + 1 gripper)."},
    )

    # ── Device ───────────────────────────────────────────────────────────
    policy_device: str = field(
        default="cuda",
        metadata={"description": "Device to run the SmolVLA model on ('cuda' or 'cpu')."},
    )

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = field(default=42, metadata={"description": "Random seed for reproducibility."})

    # ── Validation ───────────────────────────────────────────────────────
    def __post_init__(self):
        if self.action_chunk_length > self.action_horizon:
            raise ValueError(
                f"action_chunk_length ({self.action_chunk_length}) must be "
                f"<= action_horizon ({self.action_horizon})"
            )
        # If model_path looks like a local path, verify it exists
        if self.model_path and self.model_path.startswith((".", "/")) or (
            len(self.model_path) >= 2 and self.model_path[1] == ":"
        ):
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"model_path does not exist: {self.model_path}")
