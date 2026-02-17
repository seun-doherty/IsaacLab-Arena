# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for the LeRobot replay action policy (no GR00T deps)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SmolVLAReplayActionPolicyConfig:
    """Configuration for replaying actions from a standard LeRobot v3.0 dataset.

    Unlike ``LerobotReplayActionPolicyConfig`` (GR00T), this config has:
    - No ``embodiment_tag`` or ``modality_config_path``
    - No joint remapping
    - Direct action pass-through from the dataset
    """

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_path: str = field(
        default="",
        metadata={"description": "Absolute path to the LeRobot v3.0 dataset directory."},
    )

    # ── Action chunking ──────────────────────────────────────────────────
    action_horizon: int = field(
        default=1,
        metadata={"description": "Number of future actions to read per step."},
    )
    action_chunk_length: int = field(
        default=1,
        metadata={
            "description": (
                "Number of actions to execute per inference step.  For replay "
                "this is normally 1 (one action per recorded timestamp)."
            )
        },
    )

    # ── Robot dimensions ─────────────────────────────────────────────────
    action_dim: int = field(
        default=7,
        metadata={"description": "Dimensionality of the action vector."},
    )

    # ── Validation ───────────────────────────────────────────────────────
    def __post_init__(self):
        if self.action_chunk_length > self.action_horizon:
            raise ValueError(
                f"action_chunk_length ({self.action_chunk_length}) must be "
                f"<= action_horizon ({self.action_horizon})"
            )
        # Resolve to absolute and verify existence
        self.dataset_path = str(Path(self.dataset_path).resolve())
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"dataset_path does not exist: {self.dataset_path}")
