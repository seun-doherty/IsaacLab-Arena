# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Replay-action policy for the Franka robot using a standard LeRobot dataset.

This policy reads recorded actions from a LeRobot v3.0 dataset and plays them
back in simulation.  It is the GR00T-free counterpart of
:class:`isaaclab_arena_gr00t.policy.replay_lerobot_action_policy.ReplayLerobotActionPolicy`.

Key differences from the GR00T replay policy:
- No ``gr00t`` imports (no ``EmbodimentTag``, ``ReplayPolicy``, ``modality_config``)
- No joint remapping – actions are used as-is from the dataset
- Reads parquet-based LeRobot v3.0 datasets directly via ``pandas``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from isaaclab_arena.policy.policy_base import PolicyBase

log = logging.getLogger(__name__)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Args dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReplayLerobotFrankaPolicyArgs:
    """Configuration for :class:`ReplayLerobotFrankaPolicy`."""

    policy_config_yaml_path: str = field(
        metadata={
            "help": "Path to the replay action policy config YAML file",
            "required": True,
        }
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device for policy output tensors"},
    )
    num_envs: int = field(
        default=1,
        metadata={"help": "Number of environments to simulate"},
    )
    trajectory_index: int = field(
        default=0,
        metadata={"help": "Index of the episode/trajectory to replay"},
    )
    max_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of steps to run the replay for"},
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ReplayLerobotFrankaPolicyArgs:
        return cls(
            policy_config_yaml_path=args.config_yaml_path,
            device=getattr(args, "device", "cuda"),
            num_envs=args.num_envs,
            trajectory_index=args.trajectory_index,
            max_steps=args.max_steps,
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class ReplayLerobotFrankaPolicy(PolicyBase):
    """Replay recorded Franka actions from a LeRobot v3.0 dataset.

    The dataset is expected to contain at minimum an ``action`` column whose
    entries are lists/arrays of length ``action_dim``.  Episode boundaries are
    identified by the ``episode_index`` column.
    """

    name = "replay_lerobot_franka"
    config_class = ReplayLerobotFrankaPolicyArgs

    def __init__(self, config: ReplayLerobotFrankaPolicyArgs):
        super().__init__(config)

        from isaaclab_arena_smolvla.policy.config.smolvla_replay_action_policy_config import (
            SmolVLAReplayActionPolicyConfig,
        )

        yaml_data = _load_yaml(config.policy_config_yaml_path)
        self.policy_config = SmolVLAReplayActionPolicyConfig(**yaml_data)

        self.num_envs = config.num_envs
        self.device = config.device
        self.trajectory_index = config.trajectory_index
        self.max_steps = config.max_steps
        self.action_dim = self.policy_config.action_dim
        self.action_chunk_length = self.policy_config.action_chunk_length

        # Load actions from the dataset
        self._actions, self._episode_lengths = self._load_dataset(self.policy_config.dataset_path)
        self._num_episodes = len(self._episode_lengths)

        assert self.trajectory_index < self._num_episodes, (
            f"trajectory_index {self.trajectory_index} >= number of episodes {self._num_episodes}"
        )

        # Set up cursors for the current episode
        self._episode_actions = self._get_episode_actions(self.trajectory_index)
        self._step_index = 0

        self.current_action_chunk: torch.Tensor | None = None
        self.current_action_index = 0

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self, dataset_path: str) -> tuple[pd.DataFrame, dict[int, int]]:
        """Load the parquet data and compute per-episode lengths.

        Returns:
            (actions_df, {episode_index: length})
        """
        ds_path = Path(dataset_path)

        # LeRobot v3.0 layout: data/chunk-*/file-*.parquet
        parquet_dir = ds_path / "data"
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {parquet_dir}")

        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        log.info("Loaded %d rows from %d parquet files", len(df), len(parquet_files))

        assert "action" in df.columns, f"Expected 'action' column, got {list(df.columns)}"
        assert "episode_index" in df.columns, f"Expected 'episode_index' column, got {list(df.columns)}"

        episode_lengths: dict[int, int] = df.groupby("episode_index").size().to_dict()
        return df, episode_lengths

    def _get_episode_actions(self, episode_index: int) -> np.ndarray:
        """Extract the action array for a single episode.

        Returns:
            np.ndarray of shape (T, action_dim)
        """
        episode_df = self._actions[self._actions["episode_index"] == episode_index]
        actions = np.stack(episode_df["action"].values)
        return actions

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return the next recorded action."""
        if self.current_action_chunk is None and self.current_action_index == 0:
            self.current_action_chunk = self._get_next_action_chunk()
            assert self.current_action_chunk is not None

        assert self.current_action_chunk is not None
        assert self.current_action_index < self.action_chunk_length

        action = self.current_action_chunk[:, self.current_action_index]
        self.current_action_index += 1

        if self.current_action_index == self.action_chunk_length:
            self.current_action_chunk = None
            self.current_action_index = 0

        return action

    def _get_next_action_chunk(self) -> torch.Tensor:
        """Read the next ``action_chunk_length`` actions from the dataset."""
        chunk_actions = []
        for _ in range(self.action_chunk_length):
            if self._step_index >= len(self._episode_actions):
                # Reached end of episode – repeat last action
                action = self._episode_actions[-1]
            else:
                action = self._episode_actions[self._step_index]
                self._step_index += 1
            chunk_actions.append(action)

        # (chunk_length, action_dim)
        chunk_np = np.stack(chunk_actions, axis=0)
        # (1, chunk_length, action_dim) → tile to (num_envs, chunk_length, action_dim)
        chunk_tensor = torch.from_numpy(chunk_np).float().unsqueeze(0)
        chunk_tensor = chunk_tensor.expand(self.num_envs, -1, -1).to(self.device)
        return chunk_tensor

    # ------------------------------------------------------------------
    # Trajectory / length helpers
    # ------------------------------------------------------------------

    def get_trajectory_length(self, trajectory_index: int) -> int:
        assert trajectory_index in self._episode_lengths, (
            f"Episode {trajectory_index} not found (available: {list(self._episode_lengths.keys())})"
        )
        return self._episode_lengths[trajectory_index]

    def get_trajectory_index(self) -> int:
        return self.trajectory_index

    def has_length(self) -> bool:
        return True

    def length(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return self.get_trajectory_length(self.trajectory_index)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, trajectory_index: int = 0):
        """Reset the replay cursor to the start of the given episode."""
        self.trajectory_index = trajectory_index
        self._episode_actions = self._get_episode_actions(trajectory_index)
        self._step_index = 0
        self.current_action_chunk = None
        self.current_action_index = 0

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        grp = parser.add_argument_group("Replay LeRobot Franka Policy")
        grp.add_argument(
            "--config_yaml_path",
            type=str,
            required=True,
            help="Path to the replay action policy config YAML file",
        )
        grp.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="Maximum number of steps to replay",
        )
        grp.add_argument(
            "--trajectory_index",
            type=int,
            default=0,
            help="Index of the trajectory/episode to replay (default: 0)",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> ReplayLerobotFrankaPolicy:
        config = ReplayLerobotFrankaPolicyArgs.from_cli_args(args)
        return ReplayLerobotFrankaPolicy(config)
