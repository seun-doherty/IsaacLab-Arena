# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA closed-loop policy for Isaac Lab Arena.

This module wraps Hugging Face's SmolVLA (via LeRobot) so that it conforms to
the :class:`isaaclab_arena.policy.policy_base.PolicyBase` interface.  It mirrors
the structure of :class:`Gr00tClosedloopPolicy` but removes all GR00T-specific
dependencies (embodiment tags, modality configs, joint remapping).

Key design decisions
--------------------
* **No joint remapping** – Franka's joint positions / EE-delta actions map 1-to-1
  between the simulation and the policy.  The GR00T-era
  ``remap_sim_joints_to_policy_joints`` / ``remap_policy_joints_to_sim_joints``
  helpers are therefore not needed.
* **Preprocessor pipeline** – We build the standard SmolVLA pre / post-processor
  pipeline (tokenisation, normalisation, device transfer) once at init time and
  apply it on every observation.
* **Action chunking** – Identical to the GR00T closed-loop policy: the model
  predicts ``action_horizon`` steps, and we execute ``action_chunk_length`` of
  them before requesting a new chunk.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from isaaclab_arena.policy.policy_base import PolicyBase

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports – keeps ``import isaaclab_arena_smolvla`` cheap when lerobot is
# not installed (e.g. in a pure-sim env without the model).
# ---------------------------------------------------------------------------

def _lazy_import_lerobot():
    """Import lerobot SmolVLA components.  Raises a helpful error if missing."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        from lerobot.utils.constants import (
            OBS_STATE,
            OBS_IMAGES,
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        )
        return (
            _SmolVLAPolicy,
            _SmolVLAConfig,
            make_smolvla_pre_post_processors,
            OBS_STATE,
            OBS_IMAGES,
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        )
    except ImportError as exc:
        raise ImportError(
            "SmolVLA requires `lerobot[smolvla]`.  "
            "Install with:  pip install lerobot[smolvla]"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _load_dataset_stats(stats_path: str | Path) -> dict[str, dict[str, torch.Tensor]] | None:
    """Load dataset normalisation statistics (``stats.json``) if available.

    Returns ``None`` when the file does not exist – in that case the
    pre-processor will skip normalisation.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        log.warning("Dataset stats file not found at %s – normalisation disabled", stats_path)
        return None

    with open(stats_path) as f:
        raw: dict = json.load(f)

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, val in raw.items():
        stats[key] = {}
        for stat_name, data in val.items():
            stats[key][stat_name] = torch.tensor(data, dtype=torch.float32)
    return stats


# ---------------------------------------------------------------------------
# Args dataclass (CLI / dict entry-point)
# ---------------------------------------------------------------------------

@dataclass
class SmolVLAClosedloopPolicyArgs:
    """Configuration dataclass for :class:`SmolVLAClosedloopPolicy`.

    Works exactly like ``Gr00tClosedloopPolicyArgs``: supports both
    dict-based (``from_dict``) and CLI-based (``from_cli_args``) construction.
    """

    policy_config_yaml_path: str = field(
        metadata={
            "help": "Path to the SmolVLA closedloop policy config YAML file",
            "required": True,
        }
    )
    policy_device: str = field(
        default="cuda",
        metadata={"help": "Device to use for the policy-related operations"},
    )
    num_envs: int = field(
        default=1,
        metadata={"help": "Number of environments to simulate"},
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> SmolVLAClosedloopPolicyArgs:
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
        )


# ---------------------------------------------------------------------------
# Main policy class
# ---------------------------------------------------------------------------

class SmolVLAClosedloopPolicy(PolicyBase):
    """Closed-loop SmolVLA policy for Isaac Lab Arena.

    Lifecycle::

        config_yaml  ─→  SmolVLAClosedloopPolicyConfig   (dataclass)
                                   │
        SmolVLAClosedloopPolicyArgs ─→ __init__
                                           │
                             ┌─────────────┴──────────────┐
                             │  load SmolVLAPolicy         │
                             │  build pre/post-processor   │
                             │  allocate action buffers    │
                             └─────────────┬──────────────┘
                                           │
                      env loop:  get_action(env, obs)
                                           │
                         ┌─────────────────┴────────────────┐
                         │  if chunk exhausted:              │
                         │    get_observations → preprocess  │
                         │    predict_action_chunk           │
                         │    postprocess → store chunk      │
                         │  return current_action_chunk[idx] │
                         └──────────────────────────────────┘
    """

    name = "smolvla_closedloop"
    config_class = SmolVLAClosedloopPolicyArgs

    def __init__(self, config: SmolVLAClosedloopPolicyArgs):
        super().__init__(config)

        # Import lerobot lazily
        (
            self._SmolVLAPolicy,
            self._SmolVLAConfig,
            self._make_processors,
            self._OBS_STATE,
            self._OBS_IMAGES,
            self._OBS_LANGUAGE_TOKENS,
            self._OBS_LANGUAGE_ATTENTION_MASK,
        ) = _lazy_import_lerobot()

        # Load our policy-level config (image sizes, dims, etc.)
        from isaaclab_arena_smolvla.policy.config.smolvla_closedloop_policy_config import (
            SmolVLAClosedloopPolicyConfig,
        )

        self.policy_config = self._load_policy_config(
            config.policy_config_yaml_path, SmolVLAClosedloopPolicyConfig
        )

        # Core settings
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.action_horizon = self.policy_config.action_horizon
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.action_dim = self.policy_config.action_dim

        # Load SmolVLA model
        self.policy = self._load_smolvla_policy()

        # Build pre/post-processor pipeline
        self.preprocessor, self.postprocessor = self._build_processors()

        # Tokenise the language instruction once (reused every step)
        self._tokenize_task()

        # ---------- action chunking buffers (mirrors GR00T) ----------
        self.current_action_chunk = torch.zeros(
            (self.num_envs, self.action_chunk_length, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.env_requires_new_action_chunk = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_action_index = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Task description (may be overridden by the env)
        self.task_description: str | None = self.policy_config.language_instruction or None

    # ------------------------------------------------------------------
    # Static factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_args(args: argparse.Namespace) -> SmolVLAClosedloopPolicy:
        config = SmolVLAClosedloopPolicyArgs.from_cli_args(args)
        return SmolVLAClosedloopPolicy(config)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        grp = parser.add_argument_group("SmolVLA Closedloop Policy")
        grp.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the SmolVLA closedloop policy config YAML file",
        )
        grp.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for policy operations (default: cuda)",
        )
        return parser

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_policy_config(yaml_path: str | Path, config_cls: type) -> Any:
        """Load a YAML file into a dataclass, mirroring ``create_config_from_yaml``."""
        data = _load_yaml(yaml_path)
        return config_cls(**data)

    def _load_smolvla_policy(self) -> Any:
        """Load SmolVLA weights from a local path or HuggingFace repo id."""
        model_path = self.policy_config.model_path
        log.info("Loading SmolVLA model from: %s", model_path)
        policy = self._SmolVLAPolicy.from_pretrained(
            model_path,
        )
        policy.to(self.device)
        policy.eval()
        return policy

    def _build_processors(self):
        """Build the SmolVLA pre/post-processor pipeline.

        If the model checkpoint directory contains saved processor configs
        (``policy_preprocessor.json`` / ``policy_postprocessor.json``), we
        load them with ``DataProcessorPipeline.from_pretrained`` so that
        rename maps, normalization stats, and tokenizer settings from
        training are faithfully restored.

        Falls back to building fresh processors via
        ``make_smolvla_pre_post_processors`` when no saved configs exist
        (e.g. when using the base ``lerobot/smolvla_base`` hub model).
        """
        from lerobot.processor.pipeline import DataProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        model_path = Path(self.policy_config.model_path)
        pre_cfg = model_path / "policy_preprocessor.json"
        post_cfg = model_path / "policy_postprocessor.json"

        if pre_cfg.exists() and post_cfg.exists():
            log.info("Loading saved preprocessor / postprocessor from %s", model_path)
            preprocessor = DataProcessorPipeline.from_pretrained(
                str(model_path),
                config_filename="policy_preprocessor.json",
            )
            postprocessor = DataProcessorPipeline.from_pretrained(
                str(model_path),
                config_filename="policy_postprocessor.json",
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )
            return preprocessor, postprocessor

        # Fallback: build fresh processors (e.g. for base hub models)
        log.info("No saved processor configs found – building from scratch")
        stats: dict[str, dict[str, torch.Tensor]] | None = None
        for candidate in [
            model_path / "stats.json",
            model_path.parent / "stats.json",
        ]:
            if candidate.exists():
                stats = _load_dataset_stats(candidate)
                break

        preprocessor, postprocessor = self._make_processors(
            config=self.policy.config,
            dataset_stats=stats,
        )
        return preprocessor, postprocessor

    def _tokenize_task(self) -> None:
        """Pre-tokenise the language task description.

        SmolVLA expects ``observation.language.tokens`` and
        ``observation.language.attention_mask`` in the batch dict.
        The preprocessor's ``TokenizerProcessorStep`` handles this
        automatically when a ``task`` key is present in the batch's
        complementary data, so we simply store the instruction string.
        """
        self._task_string = self.policy_config.language_instruction or ""

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        self.task_description = task_description
        self._task_string = task_description
        return self.task_description

    def get_observations(
        self,
        observation: dict[str, Any],
        camera_name: str | None = None,
    ) -> dict[str, Any]:
        """Convert an Isaac Lab Arena observation dict into a SmolVLA batch.

        The Arena observation has the shape::

            {
                "camera_obs": {"wrist_cam_rgb": Tensor[N, H, W, 3], ...},
                "policy":     {"joint_pos": Tensor[N, D], ...},
            }

        We produce a dict ready for ``SmolVLAPolicy.predict_action_chunk``::

            {
                "observation.images.<key>": Tensor[B, C, H, W],   float [0, 1]
                "observation.state":        Tensor[B, state_dim],
                "task":                     str   (complementary)
            }
        """
        if camera_name is None:
            camera_name = self.policy_config.pov_cam_name_sim

        # -- image ----------------------------------------------------------
        assert "camera_obs" in observation, "camera_obs missing from observation"
        assert camera_name in observation["camera_obs"], (
            f"Camera '{camera_name}' not in camera_obs (available: {list(observation['camera_obs'].keys())})"
        )
        rgb = observation["camera_obs"][camera_name]  # (N, H, W, C) uint8 or float
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        # (N, H, W, C) → (N, C, H, W)
        if rgb.ndim == 4 and rgb.shape[-1] in (1, 3):
            rgb = rgb.permute(0, 3, 1, 2)

        # -- state ----------------------------------------------------------
        state_key = self.policy_config.state_obs_key
        assert state_key in observation["policy"], (
            f"State key '{state_key}' not in policy obs "
            f"(available: {list(observation['policy'].keys())})"
        )
        joint_pos = observation["policy"][state_key]  # (N, state_dim)
        if joint_pos.device.type != "cpu":
            joint_pos = joint_pos.cpu()
        state = joint_pos.float()

        # -- build batch dict -----------------------------------------------
        img_key = f"observation.images.{self.policy_config.observation_image_key_lerobot}"
        batch = {
            img_key: rgb.to(self.device),
            self._OBS_STATE: state.to(self.device),
        }

        return batch

    # ------------------------------------------------------------------
    # Action inference
    # ------------------------------------------------------------------

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return the next single-step action per environment.

        Mirrors the GR00T closed-loop chunking logic: when the current chunk
        is exhausted a new forward pass is executed.
        """
        if any(self.env_requires_new_action_chunk):
            chunk = self.get_action_chunk(observation, self.policy_config.pov_cam_name_sim)
            self.current_action_chunk[self.env_requires_new_action_chunk] = chunk[
                self.env_requires_new_action_chunk
            ]
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        assert self.current_action_index.min() >= 0
        assert self.current_action_index.max() < self.action_chunk_length

        action = self.current_action_chunk[
            torch.arange(self.num_envs, device=self.device),
            self.current_action_index,
        ]
        assert action.shape == (self.num_envs, self.action_dim), (
            f"action.shape={action.shape} != ({self.num_envs}, {self.action_dim})"
        )

        self.current_action_index += 1

        # Mark envs whose chunk is exhausted
        reset_ids = self.current_action_index == self.action_chunk_length
        self.current_action_chunk[reset_ids] = 0.0
        self.env_requires_new_action_chunk[reset_ids] = True
        self.current_action_index[reset_ids] = -1

        return action

    def get_action_chunk(
        self,
        observation: dict[str, Any],
        camera_name: str | None = None,
    ) -> torch.Tensor:
        """Run a full SmolVLA forward pass and return an action chunk.

        Returns:
            Tensor of shape ``(num_envs, action_chunk_length, action_dim)``
        """
        batch = self.get_observations(observation, camera_name)

        # Include the task string directly in the batch dict.
        # The DataProcessorPipeline's batch_to_transition converter
        # automatically extracts "task" into complementary_data.
        batch["task"] = self._task_string

        # Run the preprocessor pipeline (adds batch dim, tokenises task,
        # normalises, moves to device).
        batch = self.preprocessor(batch)

        # SmolVLA forward → (B, n_action_steps, padded_action_dim)
        raw_actions = self.policy.predict_action_chunk(batch)

        # Post-process (unnormalise, move to cpu)
        raw_actions = self.postprocessor(raw_actions)

        # raw_actions: (B, n_action_steps, action_dim) where B may be 1
        # Ensure batch dimension matches num_envs
        if raw_actions.shape[0] == 1 and self.num_envs > 1:
            raw_actions = raw_actions.expand(self.num_envs, -1, -1)

        # Slice to the requested chunk length and action dim
        actions = raw_actions[:, : self.action_chunk_length, : self.action_dim]
        actions = actions.to(dtype=torch.float32, device=self.device)

        # Isaac Lab Franka BinaryJointPositionAction: negative = close, non-negative = open.
        # After normalization during training, the policy learned the same convention
        # (negative = close, positive = open), so we pass the gripper dimension through as-is.
        # If instead the gripper starts closed and opens near the object, the convention
        # is inverted and you can uncomment the line below to flip the gripper sign.
        # actions[..., -1] = -actions[..., -1]

        assert actions.shape == (self.num_envs, self.action_chunk_length, self.action_dim), (
            f"Expected ({self.num_envs}, {self.action_chunk_length}, {self.action_dim}), "
            f"got {actions.shape}"
        )
        return actions

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the action chunking mechanism and the SmolVLA internal queue."""
        if env_ids is None:
            env_ids = slice(None)
        self.policy.reset()
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True
