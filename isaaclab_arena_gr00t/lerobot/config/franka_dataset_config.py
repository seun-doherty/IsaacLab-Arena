# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FrankaDatasetConfig:
    # Datasets & task specific parameters
    data_root: Path = field(
        default=Path("datasets/"),
        metadata={"description": "Root directory for all data storage."},
    )
    language_instruction: str = field(
        default=None, metadata={"description": "Instruction given to the policy in natural language."}
    )
    hdf5_name: str = field(
        default=None, metadata={"description": "Name of the HDF5 file to use for the dataset."}
    )

    # Simulation HDF5 datafield names
    # NOTE: robot joint position must exist in the HDF5 file
    state_name_sim: str = field(
        default="obs/joint_pos",
        metadata={"description": "Name of the state in the HDF5 file (9D: 7 joints + 2 gripper fingers)."},
    )
    action_name_sim: str = field(
        default="actions",
        metadata={"description": "Name of the action in the HDF5 file (7D: 6D ee delta + 1D gripper)."},
    )
    eef_pos_name_sim: str = field(
        default="obs/eef_pos",
        metadata={"description": "Name of the end-effector position in the HDF5 file (optional)."},
    )
    eef_quat_name_sim: str = field(
        default="obs/eef_quat",
        metadata={"description": "Name of the end-effector quaternion in the HDF5 file (optional)."},
    )
    pov_cam_name_sim: str = field(
        default="obs/wrist_cam", metadata={"description": "Name of the primary POV camera in the HDF5 file."}
    )
    table_cam_name_sim: str = field(
        default="obs/table_cam", metadata={"description": "Name of the table camera in the HDF5 file (optional)."}
    )

    # Gr00t-LeRobot datafield names
    state_name_lerobot: str = field(
        default="observation.state", metadata={"description": "Name of the state in the LeRobot file."}
    )
    action_name_lerobot: str = field(
        default="action", metadata={"description": "Name of the action in the LeRobot file."}
    )
    wrist_video_name_lerobot: str = field(
        default="observation.images.wrist", metadata={"description": "Name of the wrist video in the LeRobot file."}
    )
    table_video_name_lerobot: str = field(
        default="observation.images.table", metadata={"description": "Name of the table video in the LeRobot file."}
    )
    task_description_lerobot: str = field(
        default="annotation.human.action.task_description",
        metadata={"description": "Name of the task description in the LeRobot file."},
    )
    valid_lerobot: str = field(
        default="annotation.human.action.valid",
        metadata={"description": "Name of the validity in the LeRobot file."},
    )

    # Parquet configuration
    chunks_size: int = field(default=100, metadata={"description": "Number of episodes per data chunk."})

    # Video configuration
    fps: int = field(default=30, metadata={"description": "Frames per second for video recording."})

    # File path templates
    data_path: str = field(
        default="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        metadata={"description": "Template path for storing episode data files."},
    )
    video_path: str = field(
        default="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        metadata={"description": "Template path for storing episode video files."},
    )

    # Configuration file paths
    modality_template_path: Path = field(
        default=Path(__file__).resolve().parent.parent.parent / "embodiments" / "franka" / "modality.json",
        metadata={"description": "Path to the modality template JSON file."},
    )
    modality_fname: str = field(
        default="modality.json", metadata={"description": "Filename for the modality JSON file."}
    )
    episodes_fname: str = field(
        default="episodes.jsonl", metadata={"description": "Filename for the episodes JSONL file."}
    )
    tasks_fname: str = field(
        default="tasks.jsonl", metadata={"description": "Filename for the tasks JSONL file."}
    )
    info_template_path: Path = field(
        default=Path(__file__).resolve().parent.parent.parent / "embodiments" / "franka" / "info.json",
        metadata={"description": "Path to the info template JSON file."},
    )
    info_fname: str = field(
        default="info.json", metadata={"description": "Filename for the info JSON file."}
    )

    # Policy specific parameters (7 DOF end-effector space)
    policy_joints_config_path: Path = field(
        default=Path(__file__).resolve().parent.parent.parent / "embodiments" / "franka" / "8dof_joint_space.yaml",
        metadata={
            "description": "Path to the YAML file specifying the action space configuration (7D end-effector)."
        },
    )
    robot_type: str = field(
        default="franka_panda", metadata={"description": "Type of robot embodiment used in the policy fine-tuning."}
    )

    # Robot simulation specific parameters (9 DOF joint space)
    action_joints_config_path: Path = field(
        default=Path(__file__).resolve().parent.parent.parent / "embodiments" / "franka" / "8dof_joint_space.yaml",
        metadata={
            "description": "Path to the YAML file specifying the joint ordering for robot action space in simulation."
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).resolve().parent.parent.parent / "embodiments" / "franka" / "9dof_joint_space.yaml",
        metadata={
            "description": "Path to the YAML file specifying the joint ordering for robot state space in simulation."
        },
    )

    # Image configuration (height, width, channels)
    original_image_size: tuple[int, int, int] = field(
        default=(200, 200, 3), metadata={"description": "Original size of input images as (height, width, channels)."}
    )
    target_image_size: tuple[int, int, int] = field(
        default=(200, 200, 3), metadata={"description": "Target size for images after resizing and padding."}
    )

    # Derived fields
    hdf5_file_path: Path = field(init=False)
    lerobot_data_dir: Path = field(init=False)
    task_index: int = field(
        default=0, metadata={"description": "Task index for the task description in LeRobot file."}
    )

    def __post_init__(self):
        # Construct file paths
        self.hdf5_file_path = self.data_root / self.hdf5_name
        self.lerobot_data_dir = self.data_root / self.hdf5_name.replace(".hdf5", "") / "lerobot"

        # Validate required files exist
        assert self.hdf5_file_path.exists(), f"{self.hdf5_file_path} does not exist"
        assert Path(self.policy_joints_config_path).exists(), f"{self.policy_joints_config_path} does not exist"
        assert Path(self.action_joints_config_path).exists(), f"{self.action_joints_config_path} does not exist"
        assert Path(self.state_joints_config_path).exists(), f"{self.state_joints_config_path} does not exist"
        assert Path(self.info_template_path).exists(), f"{self.info_template_path} does not exist"
        assert Path(self.modality_template_path).exists(), f"{self.modality_template_path} does not exist"

        # Handle existing lerobot directory
        if self.lerobot_data_dir.exists():
            print(f"Warning: lerobot_data_dir {self.lerobot_data_dir} already exists.")
            if input(f"Are you sure you want to remove {self.lerobot_data_dir}? (y/n): ") == "y":
                shutil.rmtree(self.lerobot_data_dir)
            else:
                print(f"Skipping removal of {self.lerobot_data_dir}")

        # Prepare data keys for simulation HDF5 file
        # Minimum required keys: state and action
        self.hdf5_keys = {
            "state": self.state_name_sim,
            "action": self.action_name_sim,
            "wrist_cam": self.pov_cam_name_sim,
        }

        # Optional keys if provided
        if self.eef_pos_name_sim:
            self.hdf5_keys["eef_pos"] = self.eef_pos_name_sim
        if self.eef_quat_name_sim:
            self.hdf5_keys["eef_quat"] = self.eef_quat_name_sim
        if self.table_cam_name_sim:
            self.hdf5_keys["table_cam"] = self.table_cam_name_sim

        # Prepare data keys for LeRobot file
        self.lerobot_keys = {
            "state": self.state_name_lerobot,
            "action": self.action_name_lerobot,
            "wrist_video": self.wrist_video_name_lerobot,
            "annotation": (self.task_description_lerobot,),
        }

        # Add optional LeRobot keys
        if "table_cam" in self.hdf5_keys:
            self.lerobot_keys["table_video"] = self.table_video_name_lerobot
        if "eef_pos" in self.hdf5_keys and "eef_quat" in self.hdf5_keys:
            self.lerobot_keys["eef_pose"] = "observation.eef_pose"
