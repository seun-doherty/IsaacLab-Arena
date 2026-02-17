# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Convert Franka HDF5 teleop/mimic demonstrations to standard LeRobot dataset format.

This converter is Franka-specific and does NOT depend on GR00T modality configs,
embodiment tags, or joint remapping. It reads the HDF5 directly and writes a clean
LeRobot v2 dataset (parquet + mp4 videos + meta JSONs).

HDF5 layout expected (per demo):
    data/demo_X/
        obs/
            joint_pos              # joint positions (N, 9)
            eef_pos                # end-effector position (N, 3)
            eef_quat               # end-effector quaternion (N, 4)
            gripper_pos            # gripper finger positions (N, 2)
            wrist_cam              # RGB wrist camera frames (N, H, W, 3)
            table_cam              # RGB table camera frames (N, H, W, 3)
        actions                    # raw teleop actions (N, 7): 6D EE delta + 1D gripper
        processed_actions          # processed actions (N, 8): 6D scaled EE delta + 2D gripper joints
        states/                    # full articulation/rigid-object state
            articulation/robot/joint_position  # (N, 9)

If state has one more frame than action (IsaacLab trailing-frame artifact),
the extra state frame is trimmed automatically.

Usage:
    python convert_hdf5_to_lerobot_franka.py --yaml_file <path-to-franka-config.yaml>
"""

import h5py
import json
import multiprocessing as mp
import numpy as np
import shutil
import subprocess
import time
import torchvision
import traceback
from dataclasses import fields
from pathlib import Path
from tqdm import tqdm
from typing import Any

import pandas as pd

from isaaclab_arena_gr00t.lerobot.config.franka_dataset_config import FrankaDatasetConfig
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    dump_json,
    dump_jsonl,
    load_json,
)


# ---------------------------------------------------------------------------
# Video helpers (reused from the generic converter, no GR00T dependency)
# ---------------------------------------------------------------------------

def wait_for_video_completion(video_path: str | Path, max_wait_time: int = 60, check_interval: float = 0.5) -> bool:
    """Wait for a video file to be completely written and accessible."""
    video_path = Path(video_path)
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if not video_path.exists():
            time.sleep(check_interval)
            continue
        try:
            size1 = video_path.stat().st_size
            time.sleep(check_interval)
            size2 = video_path.stat().st_size
            if size1 == size2 and size1 > 0:
                try:
                    with open(video_path, "rb") as f:
                        f.read(1024)
                    return True
                except OSError:
                    pass
        except OSError:
            pass
        time.sleep(check_interval)
    return False


def get_video_metadata(video_path: str) -> dict[str, Any] | None:
    """Extract video metadata via ffprobe."""
    if not wait_for_video_completion(video_path, max_wait_time=60):
        print(f"Timeout waiting for video completion: {video_path}")
        return None

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of", "json",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        audio_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(video_path),
        ]
        audio_output = subprocess.check_output(audio_cmd).decode("utf-8")
        audio_data = json.loads(audio_output)
        has_audio = len(audio_data.get("streams", [])) > 0

        return {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.width": stream["width"],
                "video.height": stream["height"],
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": has_audio,
            },
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"Error extracting video metadata for {video_path}: {e}")
        return None


def write_video_job(queue: mp.Queue, error_queue: mp.Queue, config: FrankaDatasetConfig) -> None:
    """Worker: encode frames to mp4 from the queue."""
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            video_path, frames, fps = job
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            # Resize if needed
            if frames.shape[1:] != config.original_image_size:
                print(f"Warning: frame shape {frames.shape[1:]} != expected {config.original_image_size}")
            if config.target_image_size != config.original_image_size:
                import torch
                t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                h, w = config.target_image_size[0], config.target_image_size[1]
                t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
                frames = t.permute(0, 2, 3, 1).byte().numpy()

            torchvision.io.write_video(str(video_path), frames, fps, video_codec="h264")
        except Exception as e:
            error_msg = f"Error creating video {video_path}: {e}\n{traceback.format_exc()}"
            print(error_msg)
            error_queue.put(error_msg)


# ---------------------------------------------------------------------------
# Trajectory → DataFrame (no GR00T remapping / modality)
# ---------------------------------------------------------------------------

def _resolve_hdf5_dataset(group, key_path: str):
    """Resolve a slash-separated key path in an HDF5 group.

    E.g. key_path="states/articulation/robot/joint_position" traverses
    group["states"]["articulation"]["robot"]["joint_position"].
    """
    parts = key_path.split("/")
    node = group
    for p in parts:
        assert p in node, f"Key '{p}' not found in HDF5 group. Available: {list(node.keys())}"
        node = node[p]
    return node


def convert_trajectory_to_df(
    trajectory,
    episode_index: int,
    index_start: int,
    config: FrankaDatasetConfig,
) -> dict[str, Any]:
    """Convert a single Franka HDF5 trajectory to a pandas DataFrame.

    Reads state and action arrays directly (no joint remapping), trims
    the trailing IsaacLab frame, and builds a LeRobot-format DataFrame.
    """
    data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. State (observation.state)
    # ------------------------------------------------------------------
    state_ds = _resolve_hdf5_dataset(trajectory, config.hdf5_keys["state"])
    state = np.array(state_ds, dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. Action
    # ------------------------------------------------------------------
    action_ds = _resolve_hdf5_dataset(trajectory, config.hdf5_keys["action"])
    action = np.array(action_ds, dtype=np.float32)

    # ------------------------------------------------------------------
    # Handle IsaacLab trailing-frame artifact (N+1 → N).
    # Some IsaacLab recordings add an extra observation frame at the end
    # that has no matching action. Detect this by comparing lengths: if
    # state has one more row than action, trim the trailing state row.
    # If they match, keep everything as-is.
    # ------------------------------------------------------------------
    if state.shape[0] == action.shape[0] + 1:
        print(f"  Trimming trailing IsaacLab state frame ({state.shape[0]} → {action.shape[0]})")
        state = state[:-1]
    elif state.shape[0] != action.shape[0]:
        raise ValueError(
            f"State and action length mismatch: state={state.shape[0]}, action={action.shape[0]}. "
            f"Expected equal lengths or state to have exactly one extra trailing frame."
        )
    length = state.shape[0]

    data[config.lerobot_keys["state"]] = [row for row in state]
    data[config.lerobot_keys["action"]] = [row for row in action]

    # ------------------------------------------------------------------
    # 3. Optional: end-effector pose
    # ------------------------------------------------------------------
    if "eef_pos" in config.hdf5_keys and "eef_quat" in config.hdf5_keys:
        eef_pos = np.array(_resolve_hdf5_dataset(trajectory, config.hdf5_keys["eef_pos"]), dtype=np.float32)[:length]
        eef_quat = np.array(_resolve_hdf5_dataset(trajectory, config.hdf5_keys["eef_quat"]), dtype=np.float32)[:length]
        eef_pose = np.concatenate([eef_pos, eef_quat], axis=1)
        assert eef_pose.shape[0] == length
        data[config.lerobot_keys["eef_pose"]] = [row for row in eef_pose]

    # ------------------------------------------------------------------
    # 4. Timestamps and episode metadata
    # ------------------------------------------------------------------
    data["timestamp"] = (np.arange(length, dtype=np.float64) * (1.0 / config.fps))
    data["episode_index"] = np.ones(length, dtype=np.int64) * episode_index
    data["task_index"] = np.ones(length, dtype=np.int64) * config.task_index
    data["index"] = np.arange(length, dtype=np.int64) + index_start
    data["frame_index"] = np.arange(length, dtype=np.int64)

    # Reward: last frame = 1 (assuming successful demo)
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = 1.0
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done

    # Annotation (task description index)
    annotation_key = config.lerobot_keys["annotation"][0]
    data[annotation_key] = np.ones(length, dtype=np.int64) * config.task_index

    return {
        "data": pd.DataFrame(data),
        "length": length,
        "annotation": {config.task_index},
    }


# ---------------------------------------------------------------------------
# Feature info for info.json
# ---------------------------------------------------------------------------

def get_feature_info(
    step_data: pd.DataFrame,
    video_paths: dict[str, str],
    config: FrankaDatasetConfig,
) -> dict[str, Any]:
    """Build the 'features' dict for info.json."""
    features: dict[str, Any] = {}

    # Video features
    for video_key, video_path in video_paths.items():
        meta = get_video_metadata(str(video_path))
        if meta is not None:
            features[video_key] = meta

    # Tabular features
    for column in step_data.columns:
        first_val = step_data[column].iloc[0]
        if isinstance(first_val, np.ndarray):
            col_data = np.stack(step_data[column].values, axis=0)  # type: ignore[arg-type]
        else:
            col_data = step_data[column].values
        shape = col_data.shape
        if len(shape) == 1:
            shape_out = [1]
        else:
            shape_out = list(shape[1:])

        entry: dict[str, Any] = {
            "dtype": str(col_data.dtype),
            "shape": shape_out,
        }

        # Annotate state/action dimension names from info template
        if column == config.lerobot_keys["state"]:
            info_template = load_json(config.info_template_path)
            if "features" in info_template and "observation.state" in info_template["features"]:
                entry["names"] = info_template["features"]["observation.state"].get("names")
        elif column == config.lerobot_keys["action"]:
            info_template = load_json(config.info_template_path)
            if "features" in info_template and "action" in info_template["features"]:
                entry["names"] = info_template["features"]["action"].get("names")

        features[column] = entry

    return features


def generate_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    config: FrankaDatasetConfig,
    step_data: pd.DataFrame,
    video_paths: dict[str, str],
) -> dict[str, Any]:
    """Generate the info.json content."""
    info = load_json(config.info_template_path)

    info["robot_type"] = config.robot_type
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = total_tasks
    info["total_videos"] = total_videos
    info["total_chunks"] = total_chunks
    info["chunks_size"] = config.chunks_size
    info["fps"] = config.fps
    info["data_path"] = config.data_path
    info["video_path"] = config.video_path
    info["features"] = get_feature_info(step_data, video_paths, config)

    return info


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def generate_dataset_stats(
    parquet_dir: Path,
    camera_lerobot_keys: list[str],
    total_frames: int,
    output_path: Path,
) -> None:
    """Generate ``stats.json`` for the LeRobot dataset.

    This computes per-feature statistics (mean, std, min, max, count) from the
    parquet data **and** adds placeholder entries for video/image features so
    that the LeRobot training pipeline can inject ImageNet normalisation stats
    without crashing.

    Without the image key entries the training factory raises::

        KeyError: 'observation.images.wrist'

    because ``make_dataset`` iterates ``dataset.meta.camera_keys`` and expects
    each key to already exist in ``stats``.
    """
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        print("  WARNING: No parquet files found – skipping stats generation")
        return

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    stats: dict[str, dict[str, Any]] = {}

    # --- Numeric feature stats ---
    for col in df.columns:
        sample = df[col].iloc[0]
        # Only process numeric columns (scalars, lists, arrays)
        if isinstance(sample, (int, float, np.integer, np.floating)):
            values = df[col].values.astype(float)
            stats[col] = {
                "min": [float(values.min())],
                "max": [float(values.max())],
                "mean": [float(values.mean())],
                "std": [float(values.std())],
                "count": [int(len(values))],
            }
        elif isinstance(sample, (list, np.ndarray)):
            arr = np.stack(df[col].values)
            stats[col] = {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "count": [int(arr.shape[0])],
            }

    # --- Video / image feature placeholder stats ---
    # These MUST exist so the training pipeline can set ImageNet stats.
    # Values here are ImageNet defaults; they get overridden by
    # ``make_dataset`` when ``use_imagenet_stats=True``, but the key
    # must be present in the dict for the code to not crash.
    for cam_key in camera_lerobot_keys:
        stats[cam_key] = {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],   # ImageNet RGB means
            "std":  [[[0.229]], [[0.224]], [[0.225]]],   # ImageNet RGB stds
            "min":  [[[0.0]], [[0.0]], [[0.0]]],
            "max":  [[[1.0]], [[1.0]], [[1.0]]],
            "count": [int(total_frames)],
        }

    dump_json(stats, output_path, indent=2)
    print(f"  Generated stats.json with {len(stats)} features "
          f"(including {len(camera_lerobot_keys)} camera keys)")


def generate_episodes_stats(
    parquet_dir: Path,
    camera_lerobot_keys: list[str],
    output_path: Path,
) -> None:
    """Generate ``episodes_stats.jsonl`` for the v2.1→v3.0 migration tool.

    Each line is a JSON object with per-episode statistics including a
    ``count`` field (required by the migration tool's ``aggregate_stats``).
    """
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        return

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    episodes = []

    for ep_idx in sorted(df["episode_index"].unique()):
        ep_df = df[df["episode_index"] == ep_idx]
        ep_stats: dict[str, dict[str, Any]] = {}
        length = len(ep_df)

        for col in ep_df.columns:
            sample = ep_df[col].iloc[0]
            if isinstance(sample, (int, float, np.integer, np.floating)):
                values = ep_df[col].values.astype(float)
                ep_stats[col] = {
                    "min": [float(values.min())],
                    "max": [float(values.max())],
                    "mean": [float(values.mean())],
                    "std": [float(values.std())],
                    "count": [length],
                }
            elif isinstance(sample, (list, np.ndarray)):
                arr = np.stack(ep_df[col].values)
                ep_stats[col] = {
                    "min": arr.min(axis=0).tolist(),
                    "max": arr.max(axis=0).tolist(),
                    "mean": arr.mean(axis=0).tolist(),
                    "std": arr.std(axis=0).tolist(),
                    "count": [int(arr.shape[0])],
                }

        # Image placeholders per episode
        for cam_key in camera_lerobot_keys:
            ep_stats[cam_key] = {
                "mean": [[[0.485]], [[0.456]], [[0.406]]],
                "std":  [[[0.229]], [[0.224]], [[0.225]]],
                "min":  [[[0.0]], [[0.0]], [[0.0]]],
                "max":  [[[1.0]], [[1.0]], [[1.0]]],
                "count": [length],
            }

        # v2.1→v3.0 converter expects {"episode_index": N, "stats": {...}}; use int() for JSON
        episodes.append({"episode_index": int(ep_idx), "stats": ep_stats})

    dump_jsonl(episodes, output_path)
    print(f"  Generated episodes_stats.jsonl with {len(episodes)} episodes")


def convert_hdf5_to_lerobot_franka(config: FrankaDatasetConfig) -> None:
    """Convert Franka HDF5 demonstrations to standard LeRobot format.

    Produces:
        <lerobot_data_dir>/
            data/chunk-XXX/episode_XXXXXX.parquet
            videos/chunk-XXX/observation.images.wrist/episode_XXXXXX.mp4
            videos/chunk-XXX/observation.images.table/episode_XXXXXX.mp4  (if table cam)
            meta/info.json
            meta/episodes.jsonl
            meta/tasks.jsonl
            meta/modality.json
    """
    # --- Video worker pool ---
    max_queue_size = 10
    num_workers = 4
    queue: mp.Queue = mp.Queue(maxsize=max_queue_size)
    error_queue: mp.Queue = mp.Queue()
    workers = []
    for _ in range(num_workers):
        w = mp.Process(target=write_video_job, args=(queue, error_queue, config))
        w.start()
        workers.append(w)

    # --- Open HDF5 ---
    assert Path(config.hdf5_file_path).exists(), f"HDF5 not found: {config.hdf5_file_path}"
    hdf5_handler = h5py.File(config.hdf5_file_path, "r")
    hdf5_data = hdf5_handler["data"]

    # --- Create output dirs ---
    config.lerobot_data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = config.lerobot_data_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = {config.task_index: config.language_instruction}

    total_length = 0
    example_data = None
    video_paths: dict[str, Path] = {}
    episodes_info = []

    trajectory_ids = list(hdf5_data.keys())
    print(f"Found {len(trajectory_ids)} trajectories in HDF5")

    # Determine which cameras are available
    camera_keys: list[tuple[str, str, str]] = []  # (hdf5_key, lerobot_video_key, cam_label)
    camera_keys.append(("wrist_cam", config.lerobot_keys["wrist_video"], "wrist"))
    if "table_cam" in config.hdf5_keys and "table_video" in config.lerobot_keys:
        camera_keys.append(("table_cam", config.lerobot_keys["table_video"], "table"))

    for episode_index, trajectory_id in enumerate(tqdm(trajectory_ids, desc="Converting episodes")):
        try:
            trajectory = hdf5_data[trajectory_id]

            df_ret = convert_trajectory_to_df(
                trajectory=trajectory,
                episode_index=episode_index,
                index_start=total_length,
                config=config,
            )
        except Exception as e:
            print(f"Error converting trajectory {trajectory_id}: {e}\n{traceback.format_exc()}")
            continue

        dataframe = df_ret["data"]
        length = df_ret["length"]

        # --- Write parquet ---
        episode_chunk = episode_index // config.chunks_size
        parquet_relpath = config.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
        parquet_path = config.lerobot_data_dir / parquet_relpath
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(str(parquet_path))

        total_length += length
        episodes_info.append({
            "episode_index": episode_index,
            "tasks": [tasks[t] for t in df_ret["annotation"]],
            "length": length,
        })

        # --- Enqueue video encoding for each camera ---
        for hdf5_cam_key, lerobot_video_key, cam_label in camera_keys:
            cam_hdf5_path = config.hdf5_keys.get(hdf5_cam_key)
            if cam_hdf5_path is None:
                continue

            # Navigate to the camera data in HDF5
            try:
                frames = np.array(_resolve_hdf5_dataset(trajectory, cam_hdf5_path))
            except AssertionError:
                # Camera not present in this trajectory
                continue

            # Trim trailing frame only if camera has more frames than expected
            if len(frames) > length:
                frames = frames[:length]
            assert len(frames) == length, f"Frame count mismatch: {len(frames)} vs {length}"

            video_relpath = config.video_path.format(
                episode_chunk=episode_chunk,
                video_key=lerobot_video_key,
                episode_index=episode_index,
            )
            video_path = config.lerobot_data_dir / video_relpath

            # Track first video path per camera for metadata
            if lerobot_video_key not in video_paths:
                video_paths[lerobot_video_key] = video_path

            queue.put((str(video_path), frames, config.fps))

        if example_data is None:
            example_data = df_ret

    # --- Write meta files ---

    # tasks.jsonl
    task_jsonlines = [{"task_index": ti, "task": t} for ti, t in tasks.items()]
    dump_jsonl(task_jsonlines, meta_dir / config.tasks_fname)

    # episodes.jsonl
    dump_jsonl(episodes_info, meta_dir / config.episodes_fname)

    # modality.json
    shutil.copy(str(config.modality_template_path), str(meta_dir / config.modality_fname))

    # --- Finish video workers and write info.json ---
    try:
        while not error_queue.empty():
            print(f"Video worker error: {error_queue.get()}")

        for _ in range(num_workers):
            queue.put(None)
        for w in workers:
            w.join()

        # info.json (after videos are done so ffprobe can inspect them)
        assert example_data is not None, "No episodes were successfully converted"
        info_json = generate_info(
            total_episodes=len(trajectory_ids),
            total_frames=total_length,
            total_tasks=len(tasks),
            total_videos=len(trajectory_ids) * len(camera_keys),
            total_chunks=max(1, len(trajectory_ids) // config.chunks_size),
            config=config,
            step_data=example_data["data"],
            video_paths={k: str(v) for k, v in video_paths.items()},
        )
        dump_json(info_json, meta_dir / config.info_fname, indent=4)
        print(f"Successfully generated info.json")

        # --- Generate stats.json and episodes_stats.jsonl ---
        # These are critical for the LeRobot training pipeline:
        #   - stats.json must include image/video feature keys (even as
        #     placeholders) so the training factory can inject ImageNet stats.
        #   - episodes_stats.jsonl is required by the v2.1→v3.0 migration tool.
        camera_lerobot_keys = [f"observation.images.{c[2]}" for c in camera_keys]
        data_dir = config.lerobot_data_dir / "data"

        print("\nGenerating dataset statistics...")
        generate_dataset_stats(
            parquet_dir=data_dir,
            camera_lerobot_keys=camera_lerobot_keys,
            total_frames=total_length,
            output_path=meta_dir / "stats.json",
        )
        generate_episodes_stats(
            parquet_dir=data_dir,
            camera_lerobot_keys=camera_lerobot_keys,
            output_path=meta_dir / "episodes_stats.jsonl",
        )

        hdf5_handler.close()
        print(f"\nConversion complete! LeRobot dataset at: {config.lerobot_data_dir}")
        print(f"  Episodes: {len(trajectory_ids)}")
        print(f"  Total frames: {total_length}")
        print(f"  Cameras: {[c[2] for c in camera_keys]}")

    except Exception as e:
        print(f"Error in main process: {e}")
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join()
        if hdf5_handler.id.valid:  # type: ignore[union-attr]
            hdf5_handler.close()
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Franka HDF5 demonstrations to standard LeRobot format"
    )
    parser.add_argument(
        "--yaml_file",
        help="Path to Franka YAML configuration file (e.g. franka_manip_config.yaml)",
        required=True,
    )
    args = parser.parse_args()

    config = create_config_from_yaml(args.yaml_file, FrankaDatasetConfig)

    print("\n" + "=" * 60)
    print("FRANKA LEROBOT DATASET CONFIGURATION:")
    print("=" * 60)
    for f in fields(FrankaDatasetConfig):
        if f.init:
            print(f"  {f.name}: {getattr(config, f.name)}")
    print("=" * 60 + "\n")

    convert_hdf5_to_lerobot_franka(config)
