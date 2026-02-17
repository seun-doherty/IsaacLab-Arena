# env_isaaclab Dependencies Reference

This document captures the dependencies in the `env_isaaclab` conda environment used with this repo (IsaacLab-Arena), so you can replicate it elsewhere.

## Environment summary

- **Python:** 3.11.6 (conda)
- **Platform:** linux-64
- **Total packages:** ~455 (conda + pip)

---

## 1. Critical package versions (pip)

These are the versions currently in `env_isaaclab` that matter for Isaac Sim, Isaac Lab, Arena, and SmolVLA:

| Package | Version | Notes |
|---------|---------|--------|
| **torch** | 2.7.0+cu128 | CUDA 12.8 build; must match Isaac Sim |
| **torchvision** | 0.22.0+cu128 | |
| **torchaudio** | 2.7.0 | |
| **isaacsim** | 5.1.0.0 | Isaac Sim meta-package (pulls all isaacsim-*) |
| **isaaclab** | 0.54.2 | Editable from IsaacLab repo |
| **isaaclab_arena** | 1.0.0 | This repo (editable) |
| **isaaclab_assets** | 0.2.4 | Editable from IsaacLab |
| **isaaclab_contrib** | 0.0.2 | Editable from IsaacLab |
| **isaaclab_mimic** | 1.0.16 | Editable from IsaacLab |
| **isaaclab_rl** | 0.4.7 | Editable from IsaacLab |
| **isaaclab_tasks** | 0.11.12 | Editable from IsaacLab |
| **lerobot** | 0.4.4 | Editable from lerobot repo; includes SmolVLA deps |
| **transformers** | 4.57.6 | For SmolVLA |
| **accelerate** | 1.12.0 | |
| **diffusers** | 0.35.2 | |
| **datasets** | 4.1.1 | HuggingFace datasets |
| **huggingface-hub** | 0.35.3 | |
| **num2words** | 0.5.14 | SmolVLA extra |
| **av** | 15.1.0 | Video (PyAV) for LeRobot |
| **h5py** | 3.15.1 | For HDF5 datasets |
| **onnxruntime** | 1.23.2 | For ONNX policies |
| **hydra-core** | 1.3.2 | Config |
| **gymnasium** | 1.2.1 | RL envs |
| **numpy** | 1.26.0 | |
| **pyarrow** | 23.0.0 | Parquet/datasets |
| **pandas** | 3.0.0 | |
| **pyyaml** | 6.0.2 | Configs |
| **pytest** | 9.0.2 | Tests |

---

## 2. Editable install paths (current machine)

On the machine where `env_isaaclab` was captured, these packages were installed in editable mode from local paths:

- **isaaclab** → `/home/bootstrap/seun/IsaacLab/source/isaaclab`
- **isaaclab_arena** → `/home/bootstrap/seun/IsaacLab-Arena`
- **isaaclab_assets** → `/home/bootstrap/seun/IsaacLab/source/isaaclab_assets`
- **isaaclab_contrib** → `/home/bootstrap/seun/IsaacLab/source/isaaclab_contrib`
- **isaaclab_mimic** → `/home/bootstrap/seun/IsaacLab/source/isaaclab_mimic`
- **isaaclab_rl** → `/home/bootstrap/seun/IsaacLab/source/isaaclab_rl`
- **isaaclab_tasks** → `/home/bootstrap/seun/IsaacLab/source/isaaclab_tasks`
- **lerobot** → `/home/bootstrap/seun/lerobot`

When replicating elsewhere you will need:

1. **IsaacLab** repo (with `source/isaaclab`, `source/isaaclab_assets`, etc.) — can be the IsaacLab-Arena submodule `submodules/IsaacLab` or a separate clone.
2. **IsaacLab-Arena** repo (this repo).
3. **LeRobot** repo (clone from HuggingFace/lerobot) if you want editable LeRobot; otherwise `pip install "lerobot[smolvla]"` is enough.

---

## 3. Isaac Sim packages (pip)

Isaac Sim 5.1.0.0 brings in many sub-packages; all at version **5.1.0.0**:

- isaacsim, isaacsim-app, isaacsim-asset, isaacsim-benchmark, isaacsim-code-editor, isaacsim-core, isaacsim-cortex, isaacsim-example, isaacsim-extscache-kit, isaacsim-extscache-kit-sdk, isaacsim-extscache-physics, isaacsim-gui, isaacsim-kernel, isaacsim-replicator, isaacsim-rl, isaacsim-robot, isaacsim-robot-motion, isaacsim-robot-setup, isaacsim-ros1, isaacsim-ros2, isaacsim-sensor, isaacsim-storage, isaacsim-template, isaacsim-test, isaacsim-utils

Install these via NVIDIA’s recommended method (e.g. `pip install isaacsim` from NGC or the Isaac Sim pip channel), not by hand.

---

## 4. Conda / system-level deps

Important conda or system packages:

- **ffmpeg** (conda or apt) — for video encoding (converter, LeRobot).
- **Python** 3.11.
- **CUDA** 12.8 — matches `torch 2.7.0+cu128` (driver and toolkit).
- Various libs pulled by conda: e.g. libgl, libglvnd, libxcb, etc. (for GUI/display if you use Isaac Sim with a display).

---

## 5. Suggested replication order

1. **Create conda env with Python 3.11**
   ```bash
   conda create -n env_isaaclab python=3.11
   conda activate env_isaaclab
   ```

2. **Install Isaac Sim (NVIDIA)**
   - Follow [NVIDIA Isaac Sim installation](https://docs.omniverse.nvidia.com/isaacsim/) for your OS.
   - This typically installs Isaac Sim 5.x and its pip packages (including a compatible PyTorch). If you use Isaac Sim’s own Python, you may not use conda for that; the capture above uses a single conda env with Isaac Sim installed via pip.

3. **Install PyTorch with CUDA 12.8** (if not already satisfied by Isaac Sim)
   ```bash
   pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0
   ```
   (Use the index URL for CUDA 12.8 from [pytorch.org](https://pytorch.org) if needed.)

4. **Install Isaac Lab and its source packages**
   - From an IsaacLab clone (e.g. `IsaacLab-Arena/submodules/IsaacLab`):
   ```bash
   cd /path/to/IsaacLab/source
   for dir in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks; do
     pip install -e "$dir"
   done
   ```

5. **Install LeRobot with SmolVLA**
   ```bash
   pip install "lerobot[smolvla]"
   ```
   Or, for editable install:
   ```bash
   git clone https://github.com/huggingface/lerobot.git
   cd lerobot && pip install -e ".[smolvla]"
   ```

6. **Install IsaacLab-Arena (this repo)**
   ```bash
   cd /path/to/IsaacLab-Arena
   pip install -e .
   ```

7. **Other pip deps** (often pulled as dependencies; install if something is missing):
   ```bash
   pip install h5py pyyaml pytest hydra-core gymnasium onnxruntime
   pip install transformers accelerate diffusers datasets huggingface-hub
   pip install num2words av imageio imageio-ffmpeg
   ```

8. **System / conda**
   - Install **ffmpeg** (e.g. `conda install ffmpeg` or `apt install ffmpeg`).

---

## 6. Exporting exact versions from this env

An exact pip snapshot has been exported to **`docs/requirements_env_isaaclab.txt`**. It includes:

- All pinned pip package versions.
- Editable installs expressed as git URLs (e.g. `isaaclab` from IsaacLab, `isaaclab_arena` from a repo, `lerobot` from HuggingFace). On another machine you can either keep using those git URLs (same commits) or replace them with local paths, e.g. `pip install -e /path/to/IsaacLab-Arena`.

To regenerate:

```bash
conda activate env_isaaclab
pip freeze > docs/requirements_env_isaaclab.txt
```

To export conda env (packages only, no pip):

```bash
conda env export -n env_isaaclab --no-builds > env_isaaclab_conda.yml
```

---

## 7. Common replication errors

- **Torch/CUDA mismatch:** Use the same torch and CUDA version as Isaac Sim (here: 2.7.0 + cu128). Mixing system CUDA and PyTorch CUDA can cause runtime errors.
- **Isaac Lab not found:** Ensure all `isaaclab*` source packages are installed and that you use the same Python that has `isaacsim` installed.
- **LeRobot / SmolVLA import errors:** Install `lerobot[smolvla]` and ensure `transformers`, `accelerate`, `datasets` are present; check Python path.
- **Missing ffmpeg:** Install ffmpeg (conda or system) for video encoding in the HDF5→LeRobot converter.
- **Editable path wrong:** On the new machine, run `pip install -e /path/to/IsaacLab-Arena` (and same for IsaacLab, lerobot) so editable paths point to the new locations.

If you hit a specific error when replicating, share the traceback and the step (e.g. “import isaaclab”, “run converter”, “train SmolVLA”) so we can narrow it down.
