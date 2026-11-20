# Ego2Act

**Egocentric Video-to-Action Extraction Engine** — A staged data compiler that transforms monocular first-person (POV) RGB videos into standardized robot manipulation trajectories in Hugging Face `LeRobotDataset` format.

## Overview

Training imitation learning policies (e.g., ACT, Diffusion Policy) and Vision-Language-Action (VLA) models (e.g., OpenVLA) typically requires expensive teleoperation data.

**Ego2Act** aims to bridge this gap by extracting metric 3D hand trajectories from egocentric human videos and converting them into robot-executable manipulation data.

The pipeline compensates for camera ego-motion, retargets 21-joint MANO hand representations into a parallel-jaw end-effector pose in `SE(3)`, validates kinematic feasibility through headless MuJoCo simulation, and exports structured episodes in Parquet and MP4 formats.

### Target Specifications

| Metric | Target |
|---|---|
| Spatial Accuracy | `< 1.5 cm` relative end-effector error against a static workspace frame |
| Kinematic Feasibility | `>= 90%` episode convergence on 6-DoF / 7-DoF robotic arms (e.g., Franka Emika Panda), without joint-limit violations or self-collision |
| Hardware Requirements | Single GPU with `8 GB VRAM`, using sequential model loading/unloading and `.safetensors` intermediate caching |

## Pipeline Architecture

Ego2Act consists of five sequential processing stages:

```text
[Raw POV Video (MP4)]
        |
        v
STAGE 1: Visual Odometry & Metric Depth
  - UniDepth / Depth Anything V2 (Metric)
      -> Metric Depth Maps
  - DROID-SLAM / LightGlue
      -> Camera Poses T_W^{C_t}
        |
        v
STAGE 2: 3D Hand & Object Perception
  - HaMeR (Hand Mesh Recovery)
      -> 21 MANO Joints
  - Hand-Object Contact Detector
      -> Contact States
        |
        v
STAGE 3: Coordinate Stabilization & Kinematic Retargeting
  - Camera-to-world ego-motion compensation
  - 5-Finger Hand -> Parallel-Jaw Gripper
      -> SE(3) Pose & Aperture g_t
        |
        v
STAGE 4: Trajectory Optimization & MuJoCo Verification
  - Cubic B-Spline / SLERP smoothing
  - Differential IK (Mink / MuJoCo)
  - Collision filtering
        |
        v
STAGE 5: Dataset Serialization
  - Hugging Face LeRobotDataset
      -> Apache Parquet + MP4
```

### Stage Descriptions

| Stage | Description | Output |
|---|---|---|
| 1. Visual Odometry & Metric Depth | Estimates metric depth and camera poses from monocular POV footage. | Metric depth maps and camera poses |
| 2. Hand & Object Perception | Reconstructs 3D hand geometry and detects hand-object interactions. | MANO joints and contact states |
| 3. Coordinate Stabilization & Retargeting | Compensates for camera motion and maps human hand movements to robot gripper poses. | End-effector `SE(3)` poses and gripper aperture |
| 4. Trajectory Optimization & Verification | Smooths trajectories, solves inverse kinematics, and filters physically infeasible movements. | Validated robot trajectories |
| 5. Dataset Serialization | Converts validated trajectories into the standardized LeRobot dataset format. | Parquet data and MP4 episodes |

**Memory Management:** Heavy vision models are loaded and unloaded sequentially. Intermediate outputs are cached using `.safetensors` to reduce GPU memory consumption.

## Development Setup

Ego2Act uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management and requires Python 3.11 or later.

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ego2act.git

# Navigate to the project directory
cd ego2act

# Install dependencies
uv sync
```

### Code Quality & Testing

Run linting with Ruff:

```bash
uv run ruff check .
```

Run the test suite with Pytest:

```bash
uv run pytest
```

## References & Technologies

The project builds upon the following research and open-source tools:

| Technology | Purpose | Reference |
|---|---|---|
| HaMeR | 3D hand mesh recovery | [CVPR 2024 Paper](https://geopavlakos.github.io/hamer/) |
| UniDepth | Monocular metric depth estimation | [CVPR 2024 Paper](https://lpiccinelli-eth.github.io/UniDepth/) |
| DROID-SLAM | Visual SLAM and camera pose estimation | [GitHub](https://github.com/princeton-vl/DROID-SLAM) |
| Mink | Differential inverse kinematics using MuJoCo | [GitHub](https://github.com/kevinzakka/mink) |
| Hugging Face LeRobot | Robot learning datasets and policy framework | [GitHub](https://github.com/huggingface/lerobot) |
| Rerun SDK | Multimodal 3D spatial visualization | [Documentation](https://rerun.io/) |

## License

This project is released under the [MIT License](LICENSE).