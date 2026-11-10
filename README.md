# Ego2Act: Egocentric Video-to-Action Extraction Engine

**Ego2Act** is a data compiler that transforms first-person POV (egocentric) worker videos into standardized robot-control trajectory datasets for [Hugging Face LeRobot](https://github.com/huggingface/lerobot), validated by inverse kinematics (IK) and direct physical collision checks on MuJoCo.

## 5-Stage Pipeline Architecture
1. **Stage 1 (`s1_ego_odometry`):** Estimates the absolute depth map ([UniDepth](https://github.com/lpiccinelli-eth/unidepth)) and camera trajectory ([DROID-SLAM](https://github.com/princeton-vl/droid-slam)).
2. **Stage 2 (`s2_hand_reconstruction`):** Reconstructs the 21 3D MANO hand joints ([HaMeR](https://github.com/geopavlakos/hamer)) and detects contact states.
3. **Stage 3 (`s3_retargeting`):** Removes human head motion, aligns median depth, and maps to the robot gripper coordinate frame $SE(3)$ via Gram-Schmidt orthogonalization.
4. **Stage 4 (`s4_kinematic_solver`):** Smooths the trajectory (Cubic B-Spline + SLERP) and solves inverse kinematics using [Mink](https://github.com/kevinzakka/mink) on MuJoCo.
5. **Stage 5 (`s5_lerobot_exporter`):** Packages the standard `LeRobotDataset` format (Apache Parquet + MP4).

## Development Progress (6-Week Engineering Roadmap)
- [x] **Week 1 - Day 1:** Initializes the Clean Architecture, `.safetensors` data contract (`entities.py`), `interfaces.py` abstraction, and VRAM-release orchestration (`staged_runner.py`).
- [ ] **Week 1 - Days 2–5:** Integrates the 3D visualization stack [Rerun SDK](https://github.com/rerun-io/rerun) (`rerun_logger.py`) with simulated trajectory data.
