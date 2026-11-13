"""
Synthetic 3D Trajectory Demo for Rerun Visualizer.

Generates a smooth spiral reach-and-grasp hand trajectory with metric depth point clouds
and streams it to an `.rrd` recording file (or spawns the native Rerun 3D Viewer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.entities import Stage1OdometryOutput, Stage3RetargetOutput  # noqa: E402
from src.visualizer.rerun_logger import RerunTrajectoryLogger  # noqa: E402


def create_canonical_mano_hand() -> np.ndarray:
    """Returns a canonical (21, 3) right-hand joint template in meters relative to wrist."""
    joints = np.zeros((21, 3), dtype=np.float32)
    # Thumb (1..4)
    joints[1:5] = [
        [-0.02, 0.02, 0.02],
        [-0.03, 0.04, 0.03],
        [-0.04, 0.05, 0.04],
        [-0.04, 0.06, 0.05],
    ]
    # Index (5..8)
    joints[5:9] = [
        [0.00, 0.02, 0.07],
        [0.00, 0.02, 0.09],
        [0.00, 0.02, 0.11],
        [0.00, 0.02, 0.13],
    ]
    # Middle (9..12)
    joints[9:13] = [
        [0.00, 0.00, 0.075],
        [0.00, 0.00, 0.10],
        [0.00, 0.00, 0.12],
        [0.00, 0.00, 0.14],
    ]
    # Ring (13..16)
    joints[13:17] = [
        [0.00, -0.02, 0.07],
        [0.00, -0.02, 0.09],
        [0.00, -0.02, 0.11],
        [0.00, -0.02, 0.125],
    ]
    # Little (17..20)
    joints[17:21] = [
        [0.00, -0.04, 0.06],
        [0.00, -0.04, 0.08],
        [0.00, -0.04, 0.095],
        [0.00, -0.04, 0.11],
    ]
    return joints


def generate_synthetic_episode(
    num_frames: int = 90,
) -> tuple[Stage1OdometryOutput, Stage3RetargetOutput]:
    """Synthesizes a 3-second (90-frame) reach-and-pinch trajectory in the world frame."""
    t_lin = np.linspace(0.0, 2.0 * np.pi, num_frames)
    base_hand = create_canonical_mano_hand()

    ee_poses = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
    joints_world = np.zeros((num_frames, 21, 3), dtype=np.float32)
    gripper_width = (0.5 * (1.0 + np.cos(t_lin))).astype(np.float32)

    for i, phase in enumerate(t_lin):
        wrist_pos = np.array(
            [
                0.45 + 0.10 * np.cos(phase),
                0.15 * np.sin(phase),
                0.25 + 0.05 * np.sin(2.0 * phase),
            ],
            dtype=np.float32,
        )
        hand_t = base_hand.copy()
        pinch_offset = (1.0 - gripper_width[i]) * 0.025
        hand_t[4, 1] -= pinch_offset
        hand_t[8, 1] += pinch_offset

        joints_world[i] = hand_t + wrist_pos
        ee_poses[i, :3, 3] = 0.5 * (joints_world[i, 4] + joints_world[i, 8])

    intrinsics = np.array(
        [[250.0, 0.0, 32.0], [0.0, 250.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    depth_maps = np.ones((num_frames, 64, 64), dtype=np.float32) * 0.75
    cam_to_world = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))

    s1_out = Stage1OdometryOutput(
        intrinsics=intrinsics,
        depth_maps=depth_maps,
        cam_to_world=cam_to_world,
    )
    s3_out = Stage3RetargetOutput(
        ee_poses_world=ee_poses,
        gripper_width=gripper_width,
        joints_3d_world=joints_world,
    )
    return s1_out, s3_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ego2Act 3D Rerun Visualizer Demo.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/synthetic_demo.rrd"),
        help="Output path for the Rerun .rrd recording file.",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="Stream live to the interactive Rerun Viewer desktop window.",
    )
    args = parser.parse_args()

    s1_out, s3_out = generate_synthetic_episode(num_frames=90)

    if args.spawn:
        # Stream directly to the live desktop Rerun Viewer
        live_logger = RerunTrajectoryLogger(
            app_id="ego2act_synthetic_demo",
            save_rrd_path=None,
            spawn_viewer=True,
        )
        live_logger.log_episode(stage3_out=s3_out, stage1_out=s1_out, fps=30.0)
        print("Streamed 90 frames live to the Rerun Viewer window.")
    else:
        # Save deterministic .rrd recording to disk
        file_logger = RerunTrajectoryLogger(
            app_id="ego2act_synthetic_demo",
            save_rrd_path=args.output,
            spawn_viewer=False,
        )
        file_logger.log_episode(stage3_out=s3_out, stage1_out=s1_out, fps=30.0)
        print(f"Saved 3D trajectory recording to: {args.output}")


if __name__ == "__main__":
    main()
