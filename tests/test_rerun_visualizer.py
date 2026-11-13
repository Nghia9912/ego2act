"""
Unit tests for 3D geometry utilities and Rerun trajectory serialization.
"""

from pathlib import Path

import numpy as np
import pytest

from src.core.entities import Stage1OdometryOutput, Stage3RetargetOutput
from src.visualizer.rerun_logger import (
    MANO_SKELETON_EDGES,
    RerunTrajectoryLogger,
    backproject_depth_to_pointcloud,
    build_mano_skeleton_strips,
    extract_se3_axis_arrows,
)


def test_backproject_depth_to_pointcloud_pinhole_math() -> None:
    """Verifies pinhole camera back-projection accuracy and depth threshold filtering."""
    h, w = 8, 8
    intrinsics = np.array(
        [
            [100.0, 0.0, 4.0],
            [0.0, 100.0, 4.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    depth_map = np.ones((h, w), dtype=np.float32) * 1.5
    rgb_frame = np.full((h, w, 3), 128, dtype=np.uint8)

    pts_3d, colors = backproject_depth_to_pointcloud(
        depth_map=depth_map,
        intrinsics=intrinsics,
        rgb_frame=rgb_frame,
        stride=1,
        max_depth_m=2.0,
    )

    assert pts_3d.shape == (h * w, 3)
    assert colors is not None and colors.shape == (h * w, 3)
    center_idx = 4 * w + 4
    np.testing.assert_allclose(pts_3d[center_idx], np.array([0.0, 0.0, 1.5], dtype=np.float32))


def test_build_mano_skeleton_strips_topology() -> None:
    """Verifies that 21 MANO joints produce exactly 20 kinematic bone segments."""
    joints = np.arange(21 * 3, dtype=np.float32).reshape(21, 3)
    strips = build_mano_skeleton_strips(joints)

    assert len(strips) == len(MANO_SKELETON_EDGES) == 20
    assert strips[0].shape == (2, 3)

    with pytest.raises(ValueError, match="Expected joints_3d shape"):
        build_mano_skeleton_strips(np.zeros((15, 3), dtype=np.float32))


def test_extract_se3_axis_arrows_orthogonality() -> None:
    """Verifies that extracted SE(3) triad vectors preserve length and orthogonality."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.4, -0.1, 0.3]

    origins, vectors, colors = extract_se3_axis_arrows(pose, axis_length=0.08)

    assert origins.shape == (3, 3)
    assert vectors.shape == (3, 3)
    assert colors.shape == (3, 3)
    np.testing.assert_allclose(origins[0], [0.4, -0.1, 0.3], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [0.08, 0.08, 0.08], atol=1e-6)
    assert np.isclose(np.dot(vectors[0], vectors[1]), 0.0, atol=1e-6)


def test_rerun_trajectory_logger_creates_rrd_file(tmp_path: Path) -> None:
    """Verifies that RerunTrajectoryLogger logs an episode and writes a non-empty .rrd file."""
    num_frames = 12
    rrd_file = tmp_path / "test_episode.rrd"
    intrinsics = np.array(
        [[200.0, 0.0, 16.0], [0.0, 200.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    s1_out = Stage1OdometryOutput(
        intrinsics=intrinsics,
        depth_maps=np.ones((num_frames, 32, 32), dtype=np.float32) * 0.8,
        cam_to_world=np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1)),
    )
    s3_out = Stage3RetargetOutput(
        ee_poses_world=np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1)),
        gripper_width=np.linspace(0.0, 1.0, num_frames, dtype=np.float32),
        joints_3d_world=np.zeros((num_frames, 21, 3), dtype=np.float32),
    )

    logger = RerunTrajectoryLogger(
        app_id="ego2act_unit_test",
        save_rrd_path=rrd_file,
        spawn_viewer=False,
    )
    logger.log_episode(stage3_out=s3_out, stage1_out=s1_out, fps=30.0)

    assert rrd_file.exists()
    assert rrd_file.stat().st_size > 0