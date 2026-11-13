"""
3D Multimodal Spatial Visualizer powered by Rerun SDK.

Provides deterministic 3D geometry helpers (depth back-projection, MANO kinematic
skeleton edges, SE(3) coordinate triad extraction) and unified trajectory logging
to inspect egocentric RGB-D frames, 3D hand meshes, and robot end-effector poses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr

from src.core.entities import (
    Stage1OdometryOutput,
    Stage3RetargetOutput,
    Stage4VerifiedTrajectory,
)

# Standard 21-joint MANO kinematic tree connectivity (20 bone segments)
MANO_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Little finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def _set_rerun_timeline(frame_idx: int, timestamp_sec: float) -> None:
    """Sets frame sequence and timestamp using modern Rerun 0.22+/0.38+ API with fallback."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("timestamp", duration=timestamp_sec)
    else:
        rr.set_time_sequence("frame_idx", frame_idx)
        rr.set_time_seconds("timestamp", timestamp_sec)


def _log_rerun_scalar(entity_path: str, value: float) -> None:
    """Logs a scalar telemetry point across Rerun SDK versions."""
    if hasattr(rr, "Scalars"):
        rr.log(entity_path, rr.Scalars([value]))
    else:
        rr.log(entity_path, rr.Scalar(value))


def backproject_depth_to_pointcloud(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    rgb_frame: np.ndarray | None = None,
    stride: int = 4,
    max_depth_m: float = 2.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Back-projects a 2D metric depth map into a 3D point cloud in the camera frame.

    Args:
        depth_map: (H, W) float array of metric depth values in meters.
        intrinsics: (3, 3) camera intrinsic matrix K.
        rgb_frame: Optional (H, W, 3) uint8 RGB image for point coloring.
        stride: Spatial downsampling factor to keep 3D rendering responsive.
        max_depth_m: Maximum depth threshold in meters to filter background noise.

    Returns:
        points_3d: (N, 3) float32 array of 3D points in camera coordinates.
        colors: Optional (N, 3) uint8 array of RGB colors corresponding to each point.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"Expected 2D depth_map (H, W), got shape {depth_map.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) intrinsics matrix, got shape {intrinsics.shape}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    h, w = depth_map.shape
    u_grid, v_grid = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    z = depth_map[::stride, ::stride].astype(np.float32)

    valid_mask = (z > 0.05) & (z <= max_depth_m) & np.isfinite(z)
    u_valid = u_grid[valid_mask].astype(np.float32)
    v_valid = v_grid[valid_mask].astype(np.float32)
    z_valid = z[valid_mask]

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    points_3d = np.stack([x_valid, y_valid, z_valid], axis=-1)

    colors: np.ndarray | None = None
    if rgb_frame is not None:
        if rgb_frame.shape[:2] != (h, w):
            raise ValueError(
                f"rgb_frame dimensions {rgb_frame.shape[:2]} do not match depth_map {(h, w)}"
            )
        colors = rgb_frame[::stride, ::stride][valid_mask]

    return points_3d, colors


def build_mano_skeleton_strips(joints_3d: np.ndarray) -> list[np.ndarray]:
    """
    Converts (21, 3) 3D hand joint coordinates into 20 bone line segments for 3D rendering.

    Args:
        joints_3d: (21, 3) float array of 3D MANO hand joint positions.

    Returns:
        List of 20 (2, 3) float32 arrays representing parent-to-child bone segments.
    """
    if joints_3d.shape != (21, 3):
        raise ValueError(f"Expected joints_3d shape (21, 3), got {joints_3d.shape}")

    return [
        np.stack([joints_3d[parent], joints_3d[child]], axis=0).astype(np.float32)
        for parent, child in MANO_SKELETON_EDGES
    ]


def extract_se3_axis_arrows(
    pose_se3: np.ndarray,
    axis_length: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts origin, direction vectors, and RGB colors for an SE(3) coordinate frame triad.

    Args:
        pose_se3: (4, 4) homogeneous transformation matrix in SE(3).
        axis_length: Length of each coordinate axis arrow in meters.

    Returns:
        origins: (3, 3) float32 array containing the frame translation repeated 3 times.
        vectors: (3, 3) float32 array where rows are scaled X, Y, Z basis vectors.
        colors: (3, 3) uint8 array of standard RGB axis colors (Red=X, Green=Y, Blue=Z).
    """
    if pose_se3.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) SE(3) matrix, got shape {pose_se3.shape}")

    origin = pose_se3[:3, 3].astype(np.float32)
    origins = np.tile(origin, (3, 1))
    # Columns of rotation matrix R are the X, Y, Z unit axes in the world frame
    vectors = (pose_se3[:3, :3].T * axis_length).astype(np.float32)
    colors = np.array(
        [
            [255, 60, 60],   # X-axis (Normal): Red
            [60, 255, 60],   # Y-axis (Gripper closing axis): Green
            [60, 120, 255],  # Z-axis (Approach axis): Blue
        ],
        dtype=np.uint8,
    )
    return origins, vectors, colors


class RerunTrajectoryLogger:
    """Logs synchronized 3D hand skeletons, point clouds, and SE(3) trajectories to Rerun."""

    def __init__(
        self,
        app_id: str = "ego2act_inspector",
        save_rrd_path: str | Path | None = None,
        spawn_viewer: bool = False,
    ) -> None:
        self.app_id = app_id
        self.save_rrd_path = Path(save_rrd_path) if save_rrd_path is not None else None
        self.spawn_viewer = spawn_viewer

        # Note: In Rerun SDK, rr.save() redirects the active sink away from the live viewer.
        # When both saving an .rrd file and spawning the viewer, we initialize without spawning,
        # stream all data into the .rrd file, and then launch the viewer on the completed file
        # (or stream directly to the live viewer when save_rrd_path is None).
        if self.save_rrd_path is not None:
            self.save_rrd_path.parent.mkdir(parents=True, exist_ok=True)
            rr.init(self.app_id, spawn=False)
            rr.save(str(self.save_rrd_path))
        else:
            rr.init(self.app_id, spawn=self.spawn_viewer)

        # Define right-handed world coordinate system (+Z up)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def log_episode(
        self,
        stage3_out: Stage3RetargetOutput,
        stage1_out: Stage1OdometryOutput | None = None,
        stage4_out: Stage4VerifiedTrajectory | None = None,
        rgb_frames: np.ndarray | None = None,
        fps: float = 30.0,
    ) -> None:
        """
        Logs an entire trajectory episode across all available pipeline stages.

        Args:
            stage3_out: Retargeted world-frame SE(3) poses, gripper widths, and 3D hand joints.
            stage1_out: Optional visual odometry and metric depth maps.
            stage4_out: Optional B-Spline smoothed and IK-verified trajectory.
            rgb_frames: Optional (T, H, W, 3) uint8 array of raw egocentric RGB images.
            fps: Frame rate for timeline synchronization.
        """
        num_frames = stage3_out.ee_poses_world.shape[0]

        # Log static complete trajectory path for global spatial context
        raw_tcp_path = stage3_out.ee_poses_world[:, :3, 3].astype(np.float32)
        rr.log(
            "world/ee_trajectory_raw",
            rr.LineStrips3D([raw_tcp_path], colors=[[255, 180, 0]], radii=0.002),
            static=True,
        )

        if stage4_out is not None:
            smooth_tcp_path = stage4_out.smoothed_ee_poses[:, :3, 3].astype(np.float32)
            rr.log(
                "world/ee_trajectory_smoothed",
                rr.LineStrips3D([smooth_tcp_path], colors=[[0, 255, 200]], radii=0.003),
                static=True,
            )

        for t in range(num_frames):
            _set_rerun_timeline(frame_idx=t, timestamp_sec=t / fps)

            # 1. Log 3D MANO Hand Joints & Bone Segments in World Frame
            joints_w = stage3_out.joints_3d_world[t]
            bone_strips = build_mano_skeleton_strips(joints_w)
            rr.log(
                "world/hand_mano/joints",
                rr.Points3D(joints_w, colors=[[255, 215, 0]], radii=0.005),
            )
            rr.log(
                "world/hand_mano/bones",
                rr.LineStrips3D(bone_strips, colors=[[220, 220, 220]], radii=0.0025),
            )

            # 2. Log End-Effector SE(3) Triad (X=Red, Y=Green, Z=Blue)
            active_pose = (
                stage4_out.smoothed_ee_poses[t]
                if stage4_out is not None
                else stage3_out.ee_poses_world[t]
            )
            origins, vectors, colors = extract_se3_axis_arrows(active_pose, axis_length=0.06)
            rr.log(
                "world/end_effector/triad",
                rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, radii=0.003),
            )

            # 3. Log Gripper Opening State & IK Validity Telemetry
            _log_rerun_scalar("telemetry/gripper_opening", float(stage3_out.gripper_width[t]))
            if stage4_out is not None:
                _log_rerun_scalar("telemetry/ik_valid", float(stage4_out.ik_valid_mask[t]))

            # 4. Log Camera Pose & Back-projected Metric Point Cloud (if Stage 1 is provided)
            if stage1_out is not None:
                cam_pose = stage1_out.cam_to_world[t]
                pts_cam, pts_colors = backproject_depth_to_pointcloud(
                    depth_map=stage1_out.depth_maps[t],
                    intrinsics=stage1_out.intrinsics,
                    rgb_frame=rgb_frames[t] if rgb_frames is not None else None,
                    stride=4,
                )
                # Transform point cloud from camera frame C_t to static world frame W
                pts_world = (cam_pose[:3, :3] @ pts_cam.T).T + cam_pose[:3, 3]
                rr.log(
                    "world/metric_pointcloud",
                    rr.Points3D(pts_world.astype(np.float32), colors=pts_colors, radii=0.003),
                )

            if rgb_frames is not None:
                rr.log("camera/ego_rgb", rr.Image(rgb_frames[t])) 