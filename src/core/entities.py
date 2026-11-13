"""
Core data entities for all 5 pipeline stages of Ego2Act.
Includes deterministic tensor shape validation and zero-copy .safetensors I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


@dataclass
class Stage1OdometryOutput:
    """Output of Stage 1: metric depth maps and camera-to-world visual odometry."""

    intrinsics: np.ndarray      # (3, 3) float32 - camera intrinsic matrix K
    depth_maps: np.ndarray      # (T, H, W) float32 - metric depth maps in meters
    cam_to_world: np.ndarray    # (T, 4, 4) float64 - camera-to-world poses in SE(3)

    def __post_init__(self) -> None:
        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must have shape (3, 3), got {self.intrinsics.shape}")
        if self.depth_maps.ndim != 3:
            raise ValueError(
                f"depth_maps must have 3 dimensions (T, H, W), got {self.depth_maps.shape}"
            )
        num_frames = self.depth_maps.shape[0]
        if self.cam_to_world.shape != (num_frames, 4, 4):
            raise ValueError(
                f"cam_to_world must have shape ({num_frames}, 4, 4), got {self.cam_to_world.shape}"
            )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "intrinsics": np.ascontiguousarray(self.intrinsics, dtype=np.float32),
                "depth_maps": np.ascontiguousarray(self.depth_maps, dtype=np.float32),
                "cam_to_world": np.ascontiguousarray(self.cam_to_world, dtype=np.float64),
            },
            str(path),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Stage1OdometryOutput:
        data = load_file(str(path))
        return cls(
            intrinsics=data["intrinsics"],
            depth_maps=data["depth_maps"],
            cam_to_world=data["cam_to_world"],
        )


@dataclass
class Stage2HandOutput:
    """Output of Stage 2: 21 3D MANO hand joints and hand-object contact states."""

    joints_3d_cam: np.ndarray   # (T, 21, 3) float32 - 21 MANO joints in camera frame
    wrist_uv: np.ndarray        # (T, 2) float32 - wrist pixel coordinates (u_w, v_w)
    contact_state: np.ndarray   # (T,) float32 - contact probability in [0, 1]

    def __post_init__(self) -> None:
        if self.joints_3d_cam.ndim != 3 or self.joints_3d_cam.shape[1:] != (21, 3):
            raise ValueError(
                f"joints_3d_cam must have shape (T, 21, 3), got {self.joints_3d_cam.shape}"
            )
        num_frames = self.joints_3d_cam.shape[0]
        if self.wrist_uv.shape != (num_frames, 2):
            raise ValueError(
                f"wrist_uv must have shape ({num_frames}, 2), got {self.wrist_uv.shape}"
            )
        if self.contact_state.shape != (num_frames,):
            raise ValueError(
                f"contact_state must have shape ({num_frames},), got {self.contact_state.shape}"
            )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "joints_3d_cam": np.ascontiguousarray(self.joints_3d_cam, dtype=np.float32),
                "wrist_uv": np.ascontiguousarray(self.wrist_uv, dtype=np.float32),
                "contact_state": np.ascontiguousarray(self.contact_state, dtype=np.float32),
            },
            str(path),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Stage2HandOutput:
        data = load_file(str(path))
        return cls(
            joints_3d_cam=data["joints_3d_cam"],
            wrist_uv=data["wrist_uv"],
            contact_state=data["contact_state"],
        )


@dataclass
class Stage3RetargetOutput:
    """Output of Stage 3: world-frame end-effector trajectory and gripper opening."""

    ee_poses_world: np.ndarray   # (T, 4, 4) float64 - T_{EE}(t) transforms in SE(3)
    gripper_width: np.ndarray    # (T,) float32 - normalized gripper opening in [0, 1]
    joints_3d_world: np.ndarray  # (T, 21, 3) float32 - stabilized world-frame hand joints

    def __post_init__(self) -> None:
        num_frames = self.ee_poses_world.shape[0]
        if self.ee_poses_world.shape != (num_frames, 4, 4):
            raise ValueError(
                f"ee_poses_world must have shape (T, 4, 4), got {self.ee_poses_world.shape}"
            )
        if self.gripper_width.shape != (num_frames,):
            raise ValueError(
                f"gripper_width must have shape ({num_frames},), got {self.gripper_width.shape}"
            )
        if self.joints_3d_world.shape != (num_frames, 21, 3):
            raise ValueError(
                f"joints_3d_world must have shape ({num_frames}, 21, 3), "
                f"got {self.joints_3d_world.shape}"
            )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "ee_poses_world": np.ascontiguousarray(self.ee_poses_world, dtype=np.float64),
                "gripper_width": np.ascontiguousarray(self.gripper_width, dtype=np.float32),
                "joints_3d_world": np.ascontiguousarray(self.joints_3d_world, dtype=np.float32),
            },
            str(path),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Stage3RetargetOutput:
        data = load_file(str(path))
        return cls(
            ee_poses_world=data["ee_poses_world"],
            gripper_width=data["gripper_width"],
            joints_3d_world=data["joints_3d_world"],
        )


@dataclass
class Stage4VerifiedTrajectory:
    """Output of Stage 4: B-Spline smoothed and MuJoCo IK-verified robot trajectory."""

    smoothed_ee_poses: np.ndarray  # (T, 4, 4) float64 - smoothed T_{EE}(t) poses
    joint_positions: np.ndarray    # (T, D) float32 - solved robot joint angles q_t
    gripper_width: np.ndarray      # (T,) float32 - normalized gripper command in [0, 1]
    ik_valid_mask: np.ndarray      # (T,) bool/uint8 - valid IK & collision-free flags

    @property
    def feasibility_rate(self) -> float:
        """Fraction of timesteps satisfying kinematic feasibility (target >= 0.90)."""
        return float(np.mean(self.ik_valid_mask.astype(np.float32)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "smoothed_ee_poses": np.ascontiguousarray(self.smoothed_ee_poses, dtype=np.float64),
                "joint_positions": np.ascontiguousarray(self.joint_positions, dtype=np.float32),
                "gripper_width": np.ascontiguousarray(self.gripper_width, dtype=np.float32),
                "ik_valid_mask": np.ascontiguousarray(self.ik_valid_mask, dtype=np.uint8),
            },
            str(path),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Stage4VerifiedTrajectory:
        data = load_file(str(path))
        return cls(
            smoothed_ee_poses=data["smoothed_ee_poses"],
            joint_positions=data["joint_positions"],
            gripper_width=data["gripper_width"],
            ik_valid_mask=data["ik_valid_mask"].astype(bool),
        )