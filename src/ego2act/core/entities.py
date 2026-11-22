"""Core domain entities shared across all Ego2Act pipeline stages."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class CameraPose:
    """Camera extrinsics (SE(3) camera-to-world) and pinhole intrinsics."""

    transform_world_cam: NDArray[np.float32]
    intrinsics: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.transform_world_cam.shape != (4, 4):
            raise ValueError(
                f"Expected transform_world_cam shape (4, 4), got {self.transform_world_cam.shape}"
            )
        if self.intrinsics.shape != (3, 3):
            raise ValueError(f"Expected intrinsics shape (3, 3), got {self.intrinsics.shape}")


@dataclass(slots=True)
class FrameState:
    """Per-frame observation state produced by Stage 1."""

    frame_index: int
    timestamp_s: float
    rgb: NDArray[np.uint8] | None = None
    depth_metric_m: NDArray[np.float32] | None = None
    camera_pose: CameraPose | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        if self.timestamp_s < 0.0:
            raise ValueError(f"timestamp_s must be non-negative, got {self.timestamp_s}")
        if self.rgb is not None and (self.rgb.ndim != 3 or self.rgb.shape[2] != 3):
            raise ValueError(f"Expected rgb shape (H, W, 3), got {self.rgb.shape}")
        if self.depth_metric_m is not None and self.depth_metric_m.ndim != 2:
            raise ValueError(
                f"Expected depth_metric_m shape (H, W), got {self.depth_metric_m.shape}"
            )
@dataclass(slots=True)
class HandObservation:
    """3D MANO hand skeleton and contact state extracted by Stage 2."""

    frame_index: int
    joints_3d_cam: NDArray[np.float32]
    wrist_uv: tuple[float, float]
    is_right: bool = True
    in_contact: bool = False
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        if self.joints_3d_cam.shape != (21, 3):
            raise ValueError(
                f"Expected joints_3d_cam shape (21, 3), got {self.joints_3d_cam.shape}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(slots=True)
class SE3Trajectory:
    """World-stabilized parallel-jaw end-effector trajectory in SE(3)."""

    timestamps_s: NDArray[np.float64]
    positions_m: NDArray[np.float32]
    orientations_xyzw: NDArray[np.float32]
    gripper_openness: NDArray[np.float32]

    def __post_init__(self) -> None:
        num_steps = len(self.timestamps_s)
        if self.timestamps_s.ndim != 1 or num_steps == 0:
            raise ValueError(
                f"timestamps_s must be non-empty 1D array, got {self.timestamps_s.shape}"
            )
        if self.positions_m.shape != (num_steps, 3):
            raise ValueError(
                f"Expected positions_m shape ({num_steps}, 3), got {self.positions_m.shape}"
            )
        if self.orientations_xyzw.shape != (num_steps, 4):
            raise ValueError(
                f"Expected orientations_xyzw shape ({num_steps}, 4), "
                f"got {self.orientations_xyzw.shape}"
            )
        if self.gripper_openness.shape != (num_steps,):
            raise ValueError(
                f"Expected gripper_openness shape ({num_steps},), "
                f"got {self.gripper_openness.shape}"
            )

    def __len__(self) -> int:
        return int(len(self.timestamps_s))
@dataclass(slots=True)
class RobotAction:
    """Standardized robot proprioceptive states, delta SE(3) actions, and optional IK joints."""

    proprio_state: NDArray[np.float32]
    actions_7d: NDArray[np.float32]
    joint_positions_rad: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if self.proprio_state.ndim != 2 or self.proprio_state.shape[1] < 7:
            raise ValueError(
                f"Expected proprio_state shape (T, D>=7), got {self.proprio_state.shape}"
            )
        num_steps = self.proprio_state.shape[0]
        if self.actions_7d.shape != (num_steps, 7):
            raise ValueError(
                f"Expected actions_7d shape ({num_steps}, 7), got {self.actions_7d.shape}"
            )
        if self.joint_positions_rad is not None and (
            self.joint_positions_rad.ndim != 2 or self.joint_positions_rad.shape[0] != num_steps
        ):
            raise ValueError(
                f"Expected joint_positions_rad shape ({num_steps}, N_joints), "
                f"got {self.joint_positions_rad.shape}"
            )

    def __len__(self) -> int:
        return int(self.proprio_state.shape[0])


@dataclass(slots=True)
class EpisodeMetadata:
    """Quality metrics and validity flags recorded after Stage 4 MuJoCo verification."""

    episode_id: str
    source_video: str
    fps: float
    num_frames: int
    robot_embodiment: str
    ik_convergence_ratio: float
    mean_position_error_m: float
    has_self_collision: bool
    is_valid: bool

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if not 0.0 <= self.ik_convergence_ratio <= 1.0:
            raise ValueError(
                f"ik_convergence_ratio must be in [0, 1], got {self.ik_convergence_ratio}"
            )
        if self.mean_position_error_m < 0.0:
            raise ValueError(
                f"mean_position_error_m must be non-negative, got {self.mean_position_error_m}"
            )