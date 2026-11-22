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