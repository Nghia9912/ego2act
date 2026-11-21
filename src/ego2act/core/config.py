"""Typed configuration schemas and YAML loader for the Ego2Act pipeline."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class HardwareConfig(BaseModel):
    """GPU and memory lifecycle constraints."""

    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    max_vram_gb: float = Field(default=8.0, gt=0.0)
    sequential_offload: bool = True
    clear_cache_between_stages: bool = True


class IOConfig(BaseModel):
    """Filesystem paths and intermediate serialization settings."""

    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/output")
    cache_format: Literal["safetensors", "zarr"] = "safetensors"
    target_fps: int = Field(default=30, gt=0, le=120)


class Stage1OdometryConfig(BaseModel):
    """Stage 1 visual odometry and metric depth settings."""

    depth_backend: Literal["unidepth", "depth_anything_v2"] = "unidepth"
    odometry_backend: Literal["droid_slam", "lightglue"] = "droid_slam"
    batch_size: int = Field(default=4, ge=1)


class Stage2HandConfig(BaseModel):
    """Stage 2 3D hand mesh recovery and contact detection settings."""

    hand_backend: Literal["hamer"] = "hamer"
    batch_size: int = Field(default=4, ge=1)
    contact_threshold_m: float = Field(default=0.035, gt=0.0)
    max_occlusion_gap_frames: int = Field(default=5, ge=0)


class Stage3RetargetingConfig(BaseModel):
    """Stage 3 coordinate stabilization and parallel-jaw retargeting settings."""

    depth_patch_radius_px: int = Field(default=3, ge=0)
    gripper_min_aperture_m: float = Field(default=0.015, ge=0.0)
    gripper_max_aperture_m: float = Field(default=0.085, gt=0.0)
    orientation_representation: Literal["quaternion", "rot6d"] = "quaternion"

    @model_validator(mode="after")
    def _validate_aperture_bounds(self) -> "Stage3RetargetingConfig":
        if self.gripper_max_aperture_m <= self.gripper_min_aperture_m:
            msg = "gripper_max_aperture_m must be strictly greater than gripper_min_aperture_m"
            raise ValueError(msg)
        return self


class Stage4KinematicsConfig(BaseModel):
    """Stage 4 trajectory smoothing and MuJoCo/Mink IK verification settings."""

    robot_embodiment: str = "franka_emika_panda"
    mjcf_path: Path = Path("configs/robot_embodiments/franka_emika_panda/scene.xml")
    smoother: Literal["bspline", "savgol"] = "bspline"
    bspline_smoothing_factor: float = Field(default=0.002, ge=0.0)
    slerp_cutoff_hz: float = Field(default=6.0, gt=0.0)
    max_position_error_m: float = Field(default=0.015, gt=0.0)
    min_ik_convergence_ratio: float = Field(default=0.90, gt=0.0, le=1.0)
    reject_self_collision: bool = True


class Stage5ExportConfig(BaseModel):
    """Stage 5 LeRobotDataset serialization settings."""

    dataset_format: Literal["lerobot", "rlds"] = "lerobot"
    video_codec: Literal["libx264", "libsvtav1"] = "libx264"
    include_joint_space_metadata: bool = True


class PipelineConfig(BaseModel):
    """Root configuration schema aggregating all five pipeline stages."""

    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    stage1_odometry: Stage1OdometryConfig = Field(default_factory=Stage1OdometryConfig)
    stage2_hand: Stage2HandConfig = Field(default_factory=Stage2HandConfig)
    stage3_retargeting: Stage3RetargetingConfig = Field(default_factory=Stage3RetargetingConfig)
    stage4_kinematics: Stage4KinematicsConfig = Field(default_factory=Stage4KinematicsConfig)
    stage5_export: Stage5ExportConfig = Field(default_factory=Stage5ExportConfig)


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle)

    if raw_data is None:
        return PipelineConfig()
    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected YAML mapping at top level in {path}")

    return PipelineConfig.model_validate(raw_data)