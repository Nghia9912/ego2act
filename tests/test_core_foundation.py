"""
Unit tests for core data entities (.safetensors serialization) and StagedPipelineRunner.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.core.entities import (
    Stage1OdometryOutput,
    Stage2HandOutput,
    Stage3RetargetOutput,
    Stage4VerifiedTrajectory,
)
from src.core.interfaces import BasePipelineStage
from src.orchestrator.staged_runner import StagedPipelineRunner


def test_stage1_to_stage3_safetensors_roundtrip(tmp_path: Path) -> None:
    """Verifies lossless .safetensors serialization for Stage 1, 2, and 3 entities."""
    num_frames, height, width = 10, 64, 64

    s1_out = Stage1OdometryOutput(
        intrinsics=np.eye(3, dtype=np.float32),
        depth_maps=np.ones((num_frames, height, width), dtype=np.float32) * 0.65,
        cam_to_world=np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1)),
    )
    s1_file = tmp_path / "episode_000_s1.safetensors"
    s1_out.save(s1_file)

    s1_loaded = Stage1OdometryOutput.load(s1_file)
    np.testing.assert_allclose(s1_loaded.intrinsics, s1_out.intrinsics)
    np.testing.assert_allclose(s1_loaded.depth_maps, s1_out.depth_maps)
    np.testing.assert_allclose(s1_loaded.cam_to_world, s1_out.cam_to_world)

    s2_out = Stage2HandOutput(
        joints_3d_cam=np.zeros((num_frames, 21, 3), dtype=np.float32),
        wrist_uv=np.full((num_frames, 2), 32.0, dtype=np.float32),
        contact_state=np.linspace(0.0, 1.0, num_frames, dtype=np.float32),
    )
    s2_file = tmp_path / "episode_000_s2.safetensors"
    s2_out.save(s2_file)

    s2_loaded = Stage2HandOutput.load(s2_file)
    np.testing.assert_allclose(s2_loaded.contact_state, s2_out.contact_state)

    s3_out = Stage3RetargetOutput(
        ee_poses_world=np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1)),
        gripper_width=np.linspace(0.1, 0.9, num_frames, dtype=np.float32),
        joints_3d_world=np.zeros((num_frames, 21, 3), dtype=np.float32),
    )
    s3_file = tmp_path / "episode_000_s3.safetensors"
    s3_out.save(s3_file)

    s3_loaded = Stage3RetargetOutput.load(s3_file)
    np.testing.assert_allclose(s3_loaded.ee_poses_world, s3_out.ee_poses_world)
    np.testing.assert_allclose(s3_loaded.gripper_width, s3_out.gripper_width)


def test_entity_shape_validation_raises_error() -> None:
    """Verifies that malformed tensor dimensions immediately raise ValueError."""
    with pytest.raises(ValueError, match="joints_3d_cam must have shape"):
        Stage2HandOutput(
            # Invalid: 20 joints instead of 21 MANO joints
            joints_3d_cam=np.zeros((10, 20, 3), dtype=np.float32),
            wrist_uv=np.zeros((10, 2), dtype=np.float32),
            contact_state=np.zeros((10,), dtype=np.float32),
        )


def test_stage4_feasibility_rate_calculation(tmp_path: Path) -> None:
    """Verifies kinematic feasibility rate computation and boolean mask roundtrip."""
    num_frames = 20
    mask = np.ones((num_frames,), dtype=bool)
    mask[-1] = False

    s4_out = Stage4VerifiedTrajectory(
        smoothed_ee_poses=np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1)),
        joint_positions=np.zeros((num_frames, 7), dtype=np.float32),
        gripper_width=np.ones((num_frames,), dtype=np.float32) * 0.5,
        ik_valid_mask=mask,
    )
    assert np.isclose(s4_out.feasibility_rate, 0.95)

    s4_file = tmp_path / "episode_000_s4.safetensors"
    s4_out.save(s4_file)
    s4_loaded = Stage4VerifiedTrajectory.load(s4_file)
    assert s4_loaded.ik_valid_mask.dtype == bool
    assert np.isclose(s4_loaded.feasibility_rate, 0.95)


class DummyStage(BasePipelineStage[int, int]):
    """Mock stage to verify Load -> Process -> Unload execution order."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.lifecycle_log: list[str] = []

    @property
    def stage_name(self) -> str:
        return "dummy_verification_stage"

    def load_models(self) -> None:
        super().load_models()
        self.lifecycle_log.append("loaded")

    def process(self, inputs: int) -> int:
        assert self._is_loaded is True
        self.lifecycle_log.append("processed")
        return inputs * 2

    def unload_models(self) -> None:
        super().unload_models()
        self.lifecycle_log.append("unloaded")


def test_staged_pipeline_runner_lifecycle() -> None:
    """Verifies that StagedPipelineRunner executes stages and unloads models deterministically."""
    runner = StagedPipelineRunner("configs/pipeline_vram_8gb.yaml")
    assert runner.config["hardware"]["max_vram_gb"] == 8.0

    stage = DummyStage(runner.config)
    result = runner.run_stage(stage, inputs=21)

    assert result == 42
    assert stage._is_loaded is False
    assert stage.lifecycle_log == ["loaded", "processed", "unloaded"]