"""
Day 1 foundation test: verifies entity integrity (.safetensors I/O)
and the memory lifecycle management of StagedPipelineRunner.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.core.entities import (
    Stage1OdometryOutput,
    Stage2HandOutput,
    Stage4VerifiedTrajectory,
)
from src.core.interfaces import BasePipelineStage
from src.orchestrator.staged_runner import StagedPipelineRunner


def test_stage1_and_stage2_safetensors_roundtrip(tmp_path: Path) -> None:
    """Verify saving and reloading Stage 1 & Stage 2 data in the .safetensors format."""
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


def test_entity_shape_validation_raises_error() -> None:
    """Ensure the entity raises a ValueError immediately if the tensor shape is invalid."""
    with pytest.raises(ValueError, match="joints_3d_cam must have shape"):
        Stage2HandOutput(
            joints_3d_cam=np.zeros((10, 20, 3), dtype=np.float32),  # Wrong: 20 joints instead of 21 MANO joints
            wrist_uv=np.zeros((10, 2), dtype=np.float32),
            contact_state=np.zeros((10,), dtype=np.float32),
        )


def test_stage4_feasibility_rate_calculation(tmp_path: Path) -> None:
    """Verify the kinematic feasibility rate calculation (Kinematic Feasibility >= 90%)."""
    num_frames = 20
    mask = np.ones((num_frames,), dtype=bool)
    mask[-1] = False  # 19/20 valid frames = 95% (>= 90% DoD)

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
    """Mock stage used to test the Load -> Process -> Unload lifecycle of StagedPipelineRunner."""

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
    """Verify that StagedPipelineRunner executes in the correct order and releases models after running."""
    runner = StagedPipelineRunner("configs/pipeline_vram_8gb.yaml")
    assert runner.config["hardware"]["max_vram_gb"] == 8.0

    stage = DummyStage(runner.config)
    result = runner.run_stage(stage, inputs=21)

    assert result == 42
    assert stage._is_loaded is False
    assert stage.lifecycle_log == ["loaded", "processed", "unloaded"]