"""Unit tests for core entities, config validation, io layer, and StagedRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ego2act.core.config import PipelineConfig, load_pipeline_config
from ego2act.core.entities import (
    CameraPose,
    EpisodeMetadata,
    FrameState,
    HandObservation,
    RobotAction,
    SE3Trajectory,
)
from ego2act.core.interfaces import PipelineStage
from ego2act.core.io import load_stage_cache, save_stage_cache, verify_cache_shapes
from ego2act.orchestrator.staged_runner import StagedRunner

# ---------------------------------------------------------------------------
# Entity construction tests
# ---------------------------------------------------------------------------


class TestCameraPose:
    def test_valid_construction(self) -> None:
        pose = CameraPose(
            transform_world_cam=np.eye(4, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
        )
        assert pose.transform_world_cam.shape == (4, 4)

    def test_invalid_transform_shape(self) -> None:
        with pytest.raises(ValueError, match="transform_world_cam"):
            CameraPose(
                transform_world_cam=np.eye(3, dtype=np.float32),
                intrinsics=np.eye(3, dtype=np.float32),
            )

    def test_invalid_intrinsics_shape(self) -> None:
        with pytest.raises(ValueError, match="intrinsics"):
            CameraPose(
                transform_world_cam=np.eye(4, dtype=np.float32),
                intrinsics=np.eye(4, dtype=np.float32),
            )


class TestFrameState:
    def test_valid_minimal(self) -> None:
        fs = FrameState(frame_index=0, timestamp_s=0.0)
        assert fs.rgb is None

    def test_negative_frame_index(self) -> None:
        with pytest.raises(ValueError, match="frame_index"):
            FrameState(frame_index=-1, timestamp_s=0.0)

    def test_invalid_rgb_shape(self) -> None:
        with pytest.raises(ValueError, match="rgb"):
            FrameState(frame_index=0, timestamp_s=0.0, rgb=np.zeros((10, 10), dtype=np.uint8))


class TestHandObservation:
    def test_valid(self) -> None:
        ho = HandObservation(
            frame_index=0,
            joints_3d_cam=np.zeros((21, 3), dtype=np.float32),
            wrist_uv=(320.0, 240.0),
        )
        assert ho.is_right is True

    def test_invalid_joints_shape(self) -> None:
        with pytest.raises(ValueError, match="joints_3d_cam"):
            HandObservation(
                frame_index=0,
                joints_3d_cam=np.zeros((20, 3), dtype=np.float32),
                wrist_uv=(0.0, 0.0),
            )

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            HandObservation(
                frame_index=0,
                joints_3d_cam=np.zeros((21, 3), dtype=np.float32),
                wrist_uv=(0.0, 0.0),
                confidence=1.5,
            )


class TestSE3Trajectory:
    def test_valid(self) -> None:
        t = 10
        traj = SE3Trajectory(
            timestamps_s=np.arange(t, dtype=np.float64),
            positions_m=np.zeros((t, 3), dtype=np.float32),
            orientations_xyzw=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (t, 1)),
            gripper_openness=np.ones(t, dtype=np.float32),
        )
        assert len(traj) == t

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="positions_m"):
            SE3Trajectory(
                timestamps_s=np.arange(5, dtype=np.float64),
                positions_m=np.zeros((4, 3), dtype=np.float32),
                orientations_xyzw=np.zeros((5, 4), dtype=np.float32),
                gripper_openness=np.zeros(5, dtype=np.float32),
            )


class TestRobotAction:
    def test_valid(self) -> None:
        t = 10
        ra = RobotAction(
            proprio_state=np.zeros((t, 7), dtype=np.float32),
            actions_7d=np.zeros((t, 7), dtype=np.float32),
        )
        assert len(ra) == t
        assert ra.joint_positions_rad is None

    def test_with_joint_positions(self) -> None:
        t = 10
        ra = RobotAction(
            proprio_state=np.zeros((t, 7), dtype=np.float32),
            actions_7d=np.zeros((t, 7), dtype=np.float32),
            joint_positions_rad=np.zeros((t, 7), dtype=np.float32),
        )
        assert ra.joint_positions_rad is not None

    def test_mismatch(self) -> None:
        with pytest.raises(ValueError, match="actions_7d"):
            RobotAction(
                proprio_state=np.zeros((10, 7), dtype=np.float32),
                actions_7d=np.zeros((9, 7), dtype=np.float32),
            )


class TestEpisodeMetadata:
    def test_valid(self) -> None:
        em = EpisodeMetadata(
            episode_id="ep_000",
            source_video="demo.mp4",
            fps=30.0,
            num_frames=150,
            robot_embodiment="franka_emika_panda",
            ik_convergence_ratio=0.95,
            mean_position_error_m=0.008,
            has_self_collision=False,
            is_valid=True,
        )
        assert em.is_valid

    def test_negative_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            EpisodeMetadata(
                episode_id="x",
                source_video="x.mp4",
                fps=-1.0,
                num_frames=10,
                robot_embodiment="panda",
                ik_convergence_ratio=0.5,
                mean_position_error_m=0.01,
                has_self_collision=False,
                is_valid=False,
            )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_construction(self) -> None:
        cfg = PipelineConfig()
        assert cfg.hardware.max_vram_gb == 8.0
        assert cfg.io.cache_format == "safetensors"
        assert (
            cfg.stage3_retargeting.gripper_max_aperture_m
            > cfg.stage3_retargeting.gripper_min_aperture_m
        )

    def test_load_yaml(self, tmp_path: Path) -> None:
        yaml_text = """\
hardware:
  max_vram_gb: 6.0
  precision: "bf16"
io:
  target_fps: 15
"""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml_text, encoding="utf-8")

        cfg = load_pipeline_config(cfg_file)
        assert cfg.hardware.max_vram_gb == 6.0
        assert cfg.hardware.precision == "bf16"
        assert cfg.io.target_fps == 15

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("nonexistent.yaml")

    def test_invalid_gripper_range(self) -> None:
        with pytest.raises(ValueError, match="gripper_max_aperture_m"):
            PipelineConfig(
                stage3_retargeting={
                    "gripper_min_aperture_m": 0.1,
                    "gripper_max_aperture_m": 0.05,
                }
            )


# ---------------------------------------------------------------------------
# IO (safetensors cache) tests
# ---------------------------------------------------------------------------


class TestStageCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        tensors = {
            "positions": np.random.randn(10, 3).astype(np.float32),
            "flags": np.array([True, False, True]),
        }
        path = save_stage_cache(tensors, tmp_path / "ep.safetensors")
        loaded = load_stage_cache(path)
        assert set(loaded.keys()) == {"positions", "flags"}
        np.testing.assert_array_equal(loaded["positions"], tensors["positions"])

    def test_save_empty_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            save_stage_cache({}, tmp_path / "empty.safetensors")

    def test_load_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_stage_cache("no_such_file.safetensors")

    def test_verify_shapes_pass(self) -> None:
        tensors = {"a": np.zeros((5, 3)), "b": np.zeros((5,))}
        assert verify_cache_shapes(tensors, {"a": (-1, 3), "b": (-1,)}) == []

    def test_verify_shapes_mismatch(self) -> None:
        tensors = {"a": np.zeros((5, 4))}
        errors = verify_cache_shapes(tensors, {"a": (-1, 3)})
        assert len(errors) == 1
        assert "axis 1" in errors[0]

    def test_verify_shapes_missing_key(self) -> None:
        errors = verify_cache_shapes({}, {"x": (10,)})
        assert any("Missing" in e for e in errors)


# ---------------------------------------------------------------------------
# PipelineStage interface tests
# ---------------------------------------------------------------------------


class _MockStage(PipelineStage):
    """Minimal concrete stage that records lifecycle calls."""

    def __init__(self, stage_name: str = "mock", output: dict[str, Any] | None = None) -> None:
        self._name = stage_name
        self._output = output or {}
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def load_model(self) -> None:
        self.calls.append("load")

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("run")
        return self._output

    def unload(self) -> None:
        self.calls.append("unload")


class TestPipelineStageInterface:
    def test_cache_key_default(self) -> None:
        stage = _MockStage("depth")
        assert stage.cache_key("ep_001") == "ep_001_depth"

    def test_cache_path(self, tmp_path: Path) -> None:
        stage = _MockStage("hands")
        p = stage.cache_path(tmp_path, "ep_002")
        assert p.name == "ep_002_hands.safetensors"
        assert p.parent == tmp_path


# ---------------------------------------------------------------------------
# StagedRunner orchestration tests
# ---------------------------------------------------------------------------


class TestStagedRunner:
    def test_lifecycle_order(self, tmp_path: Path) -> None:
        """Runner must call load → run → unload for each stage."""
        s1 = _MockStage("s1", {"a": np.array([1.0])})
        s2 = _MockStage("s2", {"b": np.array([2.0])})

        runner = StagedRunner([s1, s2], cache_dir=tmp_path, skip_cached=False)
        result = runner.run("ep_test")

        assert s1.calls == ["load", "run", "unload"]
        assert s2.calls == ["load", "run", "unload"]
        assert "a" in result
        assert "b" in result

    def test_cache_skip(self, tmp_path: Path) -> None:
        """When a cache file exists, the stage must NOT be executed."""
        stage = _MockStage("cached_stage", {"val": np.array([42.0])})

        # Pre-populate the cache.
        cache_file = stage.cache_path(tmp_path, "ep_skip")
        save_stage_cache({"val": np.array([99.0])}, cache_file)

        runner = StagedRunner([stage], cache_dir=tmp_path, skip_cached=True)
        result = runner.run("ep_skip")

        assert stage.calls == []  # stage was never invoked
        np.testing.assert_array_equal(result["val"], np.array([99.0]))

    def test_unload_called_on_error(self, tmp_path: Path) -> None:
        """Even when run() raises, unload() must still be called."""

        class _FailStage(_MockStage):
            def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
                self.calls.append("run")
                raise RuntimeError("boom")

        stage = _FailStage("fail")
        runner = StagedRunner([stage], cache_dir=tmp_path, skip_cached=False)

        with pytest.raises(RuntimeError, match="boom"):
            runner.run("ep_fail")

        assert "unload" in stage.calls

    def test_empty_stages_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            StagedRunner([])
