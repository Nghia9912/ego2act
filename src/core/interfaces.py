"""
Defines the abstract base classes (ABCs) for the processing layers in the Staged Pipeline architecture.
Ensures each stage has a clear lifecycle: load_model() -> process() -> unload_model().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BasePipelineStage(ABC, Generic[InputT, OutputT]):
    """Base class for all stages in Ego2Act."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._is_loaded: bool = False

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Stage identifier name (for example: 'stage1_ego_odometry')."""

    def load_models(self) -> None:
        """Loads model weights onto the GPU/CPU (override only in stages that use deep learning)."""
        self._is_loaded = True

    @abstractmethod
    def process(self, inputs: InputT) -> OutputT:
        """Executes the main algorithm of the stage and returns a normalized entity."""

    def unload_models(self) -> None:
        """Releases model references so the orchestrator can reclaim VRAM."""
        self._is_loaded = False