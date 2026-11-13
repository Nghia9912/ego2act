"""
Abstract Base Classes (ABCs) for processing stages in the Staged Pipeline architecture.
Ensures each stage follows a strict lifecycle: load_models() -> process() -> unload_models().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BasePipelineStage(ABC, Generic[InputT, OutputT]):
    """Base contract for all processing stages in Ego2Act."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._is_loaded: bool = False

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Unique identifier for the stage (e.g., 'stage1_ego_odometry')."""

    def load_models(self) -> None:
        """Loads model weights into GPU/CPU memory."""
        self._is_loaded = True

    @abstractmethod
    def process(self, inputs: InputT) -> OutputT:
        """Executes the stage algorithm and returns a validated entity."""

    def unload_models(self) -> None:
        """Releases model references so the orchestrator can reclaim VRAM."""
        self._is_loaded = False