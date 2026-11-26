"""Abstract base classes defining the stage contract for the Ego2Act pipeline."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any


class PipelineStage(abc.ABC):
    """Base class every pipeline stage must implement.

    Each concrete stage follows a strict lifecycle:
      1. ``load_model()``  — acquire GPU/CPU resources and load weights.
      2. ``run()``         — process one episode and return typed output.
      3. ``unload()``      — release all resources so the next stage can reclaim VRAM.

    The :class:`StagedRunner` orchestrator enforces this order and
    guarantees ``unload()`` is called even when ``run()`` raises.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in logs and cache keys."""

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load model weights and allocate device memory."""

    @abc.abstractmethod
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the stage logic on *inputs* and return typed results.

        Parameters
        ----------
        inputs:
            Dictionary of named tensors / entities produced by upstream stages.

        Returns
        -------
        dict[str, Any]
            Dictionary of outputs that downstream stages or the cache layer
            will consume.
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Free all GPU/CPU resources acquired by :meth:`load_model`."""

    # ------------------------------------------------------------------
    # Optional hooks with sensible defaults
    # ------------------------------------------------------------------

    def cache_key(self, episode_id: str) -> str:
        """Return the stem of the cache file for a given *episode_id*.

        The default implementation produces ``<episode_id>_<stage_name>``.
        Override if the stage requires a more specific key derivation.
        """
        return f"{episode_id}_{self.name}"

    def cache_path(self, cache_dir: Path, episode_id: str) -> Path:
        """Return the full path of the cache file for *episode_id*."""
        # Normalize to forward slashes to stay cross-platform.
        return cache_dir / f"{self.cache_key(episode_id)}.safetensors"
