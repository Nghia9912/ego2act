"""Staged pipeline runner with VRAM lifecycle management."""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

from ego2act.core.interfaces import PipelineStage
from ego2act.core.io import load_stage_cache, save_stage_cache

logger = logging.getLogger(__name__)


def flush_vram() -> None:
    """Force Python garbage collection and release all CUDA cached memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug(
                "CUDA memory after flush — allocated: %.1f MB, reserved: %.1f MB",
                torch.cuda.memory_allocated() / 1e6,
                torch.cuda.memory_reserved() / 1e6,
            )
    except ImportError:
        pass


class StagedRunner:
    """Execute an ordered sequence of :class:`PipelineStage` instances.

    The runner enforces the ``load_model → run → unload`` lifecycle and
    flushes GPU memory between stages so the total peak VRAM stays within
    the capacity of a single 8 GB GPU.

    When a valid cache file already exists for a stage and *skip_cached* is
    ``True``, the runner loads the cached result instead of re-running the
    stage — saving both GPU time and VRAM.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        cache_dir: str | Path = "data/cache",
        *,
        skip_cached: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("At least one PipelineStage is required.")
        self._stages = stages
        self._cache_dir = Path(cache_dir)
        self._skip_cached = skip_cached

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, episode_id: str, initial_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute all stages sequentially for one episode.

        Parameters
        ----------
        episode_id:
            Unique identifier for the episode being processed.
        initial_inputs:
            Seed data fed into the first stage (e.g. raw video frames).

        Returns
        -------
        dict[str, Any]
            Merged output dictionary accumulated across all stages.
        """
        accumulated: dict[str, Any] = dict(initial_inputs or {})

        for stage in self._stages:
            cache_path = stage.cache_path(self._cache_dir, episode_id)

            if self._skip_cached and cache_path.is_file():
                logger.info("[%s] Cache hit — loading %s", stage.name, cache_path)
                cached = load_stage_cache(cache_path)
                accumulated.update(cached)
                continue

            logger.info("[%s] Starting execution …", stage.name)
            t0 = time.perf_counter()

            stage.load_model()
            try:
                result = stage.run(accumulated)
            finally:
                stage.unload()
                flush_vram()

            elapsed = time.perf_counter() - t0
            logger.info("[%s] Completed in %.2f s", stage.name, elapsed)

            # Persist to cache for future re-runs.
            cacheable = {k: v for k, v in result.items() if _is_cacheable(v)}
            if cacheable:
                save_stage_cache(
                    cacheable,
                    cache_path,
                    metadata={"stage": stage.name, "episode_id": episode_id},
                )

            accumulated.update(result)

        return accumulated


def _is_cacheable(value: Any) -> bool:
    """Return True if *value* is a NumPy array eligible for safetensors."""
    try:
        import numpy as np

        return isinstance(value, np.ndarray)
    except ImportError:  # pragma: no cover
        return False
