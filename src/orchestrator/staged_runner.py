"""
Sequential staged runner coordinator.
Ensures each stage loads models, processes data, caches .safetensors, and releases VRAM cleanly
before moving to the next stage to run smoothly on an 8GB VRAM GPU.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from src.core.interfaces import BasePipelineStage

logger = logging.getLogger("ego2act.orchestrator")


def flush_memory() -> dict[str, float]:
    """
    Forces Python's garbage collector to reclaim memory and clears CUDA cache (if PyTorch is installed).
    Returns allocated VRAM usage information (GB) for monitoring against the <8GB VRAM target.
    """
    gc.collect()
    vram_stats = {"allocated_gb": 0.0, "reserved_gb": 0.0}

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vram_stats["allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            vram_stats["reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
    except ImportError:
        pass

    return vram_stats


class StagedPipelineRunner:
    """Manages the sequential stage lifecycle and memory control."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: dict[str, Any] = yaml.safe_load(f)

    def run_stage(self, stage: BasePipelineStage[Any, Any], inputs: Any) -> Any:
        """Executes a single stage with the flow: Load -> Process -> Unload -> Flush VRAM."""
        logger.info("=== Start [%s] ===", stage.stage_name)
        t0 = time.perf_counter()

        stage.load_models()
        try:
            output = stage.process(inputs)
        finally:
            stage.unload_models()
            vram_after = flush_memory()

        elapsed = time.perf_counter() - t0
        logger.info(
            "=== Complete [%s] in %.2fs | VRAM remaining: %.3f GB ===",
            stage.stage_name,
            elapsed,
            vram_after["allocated_gb"],
        )
        return output