"""Safetensors-based cache reader and writer for inter-stage tensor exchange."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

logger = logging.getLogger(__name__)


def save_stage_cache(
    tensors: dict[str, np.ndarray],
    path: str | Path,
    *,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Serialize a dictionary of named NumPy arrays to a ``.safetensors`` file.

    Parameters
    ----------
    tensors:
        Mapping of tensor names to :class:`numpy.ndarray` values.
        Each array must be contiguous and have a dtype supported by
        ``safetensors`` (float16/32/64, int8/16/32/64, uint8/16/32/64, bool).
    path:
        Destination file path.  Parent directories are created automatically.
    metadata:
        Optional key-value metadata stored alongside the tensors (e.g. stage
        name, creation timestamp, source video hash).

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    ValueError
        If *tensors* is empty or contains unsupported dtypes.
    """
    if not tensors:
        raise ValueError("Cannot save an empty tensor dictionary.")

    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    save_file(tensors, str(dest), metadata=metadata)
    logger.info("Saved %d tensors to %s (%.2f MB)", len(tensors), dest, dest.stat().st_size / 1e6)
    return dest


def load_stage_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Deserialize a ``.safetensors`` cache file back into NumPy arrays.

    Parameters
    ----------
    path:
        Path to an existing ``.safetensors`` file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of tensor names to :class:`numpy.ndarray` values.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cache file not found: {source}")

    tensors = load_file(str(source))
    logger.info("Loaded %d tensors from %s", len(tensors), source)
    return tensors


def verify_cache_shapes(
    tensors: dict[str, np.ndarray],
    expected: dict[str, tuple[int, ...]],
) -> list[str]:
    """Check that cached tensor shapes match expectations.

    Parameters
    ----------
    tensors:
        Tensors loaded from a cache file.
    expected:
        Mapping of tensor name → expected shape.  Use ``-1`` for axes whose
        size may vary (e.g. the temporal dimension).

    Returns
    -------
    list[str]
        List of human-readable mismatch descriptions (empty if all match).
    """
    mismatches: list[str] = []
    for name, exp_shape in expected.items():
        if name not in tensors:
            mismatches.append(f"Missing tensor '{name}'")
            continue
        actual_shape = tensors[name].shape
        if len(actual_shape) != len(exp_shape):
            mismatches.append(
                f"'{name}': expected {len(exp_shape)}D, got {len(actual_shape)}D "
                f"(shape {actual_shape})"
            )
            continue
        for axis, (a, e) in enumerate(zip(actual_shape, exp_shape, strict=False)):
            if e != -1 and a != e:
                mismatches.append(
                    f"'{name}' axis {axis}: expected {e}, got {a} (shape {actual_shape})"
                )
    return mismatches
