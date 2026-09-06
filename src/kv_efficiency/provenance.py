r"""Reproducibility metadata shared by every Milestone-2 experiment.

Every new experiment result file embeds a `provenance` block (config
snapshot, seed, model id, dtype, device, package versions, timestamp) so a
result can be interpreted without cross-referencing code history.
"""

from __future__ import annotations

import datetime
import importlib.metadata as md
import platform
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "capture_environment", "memory_bytes_saved"]


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def capture_environment(config_path: str | Path, model=None, seed: int | None = None) -> dict[str, Any]:
    """A single dict to embed verbatim in every results JSON file."""
    import torch

    model_revision = None
    if model is not None:
        model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_snapshot": load_config(config_path),
        "config_path": str(config_path),
        "seed": seed,
        "model_revision": model_revision,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_version": platform.python_version(),
        "package_versions": {
            pkg: _pkg_version(pkg)
            for pkg in ["torch", "transformers", "numpy", "scipy", "pandas", "pyyaml"]
        },
    }


def memory_bytes_saved(
    *, n_entries_removed: int, head_dim: int, n_kv_heads: int = 1, dtype_bytes: int = 4
) -> int:
    """Bytes saved by removing `n_entries_removed` key+value pairs for one
    (layer, head) -- both K and V vectors of size `head_dim`, `dtype_bytes`
    each, times `n_kv_heads` if the removal applies uniformly across a
    GQA-shared KV group (1 for GPT-NeoX's plain MHA, used here).
    """
    return n_entries_removed * 2 * head_dim * n_kv_heads * dtype_bytes
