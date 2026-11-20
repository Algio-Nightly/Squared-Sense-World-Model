"""Diagnostics and evaluation metrics for representation quality (RankMe, logging).

Strictly follows docs/TRAIN_CONTRACT.md.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np


def compute_rankme(embeddings: np.ndarray | jax.Array, eps: float = 1e-7) -> float:
    """Compute RankMe effective rank metric on uncentered embeddings.

    RankMe (Garrido et al., NeurIPS 2023) measures the continuous effective rank
    of a representation matrix without thresholding singular values:
      p_k = sigma_k / (sum_j sigma_j + eps)
      RankMe = exp(-sum_k p_k * log(p_k + eps))

    Note: As specified in the RankMe paper and docs/TRAIN_CONTRACT.md, the singular
    value decomposition is computed directly on the UNCENTERED matrix (do NOT subtract
    the mean), preserving the origin and full feature energy.

    Args:
        embeddings: Latent representations of shape (N, D) or (..., D).
        eps: Small epsilon for numerical stability.

    Returns:
        Continuous effective dimensionality in [1.0, D].
    """
    z = jnp.asarray(embeddings, dtype=jnp.float32)
    if z.ndim > 2:
        z = z.reshape((-1, z.shape[-1]))

    # Compute singular values on uncentered embedding matrix
    s = jnp.linalg.svd(z, compute_uv=False)
    p = s / (jnp.sum(s) + eps)
    entropy = -jnp.sum(p * jnp.log(p + eps))
    return float(jnp.exp(entropy))


def append_metrics_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a training/validation metric record to a JSON Lines (jsonl) file.

    Handles converting JAX/NumPy scalar and array values to JSON-serializable types.

    Args:
        path: Path to target .jsonl file.
        record: Dictionary of metrics for the current step/epoch.
    """
    def _serialize(v: Any) -> Any:
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (np.ndarray, jax.Array)):
            return v.tolist()
        return v

    clean_record = {k: _serialize(v) for k, v in record.items()}

    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean_record) + "\n")
