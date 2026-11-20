"""Zero-dependency, cross-platform checkpointing for Flax NNX modules.

Saves and restores pure-dict parameter states converted to NumPy arrays via pickle,
strictly following docs/TRAIN_CONTRACT.md.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def save_checkpoint(
    path: str,
    module: nnx.Module,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save Flax NNX module parameter state and optional metadata to disk.

    Extracts nnx.to_pure_dict(nnx.state(module, nnx.Param)), converts all leaves
    to NumPy arrays for robust platform-independent serialization, and dumps via pickle.

    Args:
        path: File path where checkpoint will be saved (e.g. 'checkpoints/run/best.pkl').
        module: Flax NNX module instance to save.
        metadata: Optional dictionary of training metadata (e.g. epoch, step, loss).
    """
    state = nnx.state(module, nnx.Param)
    pure_dict = nnx.to_pure_dict(state)
    np_pure_dict = jax.tree.map(np.asarray, pure_dict)

    payload = {
        "state": np_pure_dict,
        "metadata": metadata or {},
    }

    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str, module: nnx.Module) -> Dict[str, Any]:
    """Load checkpoint parameters into a Flax NNX module and return metadata.

    Loads the serialized pure dict, converts array leaves to JAX arrays, and restores
    parameters in-place via nnx.replace_by_pure_dict and nnx.update.

    Args:
        path: File path of the checkpoint to load.
        module: Flax NNX module instance to update with restored parameters.

    Returns:
        Dictionary containing saved metadata.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    np_pure_dict = payload["state"]
    metadata = payload.get("metadata", {})

    jnp_pure_dict = jax.tree.map(jnp.asarray, np_pure_dict)
    current_state = nnx.state(module, nnx.Param)
    nnx.replace_by_pure_dict(current_state, jnp_pure_dict)
    nnx.update(module, current_state)

    return metadata
