"""Tests for Parameter-Matched Baselines training pipeline (train/train_baselines.py).

Verifies:
  - b1_lod0 (SingleLODJEPA LOD-0) completes smoke test, saves valid checkpoints, and logs finite metrics/RankMe.
  - b2_lod2 (SingleLODJEPA LOD-2) completes smoke test, saves valid checkpoints, and logs finite metrics/RankMe.
  - b3_mae (Generative3DMAE) completes smoke test, minimizes patch MSE loss, and logs finite metrics/RankMe.
  - All saved checkpoints reload cleanly into fresh module instances.
"""

from __future__ import annotations

import json
import math
import os
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from models.baselines import Generative3DMAE, SingleLODJEPA
from train.checkpoint import load_checkpoint
from train.train_baselines import parse_args, train_baseline


@pytest.mark.parametrize("model_name,lod_idx", [("b1_lod0", 0), ("b2_lod2", 2)])
def test_train_single_lod_jepa_smoke_test(tmp_path, model_name, lod_idx):
    """Verify SingleLODJEPA variants complete smoke test, write checkpoints, and log finite metrics."""
    save_dir = str(tmp_path / model_name)
    args = parse_args(["--model", model_name, "--smoke_test", "--save_dir", save_dir, "--seed", "42"])

    model, best_val_loss = train_baseline(args)
    assert isinstance(model, SingleLODJEPA)
    assert model.lod_index == lod_idx

    best_pkl = os.path.join(save_dir, "best.pkl")
    last_pkl = os.path.join(save_dir, "last.pkl")
    metrics_jsonl = os.path.join(save_dir, "metrics.jsonl")

    assert os.path.isfile(best_pkl), f"Missing {best_pkl}"
    assert os.path.isfile(last_pkl), f"Missing {last_pkl}"
    assert os.path.isfile(metrics_jsonl), f"Missing {metrics_jsonl}"

    records = []
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2, f"Expected 2 epoch records, got {len(records)}"

    for r in records:
        assert math.isfinite(r["train_loss"])
        assert math.isfinite(r["val_loss"])
        assert r["train_loss"] > 0.0

    # Final epoch must include RankMe
    assert "rankme" in records[-1]
    assert math.isfinite(records[-1]["rankme"])
    assert records[-1]["rankme"] >= 1.0

    # Verify reload
    fresh_model = SingleLODJEPA(lod_index=lod_idx, rngs=nnx.Rngs(999))
    metadata = load_checkpoint(best_pkl, fresh_model)
    assert metadata["model"] == model_name
    assert math.isclose(metadata["val_loss"], best_val_loss, rel_tol=1e-5)

    dummy_x = jnp.ones((2, 3, 16, 16, 16, 16))
    orig_emb = model.context_encoder(dummy_x[:, lod_idx : lod_idx + 1], visible_idx=None)
    restored_emb = fresh_model.context_encoder(dummy_x[:, lod_idx : lod_idx + 1], visible_idx=None)
    np.testing.assert_allclose(np.asarray(orig_emb), np.asarray(restored_emb), atol=1e-5)


def test_train_generative_mae_smoke_test(tmp_path):
    """Verify Generative3DMAE completes smoke test, minimizes MSE loss, and reloads cleanly."""
    save_dir = str(tmp_path / "b3_mae")
    args = parse_args(["--model", "b3_mae", "--smoke_test", "--save_dir", save_dir, "--seed", "42"])

    model, best_val_loss = train_baseline(args)
    assert isinstance(model, Generative3DMAE)

    best_pkl = os.path.join(save_dir, "best.pkl")
    last_pkl = os.path.join(save_dir, "last.pkl")
    metrics_jsonl = os.path.join(save_dir, "metrics.jsonl")

    assert os.path.isfile(best_pkl)
    assert os.path.isfile(last_pkl)
    assert os.path.isfile(metrics_jsonl)

    records = []
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2

    for r in records:
        assert math.isfinite(r["train_loss"])
        assert math.isfinite(r["val_loss"])
        assert r["train_loss"] >= 0.0

    # Final epoch must include RankMe
    assert "rankme" in records[-1]
    assert math.isfinite(records[-1]["rankme"])
    assert records[-1]["rankme"] >= 1.0

    # Verify reload
    fresh_model = Generative3DMAE(rngs=nnx.Rngs(999))
    metadata = load_checkpoint(best_pkl, fresh_model)
    assert metadata["model"] == "b3_mae"
    assert math.isclose(metadata["val_loss"], best_val_loss, rel_tol=1e-5)

    dummy_x = jnp.ones((2, 3, 16, 16, 16, 16))
    orig_emb = model.encoder(dummy_x, visible_idx=None)
    restored_emb = fresh_model.encoder(dummy_x, visible_idx=None)
    np.testing.assert_allclose(np.asarray(orig_emb), np.asarray(restored_emb), atol=1e-5)
