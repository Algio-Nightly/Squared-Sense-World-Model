"""Tests for Stage 1A Spatial JEPA training pipeline (train/train_spatial_jepa.py).

Verifies:
  - Smoke test execution creates best.pkl, last.pkl, and metrics.jsonl.
  - All logged losses and RankMe diagnostic values are finite.
  - Checkpoints reload cleanly into a fresh SpatialJEPA with identical representations.
  - Optimizer strictly updates context_encoder and predictor without touching target_encoder.
"""

from __future__ import annotations

import json
import math
import os
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.transformer import SpatialJEPA
from train.checkpoint import load_checkpoint
from train.train_spatial_jepa import parse_args, train_spatial_jepa, train_step


def test_train_spatial_smoke_test(tmp_path):
    """Verify smoke test completes, saves checkpoints, and logs finite metrics."""
    save_dir = str(tmp_path / "smoke_run")
    args = parse_args(["--smoke_test", "--save_dir", save_dir, "--seed", "123"])

    model, best_val_loss = train_spatial_jepa(args)

    # 1. Assert required files exist
    best_pkl = os.path.join(save_dir, "best.pkl")
    last_pkl = os.path.join(save_dir, "last.pkl")
    metrics_jsonl = os.path.join(save_dir, "metrics.jsonl")

    assert os.path.isfile(best_pkl), f"Missing {best_pkl}"
    assert os.path.isfile(last_pkl), f"Missing {last_pkl}"
    assert os.path.isfile(metrics_jsonl), f"Missing {metrics_jsonl}"

    # 2. Inspect metrics.jsonl records
    records = []
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2, f"Expected 2 epoch records, got {len(records)}"

    for r in records:
        assert math.isfinite(r["train_loss"]), f"Non-finite train_loss: {r['train_loss']}"
        assert math.isfinite(r["val_loss"]), f"Non-finite val_loss: {r['val_loss']}"
        assert math.isfinite(r["emb_std"]), f"Non-finite emb_std: {r['emb_std']}"
        assert r["train_loss"] > 0.0

    # Final epoch (epoch 2) must have computed RankMe
    assert "rankme" in records[-1], "Final epoch must record RankMe metric"
    assert math.isfinite(records[-1]["rankme"]), f"Non-finite RankMe: {records[-1]['rankme']}"
    assert records[-1]["rankme"] >= 1.0, f"RankMe must be >= 1.0, got {records[-1]['rankme']}"

    # 3. Reload best.pkl into a fresh SpatialJEPA
    fresh_model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(999))
    metadata = load_checkpoint(best_pkl, fresh_model)

    assert "epoch" in metadata
    assert "val_loss" in metadata
    assert math.isclose(metadata["val_loss"], best_val_loss, rel_tol=1e-5)

    # Verify identical predictions on unmasked input
    dummy_x = jnp.ones((2, 3, 16, 16, 16, 16), dtype=jnp.float32)
    orig_emb = model.context_encoder(dummy_x, visible_idx=None)
    restored_emb = fresh_model.context_encoder(dummy_x, visible_idx=None)
    np.testing.assert_allclose(np.asarray(orig_emb), np.asarray(restored_emb), atol=1e-5)


def test_optimizer_exclusion_target_encoder():
    """Verify target_encoder parameters are NOT updated by gradients or optimizer."""
    import optax

    model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(42))
    tx = optax.adamw(learning_rate=1e-3, weight_decay=0.05)
    optimizer = nnx.Optimizer((model.context_encoder, model.predictor), tx, wrt=nnx.Param)

    x = jax.random.uniform(jax.random.key(1), (2, 3, 16, 16, 16, 16))

    # Record initial target encoder weights
    target_state_before = nnx.to_pure_dict(nnx.state(model.target_encoder, nnx.Param))

    # Run train_step with ema_momentum = 1.0 (so EMA performs target = 1.0 * target + 0.0 * context)
    # If the optimizer or gradients were touching target_encoder, its weights would change.
    train_step(
        model=model,
        optimizer=optimizer,
        x=x,
        rng_key=jax.random.key(2),
        ema_momentum=1.0,
        alpha=0.1,
        beta=0.04,
    )

    target_state_after = nnx.to_pure_dict(nnx.state(model.target_encoder, nnx.Param))

    # Assert target encoder weights are completely untouched when ema_momentum=1.0
    before_leaves = jax.tree.leaves(target_state_before)
    after_leaves = jax.tree.leaves(target_state_after)

    for b, a in zip(before_leaves, after_leaves):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
