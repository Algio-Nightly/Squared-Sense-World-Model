"""Tests for Stage 1B Dual-LSTM Temporal JEPA trainer (train/train_temporal_jepa.py).

Verifies:
  - Smoke test execution completes Phase 1 and Phase 2, saving checkpoints and logging finite metrics.
  - All logged losses and RankMe diagnostic values on fused context states c_t are finite.
  - Phase 1 does NOT modify any spatial_model weights.
  - Phase 2 updates ONLY blocks[-2:] and final norm, leaving cnn and blocks[:2] completely untouched.
  - Saved checkpoints reload cleanly into fresh models.
"""

from __future__ import annotations

import json
import math
import os
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import SpatialJEPA
from train.checkpoint import load_checkpoint
from train.train_temporal_jepa import (
    parse_args,
    train_step_phase1,
    train_step_phase2,
    train_temporal_jepa,
)


def test_train_temporal_smoke_test(tmp_path):
    """Verify smoke test completes two phases, writes valid checkpoints and metrics."""
    save_dir = str(tmp_path / "smoke_temporal")
    args = parse_args(["--smoke_test", "--save_dir", save_dir, "--seed", "42"])

    temporal_model, spatial_model = train_temporal_jepa(args)

    temp_best_pkl = os.path.join(save_dir, "temporal_best.pkl")
    spat_best_pkl = os.path.join(save_dir, "spatial_finetuned_best.pkl")
    metrics_jsonl = os.path.join(save_dir, "metrics.jsonl")

    assert os.path.isfile(temp_best_pkl), f"Missing {temp_best_pkl}"
    assert os.path.isfile(spat_best_pkl), f"Missing {spat_best_pkl}"
    assert os.path.isfile(metrics_jsonl), f"Missing {metrics_jsonl}"

    # Verify metrics.jsonl
    records = []
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2, f"Expected 2 records (1 per phase in smoke test), got {len(records)}"

    for r in records:
        assert math.isfinite(r["train_loss"]), f"Non-finite train_loss: {r['train_loss']}"
        assert math.isfinite(r["val_loss"]), f"Non-finite val_loss: {r['val_loss']}"
        assert math.isfinite(r["rollout_l1"]), f"Non-finite rollout_l1: {r['rollout_l1']}"
        assert math.isfinite(r["idm_loss"]), f"Non-finite idm_loss: {r['idm_loss']}"
        assert math.isfinite(r["rankme"]), f"Non-finite rankme: {r['rankme']}"
        assert r["rankme"] >= 1.0, f"RankMe must be >= 1.0, got {r['rankme']}"

    # Verify reload of checkpoints into fresh instances
    fresh_temporal = DualLSTMTemporalIntegrator(rngs=nnx.Rngs(100))
    t_meta = load_checkpoint(temp_best_pkl, fresh_temporal)
    assert "val_loss" in t_meta
    assert "rankme" in t_meta

    fresh_spatial = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(200))
    s_meta = load_checkpoint(spat_best_pkl, fresh_spatial)
    assert "val_loss" in s_meta

    dummy_shells = jnp.ones((2, 3, 16, 16, 16, 16))
    orig_emb = spatial_model.context_encoder(dummy_shells, visible_idx=None)
    reloaded_emb = fresh_spatial.context_encoder(dummy_shells, visible_idx=None)
    np.testing.assert_allclose(np.asarray(orig_emb), np.asarray(reloaded_emb), atol=1e-5)


def test_phase1_leaves_spatial_untouched():
    """Verify Phase 1 strictly does NOT modify ANY spatial_model weights."""
    spatial_model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(10))
    temporal_model = DualLSTMTemporalIntegrator(rngs=nnx.Rngs(20))

    tx = optax.adamw(learning_rate=3e-4)
    opt_temp = nnx.Optimizer(temporal_model, tx, wrt=nnx.Param)

    shells = jax.random.uniform(jax.random.key(1), (2, 16, 3, 16, 16, 16, 16))
    states = jax.random.normal(jax.random.key(2), (2, 16, 12))
    actions = jax.random.randint(jax.random.key(3), (2, 16), 0, 12)

    # Capture initial spatial state
    spat_before = nnx.to_pure_dict(nnx.state(spatial_model, nnx.Param))

    # Run Phase 1 step
    train_step_phase1(
        temporal_model=temporal_model,
        spatial_model=spatial_model,
        opt_temp=opt_temp,
        shells=shells,
        states=states,
        actions=actions,
        k_steps=4,
    )

    spat_after = nnx.to_pure_dict(nnx.state(spatial_model, nnx.Param))

    # Assert exact equality across all spatial parameters
    for b, a in zip(jax.tree.leaves(spat_before), jax.tree.leaves(spat_after)):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


def test_phase2_updates_only_top_blocks_and_leaves_cnn_and_blocks_0_1_untouched():
    """Verify Phase 2 updates ONLY blocks[-2:] and norm while cnn and blocks[:2] remain untouched."""
    spatial_model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(30))
    temporal_model = DualLSTMTemporalIntegrator(rngs=nnx.Rngs(40))

    tx_temp = optax.adamw(learning_rate=3e-4)
    opt_temp = nnx.Optimizer(temporal_model, tx_temp, wrt=nnx.Param)

    top_blocks = (
        spatial_model.context_encoder.blocks[2],
        spatial_model.context_encoder.blocks[3],
        spatial_model.context_encoder.norm,
    )
    tx_spat = optax.adamw(learning_rate=3e-5)
    opt_spat = nnx.Optimizer(top_blocks, tx_spat, wrt=nnx.Param)

    shells = jax.random.uniform(jax.random.key(4), (2, 16, 3, 16, 16, 16, 16))
    states = jax.random.normal(jax.random.key(5), (2, 16, 12))
    actions = jax.random.randint(jax.random.key(6), (2, 16), 0, 12)

    # Capture states before
    cnn_before = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.cnn, nnx.Param))
    b0_before = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[0], nnx.Param))
    b1_before = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[1], nnx.Param))
    b2_before = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[2], nnx.Param))
    b3_before = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[3], nnx.Param))
    norm_before = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.norm, nnx.Param))

    # Run Phase 2 step
    train_step_phase2(
        temporal_model=temporal_model,
        spatial_model=spatial_model,
        opt_temp=opt_temp,
        opt_spat=opt_spat,
        shells=shells,
        states=states,
        actions=actions,
        rng_key=jax.random.key(7),
        k_steps=4,
    )

    cnn_after = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.cnn, nnx.Param))
    b0_after = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[0], nnx.Param))
    b1_after = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[1], nnx.Param))
    b2_after = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[2], nnx.Param))
    b3_after = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.blocks[3], nnx.Param))
    norm_after = nnx.to_pure_dict(nnx.state(spatial_model.context_encoder.norm, nnx.Param))

    # 1. Assert CNN is strictly UNTOUCHED
    for b, a in zip(jax.tree.leaves(cnn_before), jax.tree.leaves(cnn_after)):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

    # 2. Assert blocks[0] and blocks[1] are strictly UNTOUCHED
    for b, a in zip(jax.tree.leaves(b0_before), jax.tree.leaves(b0_after)):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

    for b, a in zip(jax.tree.leaves(b1_before), jax.tree.leaves(b1_after)):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

    # 3. Assert top blocks (blocks[2], blocks[3], or norm) WERE updated
    top_diffs = []
    for b, a in zip(jax.tree.leaves(b2_before), jax.tree.leaves(b2_after)):
        top_diffs.append(not np.array_equal(np.asarray(b), np.asarray(a)))
    for b, a in zip(jax.tree.leaves(b3_before), jax.tree.leaves(b3_after)):
        top_diffs.append(not np.array_equal(np.asarray(b), np.asarray(a)))
    for b, a in zip(jax.tree.leaves(norm_before), jax.tree.leaves(norm_after)):
        top_diffs.append(not np.array_equal(np.asarray(b), np.asarray(a)))

    assert any(top_diffs), "Expected top blocks or final norm to be updated in Phase 2"
