"""Tests for training utilities (losses, diagnostics, checkpointing)."""

import json
import os
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from models.cnn3d import SiameseVoxelCNN
from train.checkpoint import load_checkpoint, save_checkpoint
from train.diagnostics import append_metrics_jsonl, compute_rankme
from train.losses import (
    cosine_ema_momentum,
    smooth_l1_loss,
    spatial_jepa_loss,
    temporal_jepa_loss,
    vicreg_var_cov_loss,
)


def test_compute_rankme_isotropic_and_collapsed() -> None:
    """Verify compute_rankme on isotropic matrix is > 100.0, and on rank-1 collapsed matrix is < 2.0."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    # 1. Isotropic random matrix (512, 256) -> full rank representation
    x_iso = jax.random.normal(k1, (512, 256))
    rank_iso = compute_rankme(x_iso)
    print(f"\n[RankMe Isotropic]: {rank_iso:.2f}")
    assert rank_iso > 100.0, f"Expected isotropic RankMe > 100.0, got {rank_iso}"

    # 2. Rank-1 collapsed representation: ones((512, 1)) @ rand((1, 256))
    v = jax.random.normal(k2, (1, 256))
    x_rank1 = jnp.ones((512, 1)) @ v
    rank_collapsed = compute_rankme(x_rank1)
    print(f"[RankMe Rank-1 Collapsed]: {rank_collapsed:.4f}")
    assert rank_collapsed < 2.0, f"Expected collapsed RankMe < 2.0, got {rank_collapsed}"


def test_smooth_l1_loss() -> None:
    """Verify smooth_l1_loss behavior for small and large differences."""
    p1 = jnp.array([0.0, 0.5])
    t1 = jnp.array([0.0, 0.5])
    assert jnp.allclose(smooth_l1_loss(p1, t1), 0.0)

    # Small diff (< 1.0): 0.5 * diff^2
    p2 = jnp.array([0.0])
    t2 = jnp.array([0.5])
    assert jnp.allclose(smooth_l1_loss(p2, t2, beta=1.0), 0.5 * 0.25)

    # Large diff (> 1.0): diff - 0.5
    p3 = jnp.array([0.0])
    t3 = jnp.array([2.0])
    assert jnp.allclose(smooth_l1_loss(p3, t3, beta=1.0), 2.0 - 0.5)


def test_vicreg_var_cov_loss() -> None:
    """Verify vicreg_var_cov_loss returns finite non-negative scalars."""
    key = jax.random.PRNGKey(10)
    z = jax.random.normal(key, (32, 64))

    loss_var, loss_cov = vicreg_var_cov_loss(z, gamma=1.0)
    assert jnp.isfinite(loss_var)
    assert jnp.isfinite(loss_cov)
    assert loss_var >= 0.0
    assert loss_cov >= 0.0


def test_spatial_jepa_loss_finite_and_gradients() -> None:
    """Verify spatial_jepa_loss returns finite scalar and non-zero gradients."""
    b, n_tgt, n_vis, d = 4, 32, 32, 64
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(11), 3)

    pred_targets = jax.random.normal(k1, (b, n_tgt, d))
    norm_ema_targets = jax.random.normal(k2, (b, n_tgt, d))
    context_tokens = jax.random.normal(k3, (b, n_vis, d))

    loss, metrics = spatial_jepa_loss(pred_targets, norm_ema_targets, context_tokens)
    assert jnp.isfinite(loss)
    assert "pred_loss" in metrics
    assert "var_loss" in metrics
    assert "cov_loss" in metrics

    # Test gradients with respect to pred_targets and context_tokens
    def loss_fn(p, c):
        l, _ = spatial_jepa_loss(p, norm_ema_targets, c)
        return l

    grad_p, grad_c = jax.grad(loss_fn, argnums=(0, 1))(pred_targets, context_tokens)
    assert jnp.all(jnp.isfinite(grad_p))
    assert jnp.all(jnp.isfinite(grad_c))
    assert not jnp.all(grad_p == 0.0)
    assert not jnp.all(grad_c == 0.0)


def test_temporal_jepa_loss_finite_and_gradients() -> None:
    """Verify temporal_jepa_loss returns finite scalar and non-zero gradients."""
    b, t_val, k, d = 2, 4, 3, 32
    num_actions = 12
    key = jax.random.PRNGKey(12)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    pred_future = jax.random.normal(k1, (b, t_val, k, d))
    ema_future = jax.random.normal(k2, (b, t_val, k, d))
    idm_logits = jax.random.normal(k3, (b, 6, num_actions))
    actions = jax.random.randint(k4, (b, 6), minval=0, maxval=num_actions)

    loss, metrics = temporal_jepa_loss(pred_future, ema_future, idm_logits, actions)
    assert jnp.isfinite(loss)
    assert "rollout_l1" in metrics
    assert "idm_loss" in metrics
    assert "var_loss" in metrics
    assert "cov_loss" in metrics

    # Test gradients with respect to pred_future and idm_logits
    def loss_fn(pf, logits):
        l, _ = temporal_jepa_loss(pf, ema_future, logits, actions)
        return l

    grad_pf, grad_logits = jax.grad(loss_fn, argnums=(0, 1))(pred_future, idm_logits)
    assert jnp.all(jnp.isfinite(grad_pf))
    assert jnp.all(jnp.isfinite(grad_logits))
    assert not jnp.all(grad_pf == 0.0)
    assert not jnp.all(grad_logits == 0.0)


def test_cosine_ema_momentum() -> None:
    """Verify cosine_ema_momentum smoothly progresses from start to end."""
    total_steps = 1000
    m_start = cosine_ema_momentum(0, total_steps, start=0.996, end=1.0)
    m_mid = cosine_ema_momentum(500, total_steps, start=0.996, end=1.0)
    m_end = cosine_ema_momentum(1000, total_steps, start=0.996, end=1.0)

    assert jnp.allclose(m_start, 0.996, atol=1e-5)
    assert m_start < m_mid < m_end
    assert jnp.allclose(m_end, 1.0, atol=1e-5)


def test_checkpoint_roundtrip_siamese_voxel_cnn(tmp_path) -> None:
    """Verify save_checkpoint + load_checkpoint round-trips SiameseVoxelCNN with exact bit-level equality."""
    ckpt_path = str(tmp_path / "model_ckpt.pkl")

    # 1. Initialize original model
    model_orig = SiameseVoxelCNN(rngs=nnx.Rngs(42))
    meta_saved = {"epoch": 15, "loss": 0.042, "run_id": "test_run"}

    # 2. Save checkpoint
    save_checkpoint(ckpt_path, model_orig, metadata=meta_saved)
    assert os.path.exists(ckpt_path)

    # 3. Initialize another model with different random weights
    model_restored = SiameseVoxelCNN(rngs=nnx.Rngs(999))

    # Verify initially different
    leaves_orig = jax.tree.leaves(nnx.state(model_orig, nnx.Param))
    leaves_restored = jax.tree.leaves(nnx.state(model_restored, nnx.Param))
    assert not all(jnp.array_equal(l1[...], l2[...]) for l1, l2 in zip(leaves_orig, leaves_restored))

    # 4. Load checkpoint
    meta_loaded = load_checkpoint(ckpt_path, model_restored)

    # 5. Assert exact bit-level equality
    leaves_restored_after = jax.tree.leaves(nnx.state(model_restored, nnx.Param))
    assert len(leaves_orig) == len(leaves_restored_after)
    for l1, l2 in zip(leaves_orig, leaves_restored_after):
        assert jnp.allclose(l1[...], l2[...], atol=0.0), "Mismatch in restored parameters"

    assert meta_loaded == meta_saved


def test_append_metrics_jsonl(tmp_path) -> None:
    """Verify append_metrics_jsonl writes clean JSON lines with NumPy/JAX conversion."""
    jsonl_path = str(tmp_path / "logs" / "metrics.jsonl")

    append_metrics_jsonl(jsonl_path, {"epoch": 1, "loss": jnp.array(0.5), "val_acc": 0.85})
    append_metrics_jsonl(jsonl_path, {"epoch": 2, "loss": jnp.array(0.3), "val_acc": 0.91})

    assert os.path.exists(jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0]["epoch"] == 1
    assert lines[0]["loss"] == pytest.approx(0.5)
    assert lines[1]["epoch"] == 2
    assert lines[1]["loss"] == pytest.approx(0.3)
