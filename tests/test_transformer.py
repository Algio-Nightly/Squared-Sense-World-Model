"""Tests for Transformer Encoder, EMA Target, JEPA Predictor, and 3D Masking."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from models.cnn3d import count_params
from models.transformer import (
    JEPAPredictor,
    MultiLODBackbone,
    PreLNTransformerBlock,
    SpatialJEPA,
    sample_3d_jepa_masks,
)


def test_transformer_block() -> None:
    """Verify PreLNTransformerBlock preserves input shape."""
    rngs = nnx.Rngs(0)
    block = PreLNTransformerBlock(dim=256, num_heads=8, rngs=rngs)
    x = jnp.ones((2, 64, 256), dtype=jnp.float32)
    out = block(x)
    assert out.shape == (2, 64, 256)
    assert jnp.all(jnp.isfinite(out))


def test_sample_3d_jepa_masks_disjoint_and_coverage() -> None:
    """Verify visible_idx and target_idx are strictly disjoint and cover all tokens."""
    key = jax.random.PRNGKey(42)
    batch_size = 8

    # Test 3 shells: 192 tokens total (96 visible, 96 target)
    vis_3, tgt_3 = sample_3d_jepa_masks(key, batch_size=batch_size, num_shells=3)
    assert vis_3.shape == (batch_size, 96)
    assert tgt_3.shape == (batch_size, 96)

    for b in range(batch_size):
        v = set(vis_3[b].tolist())
        t = set(tgt_3[b].tolist())
        assert len(v) == 96
        assert len(t) == 96
        assert v.isdisjoint(t), f"Overlap found in batch item {b}: {v & t}"
        assert (v | t) == set(range(192)), "Union does not cover all 192 tokens"

    # Test 1 shell: 64 tokens total (32 visible, 32 target)
    vis_1, tgt_1 = sample_3d_jepa_masks(key, batch_size=batch_size, num_shells=1)
    assert vis_1.shape == (batch_size, 32)
    assert tgt_1.shape == (batch_size, 32)

    for b in range(batch_size):
        v = set(vis_1[b].tolist())
        t = set(tgt_1[b].tolist())
        assert len(v) == 32
        assert len(t) == 32
        assert v.isdisjoint(t), f"Overlap found in 1-shell batch item {b}: {v & t}"
        assert (v | t) == set(range(64)), "Union does not cover all 64 tokens"


def test_parameter_breakdown_and_budgets() -> None:
    """Verify Core Backbone and Stage 1A Total parameter counts within ±3% of contract."""
    rngs = nnx.Rngs(123)
    jepa = SpatialJEPA(num_shells=3, rngs=rngs)

    core_backbone_params = count_params(jepa.context_encoder)
    predictor_params = count_params(jepa.predictor)
    stage1a_total = core_backbone_params + predictor_params

    target_backbone = 3_647_264
    target_stage1a = 4_649_184

    print(f"\n[Parameter Count Breakdown]")
    print(f"  - Core Backbone (context_encoder) : {core_backbone_params:,} (target ~{target_backbone:,})")
    print(f"  - Predictor                       : {predictor_params:,} (target ~1,001,920)")
    print(f"  - Stage 1A Spatial Total          : {stage1a_total:,} (target ~{target_stage1a:,})")

    diff_bb = abs(core_backbone_params - target_backbone) / target_backbone
    assert diff_bb <= 0.03, f"Core backbone params {core_backbone_params} exceeds ±3% of {target_backbone}"

    diff_total = abs(stage1a_total - target_stage1a) / target_stage1a
    assert diff_total <= 0.03, f"Stage 1A total {stage1a_total} exceeds ±3% of {target_stage1a}"


def test_update_ema_smoothness() -> None:
    """Verify update_ema(0.99) updates target_encoder weights smoothly towards context_encoder."""
    rngs = nnx.Rngs(456)
    jepa = SpatialJEPA(num_shells=3, rngs=rngs)

    # Perturb context encoder parameters
    ctx_state = nnx.state(jepa.context_encoder, nnx.Param)
    perturbed_ctx = jax.tree.map(lambda c: c + 1.0, ctx_state)
    nnx.update(jepa.context_encoder, perturbed_ctx)

    # Snapshot target encoder array values before EMA update
    leaves_before = [
        jnp.copy(x[...])
        for x in jax.tree_util.tree_leaves(nnx.state(jepa.target_encoder, nnx.Param))
    ]

    momentum = 0.99
    jepa.update_ema(momentum)

    leaves_after = [
        x[...]
        for x in jax.tree_util.tree_leaves(nnx.state(jepa.target_encoder, nnx.Param))
    ]

    # Check that every leaf moved by exactly (1 - momentum) * (ctx - tgt_before) = 0.01 * 1.0 = 0.01
    assert len(leaves_before) == len(leaves_after)
    for b, a in zip(leaves_before, leaves_after):
        diff = a - b
        # Should be smoothly shifted by exactly 0.01
        assert jnp.allclose(diff, 0.01, atol=1e-5), f"EMA step unexpected change: {diff}"


def test_spatial_jepa_jit_forward_and_gradients() -> None:
    """Verify SpatialJEPA runs cleanly inside nnx.jit and target_encoder receives ZERO gradients."""
    rngs = nnx.Rngs(789)
    jepa = SpatialJEPA(num_shells=3, rngs=rngs)

    batch_size = 2
    x = jnp.ones((batch_size, 3, 16, 16, 16, 16), dtype=jnp.float32)
    key = jax.random.PRNGKey(999)
    vis_idx, tgt_idx = sample_3d_jepa_masks(key, batch_size=batch_size, num_shells=3)

    @nnx.jit
    def step_fn(model: SpatialJEPA, voxels: jax.Array, vis: jax.Array, tgt: jax.Array):
        def loss_fn(m: SpatialJEPA):
            pred, tgt_norm, ctx = m(voxels, vis, tgt)
            loss = jnp.mean((pred - tgt_norm) ** 2)
            return loss, (pred, tgt_norm, ctx)

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        return loss, aux, grads

    loss, (pred, tgt_norm, ctx), grads = step_fn(jepa, x, vis_idx, tgt_idx)

    # Verify tensor shapes
    assert pred.shape == (batch_size, 96, 256)
    assert tgt_norm.shape == (batch_size, 96, 256)
    assert ctx.shape == (batch_size, 96, 256)

    # Verify finite loss and outputs with no NaNs
    assert jnp.isfinite(loss)
    assert not jnp.any(jnp.isnan(pred))
    assert not jnp.any(jnp.isnan(tgt_norm))
    assert not jnp.any(jnp.isnan(ctx))

    # CRITICAL CHECK: target_encoder MUST receive ZERO gradients
    tgt_grad_leaves = jax.tree_util.tree_leaves(grads.target_encoder)
    assert len(tgt_grad_leaves) > 0
    for idx, g in enumerate(tgt_grad_leaves):
        assert jnp.all(g == 0.0), f"target_encoder leaf {idx} received non-zero gradient: {g}"

    # Verify context_encoder and predictor received non-zero gradients
    ctx_grad_leaves = jax.tree_util.tree_leaves(grads.context_encoder)
    assert any(jnp.any(g != 0) for g in ctx_grad_leaves)

    pred_grad_leaves = jax.tree_util.tree_leaves(grads.predictor)
    assert any(jnp.any(g != 0) for g in pred_grad_leaves)
