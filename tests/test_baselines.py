"""Tests for Baseline A (Single-LOD JEPA) and Baseline B (Generative 3D-MAE)."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from models.baselines import Generative3DMAE, SingleLODJEPA
from models.cnn3d import count_params
from models.transformer import sample_3d_jepa_masks


@pytest.mark.parametrize("lod_index", [0, 2])
def test_single_lod_jepa_forward_jit_and_gradients(lod_index: int) -> None:
    """Verify SingleLODJEPA runs forward + nnx.value_and_grad inside nnx.jit for LOD-0 and LOD-2."""
    b = 2
    rngs = nnx.Rngs(lod_index)
    model = SingleLODJEPA(lod_index=lod_index, rngs=rngs)

    # 1. Parameter budget check (~4,648,672 params, within ±3%)
    trainable_params = model.count_trainable_params()
    target_params = 4_648_672
    diff = abs(trainable_params - target_params) / target_params
    print(f"\nSingleLODJEPA(lod={lod_index}) trainable params: {trainable_params:,} (diff={diff:.3%})")
    assert diff <= 0.03, f"SingleLODJEPA params {trainable_params} exceeds ±3% of {target_params}"

    # 2. Dummy inputs and 1-shell masks (64 tokens total: 32 visible, 32 target)
    key = jax.random.PRNGKey(10 + lod_index)
    k_data, k_mask = jax.random.split(key)
    x = jax.random.uniform(k_data, (b, 3, 16, 16, 16, 16))
    vis_idx, tgt_idx = sample_3d_jepa_masks(k_mask, batch_size=b, num_shells=1)

    assert vis_idx.shape == (b, 32)
    assert tgt_idx.shape == (b, 32)

    # 3. Forward + value_and_grad inside nnx.jit
    @nnx.jit
    def train_step(m, x_in, vis, tgt):
        def loss_fn(model_inner):
            pred_targets, norm_ema_targets, context_tokens = model_inner(x_in, vis, tgt)
            loss = jnp.mean(jnp.square(pred_targets - norm_ema_targets))
            return loss, (pred_targets, norm_ema_targets, context_tokens)

        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
        return loss, aux, grads

    loss, (pred_targets, norm_ema_targets, context_tokens), grads = train_step(
        model, x, vis_idx, tgt_idx
    )

    # 4. Assert outputs and loss
    assert jnp.isfinite(loss)
    assert pred_targets.shape == (b, 32, 256)
    assert norm_ema_targets.shape == (b, 32, 256)
    assert context_tokens.shape == (b, 32, 256)

    # 5. Verify gradient properties:
    # Context encoder and predictor receive non-zero gradients; target encoder receives ZERO gradients
    ctx_leaves = jax.tree_util.tree_leaves(grads.jepa.context_encoder)
    pred_leaves = jax.tree_util.tree_leaves(grads.jepa.predictor)
    tgt_leaves = jax.tree_util.tree_leaves(grads.jepa.target_encoder)

    for leaf in ctx_leaves + pred_leaves:
        assert jnp.all(jnp.isfinite(leaf)), "Gradient contains NaN or Inf"
        assert not jnp.all(leaf == 0.0), "Gradient contains all zeros"

    for leaf in tgt_leaves:
        assert jnp.all(leaf == 0.0), "Target encoder must receive zero gradients"


def test_generative_3d_mae_forward_jit_and_gradients() -> None:
    """Verify Generative3DMAE runs forward + nnx.value_and_grad inside nnx.jit and reconstructs patches."""
    b = 2
    rngs = nnx.Rngs(42)
    model = Generative3DMAE(rngs=rngs)

    # 1. Parameter budget check (~5.126M params, within ±7%)
    total_params = count_params(model)
    target_params = 5_126_000
    diff = abs(total_params - target_params) / target_params
    print(f"\nGenerative3DMAE total params: {total_params:,} (diff={diff:.3%})")
    assert diff <= 0.07, f"Generative3DMAE params {total_params} exceeds target budget (diff={diff:.3%})"

    # 2. Dummy inputs and 3-shell masks (192 tokens total: 96 visible, 96 target)
    key = jax.random.PRNGKey(43)
    k_data, k_mask = jax.random.split(key)
    x = jax.random.uniform(k_data, (b, 3, 16, 16, 16, 16))
    vis_idx, tgt_idx = sample_3d_jepa_masks(k_mask, batch_size=b, num_shells=3)

    assert vis_idx.shape == (b, 96)
    assert tgt_idx.shape == (b, 96)

    # 3. Forward + value_and_grad inside nnx.jit
    @nnx.jit
    def train_step(m, x_in, vis, tgt):
        def loss_fn(model_inner):
            pred_patches, target_patches, mse_loss = model_inner(x_in, vis, tgt)
            return mse_loss, (pred_patches, target_patches)

        (mse_loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m)
        return mse_loss, aux, grads

    mse_loss, (pred_patches, target_patches), grads = train_step(model, x, vis_idx, tgt_idx)

    # 4. Assert scalar loss and patch shapes
    assert mse_loss.ndim == 0, f"Expected scalar loss, got shape {mse_loss.shape}"
    assert jnp.isfinite(mse_loss)
    assert pred_patches.shape == (b, 96, 1024)
    assert target_patches.shape == (b, 96, 1024)

    # 5. Verify finite non-zero gradients across encoder and decoder
    enc_leaves = jax.tree_util.tree_leaves(grads.encoder)
    dec_leaves = [
        grads.decoder_input_proj.kernel,
        grads.pred_head.kernel,
    ]
    for leaf in enc_leaves + dec_leaves:
        assert jnp.all(jnp.isfinite(leaf)), "Gradient contains NaN or Inf"
        assert not jnp.all(leaf == 0.0), "Gradient contains all zeros"


def test_generative_3d_mae_patchify_identity() -> None:
    """Verify patchify matches direct voxel sub-volume slices."""
    model = Generative3DMAE(rngs=nnx.Rngs(100))
    x = jnp.arange(1 * 3 * 16 * 16 * 16 * 16, dtype=jnp.float32).reshape((1, 3, 16, 16, 16, 16))
    patches = model.patchify(x)

    assert patches.shape == (1, 192, 1024)

    # Check shell 0, patch (gx=0, gy=0, gz=0) -> token 0
    expected_patch0 = x[0, 0, :4, :4, :4, :].reshape(-1)
    assert jnp.array_equal(patches[0, 0], expected_patch0)

    # Check shell 1, patch (gx=1, gy=2, gz=3) -> token 64 + (1*16 + 2*4 + 3) = 64 + 27 = 91
    token_idx = 64 + (1 * 16 + 2 * 4 + 3)
    expected_patch91 = x[0, 1, 4:8, 8:12, 12:16, :].reshape(-1)
    assert jnp.array_equal(patches[0, token_idx], expected_patch91)

