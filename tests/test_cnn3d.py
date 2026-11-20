"""Tests for Siamese 3D ConvNeXt encoder (models/cnn3d.py)."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from models.cnn3d import (
    ConvNeXt3DSEBlock,
    CoordConv3D,
    SiameseVoxelCNN,
    count_params,
)


def test_coordconv3d() -> None:
    """Verify CoordConv3D adds 3 coordinate channels with non-trainable state."""
    coord = CoordConv3D(size=16)
    x = jnp.zeros((2, 16, 16, 16, 16), dtype=jnp.float32)
    out = coord(x)
    assert out.shape == (2, 16, 16, 16, 19)

    # Must have 0 learnable parameters
    assert count_params(coord) == 0


def test_convnext_block() -> None:
    """Verify ConvNeXt3DSEBlock preserves spatial dimensions and channels."""
    rngs = nnx.Rngs(42)
    block = ConvNeXt3DSEBlock(dim=64, rngs=rngs)
    x = jnp.ones((2, 8, 8, 8, 64), dtype=jnp.float32)
    out = block(x)
    assert out.shape == (2, 8, 8, 8, 64)
    assert jnp.all(jnp.isfinite(out))


def test_siamese_voxel_cnn_encode_single_shell() -> None:
    """Verify encode_single_shell maps (2, 16, 16, 16, 16) -> (2, 64, 256)."""
    rngs = nnx.Rngs(0)
    model = SiameseVoxelCNN(rngs=rngs)
    shell = jnp.ones((2, 16, 16, 16, 16), dtype=jnp.float32)
    tokens = model.encode_single_shell(shell)

    assert tokens.shape == (2, 64, 256)
    assert jnp.all(jnp.isfinite(tokens))


def test_siamese_voxel_cnn_call_and_jit() -> None:
    """Verify __call__ on (2, 3, 16, 16, 16, 16) outputs (2, 192, 256) with zero NaNs and works in nnx.jit."""
    rngs = nnx.Rngs(1)
    model = SiameseVoxelCNN(rngs=rngs)
    x = jnp.ones((2, 3, 16, 16, 16, 16), dtype=jnp.float32)

    @nnx.jit
    def forward(m: SiameseVoxelCNN, inputs: jax.Array) -> jax.Array:
        return m(inputs)

    out = forward(model, x)
    assert out.shape == (2, 192, 256)
    assert not jnp.any(jnp.isnan(out))
    assert jnp.all(jnp.isfinite(out))


def test_param_count_budget() -> None:
    """Print count_params(model) and assert it is within ±3% of 470,560."""
    rngs = nnx.Rngs(2)
    model = SiameseVoxelCNN(rngs=rngs)
    total_params = count_params(model)
    target = 470_560

    print(f"\n[SiameseVoxelCNN Parameter Count]: {total_params} (target ~{target})")
    assert abs(total_params - target) / target <= 0.03, (
        f"Param count {total_params} deviates by more than 3% from target {target}"
    )


def test_gradients_finite_and_nonzero() -> None:
    """Run nnx.value_and_grad on dummy MSE loss and assert all parameter gradients are finite and non-zero."""
    rngs = nnx.Rngs(3)
    model = SiameseVoxelCNN(rngs=rngs)
    x = jnp.ones((2, 3, 16, 16, 16, 16), dtype=jnp.float32)

    def loss_fn(m: SiameseVoxelCNN, inputs: jax.Array) -> jax.Array:
        y = m(inputs)
        return jnp.mean((y - 0.5) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model, x)

    assert jnp.isfinite(loss)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(grad_leaves) > 0

    for idx, g in enumerate(grad_leaves):
        assert jnp.all(jnp.isfinite(g)), f"Non-finite grad in leaf {idx}"
        assert jnp.any(g != 0), f"All-zero grad in leaf {idx}"

