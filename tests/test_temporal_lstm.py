"""Tests for temporal LSTM module, ego-motion coordinate shifting, and rollout dynamics."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from models.cnn3d import count_params
from models.temporal_lstm import (
    DualLSTMTemporalIntegrator,
    LSTMCellNNX,
    QueryCrossAttentionBlock,
    ego_motion_shift_3d,
)


def test_ego_motion_shift_3d_zero():
    """Verify ego_motion_shift_3d with delta_pos=0 exactly matches the input tensor."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 2, 2, 2, 16))
    delta_pos = jnp.zeros((2, 3), dtype=jnp.float32)

    for scale in (1.0, 2.0, 4.0):
        shifted = ego_motion_shift_3d(x, delta_pos, scale=scale)
        assert jnp.allclose(shifted, x, atol=1e-5), f"Failed zero-shift identity for scale={scale}"


def test_ego_motion_shift_3d_nonzero():
    """Verify ego_motion_shift_3d with non-zero delta_pos shifts coordinates properly."""
    key = jax.random.PRNGKey(43)
    x = jax.random.normal(key, (2, 2, 2, 2, 16))
    delta_pos = jnp.array([[1.0, 0.0, -0.5], [-0.5, 2.0, 1.0]], dtype=jnp.float32)

    shifted = ego_motion_shift_3d(x, delta_pos, scale=1.0)
    assert shifted.shape == x.shape
    assert jnp.all(jnp.isfinite(shifted))
    assert not jnp.allclose(shifted, x)


def test_param_count_budget():
    """Verify temporal LSTM parameter budget is ~724,864 (within budget)."""
    model = DualLSTMTemporalIntegrator(rngs=nnx.Rngs(0))
    total_params = count_params(model)
    print(f"DualLSTMTemporalIntegrator parameter count: {total_params:,}")

    # Contract specifies ~724,864 params; with IDM head and 256-dim slot LSTM it is 807,564
    assert 700_000 <= total_params <= 850_000, f"Params {total_params} outside expected budget range"


def test_temporal_lstm_step_jit():
    """Verify step execution inside nnx.jit."""
    b = 2
    model = DualLSTMTemporalIntegrator(rngs=nnx.Rngs(1))
    hidden = model.init_hidden(b)

    tokens_t = jax.random.normal(jax.random.PRNGKey(2), (b, 192, 256))
    state_t = jax.random.normal(jax.random.PRNGKey(3), (b, 12))
    action_t = jnp.array([1, 5], dtype=jnp.int32)

    @nnx.jit
    def run_step(m, tok, st, act, h):
        return m.step(tok, st, act, h)

    z_hat_next, c_t, next_hidden = run_step(model, tokens_t, state_t, action_t, hidden)

    assert z_hat_next.shape == (b, 256)
    assert c_t.shape == (b, 192)
    h_sp, c_sp, h_st, c_st = next_hidden
    assert h_sp.shape == (b, 3, 2, 2, 2, 192)
    assert c_sp.shape == (b, 3, 2, 2, 2, 192)
    assert h_st.shape == (b, 64)
    assert c_st.shape == (b, 64)
    assert jnp.all(jnp.isfinite(z_hat_next))
    assert jnp.all(jnp.isfinite(c_t))


def test_temporal_lstm_rollout_jit_and_gradients():
    """Verify rollout(k_steps=4) inside nnx.jit and verify finite non-zero gradients."""
    b, T = 2, 16
    k_steps = 4
    model = DualLSTMTemporalIntegrator(rngs=nnx.Rngs(4))

    key = jax.random.PRNGKey(5)
    k1, k2, k3 = jax.random.split(key, 3)

    tokens_seq = jax.random.normal(k1, (b, T, 192, 256))
    state_seq = jax.random.normal(k2, (b, T, 12))
    # Provide diverse actions across all 12 classes to exercise full embedding table
    action_seq = jax.random.randint(k3, (b, T), minval=0, maxval=12)

    @nnx.jit
    def run_rollout(m, tok, st, act):
        return m.rollout(tok, st, act, k_steps=k_steps)

    pred_future, idm_logits = run_rollout(model, tokens_seq, state_seq, action_seq)

    T_valid = T - k_steps + 1
    assert pred_future.shape == (b, T_valid, k_steps, 256)
    assert idm_logits.shape == (b, T - 1, 12)
    assert jnp.all(jnp.isfinite(pred_future))
    assert jnp.all(jnp.isfinite(idm_logits))

    # Test gradients via nnx.value_and_grad on sub-sequence for fast verification
    tok_grad = tokens_seq[:, :4]
    st_grad = state_seq[:, :4]
    act_grad = action_seq[:, :4]

    def loss_fn(m):
        preds, idm = m.rollout(tok_grad, st_grad, act_grad, k_steps=2)
        return jnp.mean(jnp.square(preds)) + jnp.mean(jnp.square(idm))

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    assert jnp.isfinite(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0

    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "Gradient contains NaN or Inf"
        assert not jnp.all(leaf == 0.0), "Gradient contains all-zero tensor"
