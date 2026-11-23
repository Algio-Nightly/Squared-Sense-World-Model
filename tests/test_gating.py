"""Tests for Event-Driven Multi-LOD Shell Gating (models/gating.py)."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from models.gating import EventDrivenShellGate
from models.transformer import MultiLODBackbone


def test_event_driven_shell_gate_lossless_equivalence() -> None:
    """Verify lossless equivalence (tau_gate=0.0) against dense evaluation across 5 steps."""
    rngs = nnx.Rngs(42)
    backbone = MultiLODBackbone(rngs=rngs)
    gate = EventDrivenShellGate(backbone=backbone, tau_gate=0.0)

    # Construct the 5-step sequence
    key = jax.random.PRNGKey(101)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Step 1: Fresh initial shells
    s1 = jax.random.uniform(k1, (1, 3, 16, 16, 16, 16))

    # Step 2: Identical to Step 1 (all shells unchanged)
    s2 = s1

    # Step 3: Only LOD-0 changes
    s3 = s2.at[:, 0].set(jax.random.uniform(k2, (1, 16, 16, 16, 16)))

    # Step 4: LOD-0 & LOD-1 change (LOD-2 remains unchanged)
    s4 = (
        s3.at[:, 0]
        .set(jax.random.uniform(k3, (1, 16, 16, 16, 16)))
        .at[:, 1]
        .set(jax.random.uniform(k4, (1, 16, 16, 16, 16)))
    )

    # Step 5: Identical to Step 4 (all shells unchanged)
    s5 = s4

    sequence = [s1, s2, s3, s4, s5]

    for step_idx, shell_input in enumerate(sequence):
        gated_z = gate.step(shell_input)
        dense_z = backbone(shell_input)

        max_diff = float(jnp.max(jnp.abs(gated_z - dense_z)))
        print(f"Step {step_idx + 1}: max difference vs dense = {max_diff:.8e}")
        assert jnp.allclose(
            gated_z, dense_z, atol=5e-3
        ), f"Step {step_idx + 1} discrepancy: max diff {max_diff}"

    # Verify exact evaluation counts
    stats = gate.get_stats()
    print("\n[Gating Execution Stats]:", stats)

    assert stats["total_steps"] == 5
    # LOD-0 evaluated on steps 1, 3, 4 -> 3/5
    assert stats["shell_eval_counts"][0] == 3
    # LOD-1 evaluated on steps 1, 4 -> 2/5
    assert stats["shell_eval_counts"][1] == 2
    # LOD-2 evaluated on step 1 -> 1/5
    assert stats["shell_eval_counts"][2] == 1
    # Transformer evaluated on steps 1, 3, 4 -> 3/5
    assert stats["transformer_eval_count"] == 3

    # Check skip rates
    assert stats["shell_skip_rates"] == [pytest.approx(0.4), pytest.approx(0.6), pytest.approx(0.8)]
    assert stats["transformer_skip_rate"] == pytest.approx(0.4)

    # Verify empirical MFLOPs savings
    # Dense cost: 3 * 128.2 + 1359.0 = 1743.6 MFLOPs/step
    # Gated cost: (6 * 128.2 + 3 * 1359.0) / 5 = 969.24 MFLOPs/step
    assert stats["mflops_per_step"] < 1743.7
    assert stats["mflops_per_step"] == pytest.approx(969.24, rel=1e-3)


def test_event_driven_shell_gate_reset() -> None:
    """Verify reset() clears caches and resets all counters."""
    rngs = nnx.Rngs(7)
    backbone = MultiLODBackbone(rngs=rngs)
    gate = EventDrivenShellGate(backbone=backbone, tau_gate=0.0)

    x = jnp.ones((1, 3, 16, 16, 16, 16))
    _ = gate.step(x)
    assert gate.total_steps == 1
    assert gate.prev_shells is not None

    gate.reset()
    assert gate.total_steps == 0
    assert gate.prev_shells is None
    assert gate.prev_cnn_tokens is None
    assert gate.prev_z is None
    assert gate.shell_eval_counts == [0, 0, 0]
    assert gate.transformer_eval_count == 0


def test_event_driven_shell_gate_threshold() -> None:
    """Verify tau_gate filters out sub-threshold perturbations."""
    rngs = nnx.Rngs(13)
    backbone = MultiLODBackbone(rngs=rngs)
    gate = EventDrivenShellGate(backbone=backbone, tau_gate=0.05)

    s1 = jnp.ones((1, 3, 16, 16, 16, 16))
    _ = gate.step(s1)

    # Sub-threshold perturbation on LOD-0 (mean delta = 0.01 < 0.05)
    s2 = s1.at[:, 0].set(1.01)
    _ = gate.step(s2)

    stats = gate.get_stats()
    assert stats["total_steps"] == 2
    # Second step should be completely skipped
    assert stats["shell_eval_counts"] == [1, 1, 1]
    assert stats["transformer_eval_count"] == 1

