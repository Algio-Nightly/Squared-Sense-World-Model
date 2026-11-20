"""Event-Driven Multi-LOD Shell Gating for inference acceleration.

Maintains host-side per-episode caches and conditionally skips 3D CNN and
Transformer evaluations when voxel differences delta_l <= tau_gate.
Strictly follows docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from models.transformer import MultiLODBackbone

# Computational cost constants in MFLOPs
MFLOPS_PER_SHELL: float = 128.2
MFLOPS_TRANSFORMER: float = 1359.0
DENSE_MFLOPS_PER_STEP: float = 3.0 * MFLOPS_PER_SHELL + MFLOPS_TRANSFORMER  # 1743.6 MFLOPs


@nnx.jit
def _encode_shell(backbone: MultiLODBackbone, shell: jax.Array) -> jax.Array:
    """JIT-compiled single-shell 3D CNN encoder."""
    return backbone.cnn.encode_single_shell(shell)


@nnx.jit
def _encode_transformer(backbone: MultiLODBackbone, tokens: jax.Array) -> jax.Array:
    """JIT-compiled Transformer encoder for concatenated multi-shell tokens."""
    return backbone.encode_tokens(tokens)


class EventDrivenShellGate:
    """Host-side event-driven gating wrapper around MultiLODBackbone.

    Evaluates shell 3D CNNs and the multi-LOD Transformer only when voxel
    temporal differences delta_l > tau_gate. Holds zero learnable parameters.
    """

    def __init__(
        self,
        backbone: MultiLODBackbone,
        tau_gate: float = 0.0,
    ) -> None:
        self.backbone = backbone
        self.tau_gate = tau_gate

        # Host-side per-episode cache
        self.prev_shells: Optional[jax.Array] = None
        self.prev_cnn_tokens: Optional[jax.Array] = None
        self.prev_z: Optional[jax.Array] = None

        # Diagnostic counters
        self.total_steps: int = 0
        self.shell_eval_counts: List[int] = [0, 0, 0]
        self.transformer_eval_count: int = 0

    def reset(self) -> None:
        """Clear cache and reset step/eval counters."""
        self.prev_shells = None
        self.prev_cnn_tokens = None
        self.prev_z = None
        self.total_steps = 0
        self.shell_eval_counts = [0, 0, 0]
        self.transformer_eval_count = 0

    def step(self, shells_single: jax.Array) -> jax.Array:
        """Run one inference step with event-driven shell gating.

        Args:
            shells_single: Input voxel shell array with shape (1, 3, 16, 16, 16, 16)
              or (3, 16, 16, 16, 16).

        Returns:
            Latent representation tensor z of shape (1, 192, 256).
        """
        if shells_single.ndim == 5:
            shells_single = jnp.expand_dims(shells_single, axis=0)

        # First step: evaluate all 3 shells + Transformer and populate cache
        if self.total_steps == 0 or self.prev_shells is None or self.prev_cnn_tokens is None:
            self.total_steps += 1
            toks = []
            for l in range(3):
                t = _encode_shell(self.backbone, shells_single[:, l])
                self.shell_eval_counts[l] += 1
                toks.append(t[0])

            self.prev_cnn_tokens = jnp.stack(toks, axis=0)  # (3, 64, 256)
            self.prev_shells = shells_single[0]  # (3, 16, 16, 16, 16)

            tokens = self.prev_cnn_tokens.reshape((1, 192, 256))
            self.prev_z = _encode_transformer(self.backbone, tokens)
            self.transformer_eval_count += 1
            return self.prev_z

        # Subsequent steps: compute temporal delta per shell
        self.total_steps += 1
        any_changed = False

        for l in range(3):
            delta_l = float(jnp.mean(jnp.abs(shells_single[0, l] - self.prev_shells[l])))
            if delta_l > self.tau_gate:
                any_changed = True
                t = _encode_shell(self.backbone, shells_single[:, l])
                self.shell_eval_counts[l] += 1
                self.prev_shells = self.prev_shells.at[l].set(shells_single[0, l])
                self.prev_cnn_tokens = self.prev_cnn_tokens.at[l].set(t[0])

        if any_changed:
            tokens = self.prev_cnn_tokens.reshape((1, 192, 256))
            self.prev_z = _encode_transformer(self.backbone, tokens)
            self.transformer_eval_count += 1

        assert self.prev_z is not None
        return self.prev_z

    def __call__(self, shells_single: jax.Array) -> jax.Array:
        """Alias for step(shells_single)."""
        return self.step(shells_single)

    def get_stats(self) -> Dict[str, Any]:
        """Compute execution statistics and FLOP savings."""
        total = max(1, self.total_steps)
        shell_skips = [1.0 - (c / total) for c in self.shell_eval_counts]
        trans_skip = 1.0 - (self.transformer_eval_count / total)
        empirical_mflops = (
            sum(self.shell_eval_counts) * MFLOPS_PER_SHELL
            + self.transformer_eval_count * MFLOPS_TRANSFORMER
        ) / total

        return {
            "total_steps": self.total_steps,
            "shell_eval_counts": list(self.shell_eval_counts),
            "transformer_eval_count": self.transformer_eval_count,
            "shell_skip_rates": shell_skips,
            "transformer_skip_rate": trans_skip,
            "mflops_per_step": empirical_mflops,
            "dense_mflops_per_step": DENSE_MFLOPS_PER_STEP,
        }

