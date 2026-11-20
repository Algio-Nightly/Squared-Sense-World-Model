"""Training loss functions for Squared Sense (Spatial JEPA, Temporal Dynamics, VICReg).

Strictly follows docs/TRAIN_CONTRACT.md using channel-last conventions and Optax.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax


def smooth_l1_loss(pred: jax.Array, target: jax.Array, beta: float = 1.0) -> jax.Array:
    """Compute elementwise Smooth L1 (Huber) loss averaged over all elements.

    Args:
        pred: Predicted tensor of any shape.
        target: Target tensor of identical shape.
        beta: Threshold at which to transition from L2 to L1 penalty.

    Returns:
        Scalar mean Smooth L1 loss.
    """
    diff = jnp.abs(pred - target)
    loss = jnp.where(diff < beta, 0.5 * (diff**2) / beta, diff - 0.5 * beta)
    return jnp.mean(loss)


def vicreg_var_cov_loss(
    z: jax.Array,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[jax.Array, jax.Array]:
    """Compute VICReg variance and covariance regularization penalties.

    Prevents informational collapse without contrastive negative pairs:
      - Variance loss: Enforces standard deviation of each feature dimension >= gamma.
      - Covariance loss: Penalizes off-diagonal correlations across feature dimensions.

    Args:
        z: Latent tensor of shape (..., D).
        gamma: Target standard deviation threshold (default 1.0).
        eps: Small epsilon for numerical stability.

    Returns:
        Tuple of (loss_var, loss_cov) scalar penalties.
    """
    z_flat = z.reshape((-1, z.shape[-1]))  # (N, D)
    N, D = z_flat.shape

    # 1. Variance regularization penalty
    std = jnp.sqrt(jnp.var(z_flat, axis=0) + eps)
    loss_var = jnp.mean(jax.nn.relu(gamma - std))

    # 2. Covariance regularization penalty
    z_c = z_flat - jnp.mean(z_flat, axis=0, keepdims=True)
    denom = jnp.maximum(N - 1, 1)
    cov = (z_c.T @ z_c) / denom  # (D, D)

    # Zero out diagonal elements
    diag_mask = jnp.eye(D, dtype=bool)
    cov_off_diag = jnp.where(diag_mask, 0.0, cov)
    loss_cov = jnp.sum(cov_off_diag**2) / D

    return loss_var, loss_cov


def spatial_jepa_loss(
    pred_targets: jax.Array,
    norm_ema_targets: jax.Array,
    context_tokens: jax.Array,
    alpha: float = 0.1,
    beta: float = 0.04,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute Stage 1A Spatial JEPA composite training loss.

    Combines:
      - Smooth-L1 prediction error on masked target latents
      - VICReg variance regularization on context representations (alpha = 0.1)
      - VICReg covariance decorrelation on context representations (beta = 0.04)

    Args:
        pred_targets: (B, N_tgt, D) predicted latents from JEPAPredictor.
        norm_ema_targets: (B, N_tgt, D) normalized target latents from EMA target encoder.
        context_tokens: (B, N_vis, D) visible tokens from MultiLODBackbone context encoder.
        alpha: Weight for VICReg variance penalty.
        beta: Weight for VICReg covariance penalty.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    pred_loss = smooth_l1_loss(pred_targets, norm_ema_targets)
    loss_var, loss_cov = vicreg_var_cov_loss(context_tokens)

    total_loss = pred_loss + alpha * loss_var + beta * loss_cov
    metrics = {
        "loss": total_loss,
        "pred_loss": pred_loss,
        "var_loss": loss_var,
        "cov_loss": loss_cov,
    }
    return total_loss, metrics


def temporal_jepa_loss(
    pred_future: jax.Array,
    ema_future_targets: jax.Array,
    idm_logits: jax.Array,
    actions: jax.Array,
    gamma_discount: float = 0.9,
    alpha: float = 0.1,
    beta: float = 0.04,
    omega: float = 0.2,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute Stage 1B Temporal JEPA composite training loss.

    Combines:
      - Discounted K-step autoregressive rollout L1 error (gamma_discount = 0.9)
      - Auxiliary Inverse Dynamics Model (IDM) cross-entropy (omega = 0.2)
      - VICReg variance & covariance regularizers on predicted future latents

    Args:
        pred_future: (B, T_valid, K, D) autoregressive future predictions.
        ema_future_targets: (B, T_valid, K, D) ground-truth / EMA pooled future latents.
        idm_logits: (B, T_idm, num_actions) action logits from IDM head.
        actions: (B, T) or (B, T_idm) discrete ground-truth actions.
        gamma_discount: Temporal discount factor per rollout step.
        alpha: Weight for VICReg variance penalty.
        beta: Weight for VICReg covariance penalty.
        omega: Weight for IDM cross-entropy penalty.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    # 1. Discounted K-step rollout L1 loss
    K = pred_future.shape[2]
    diff = jnp.abs(pred_future - ema_future_targets)
    discounts = (gamma_discount ** jnp.arange(K, dtype=jnp.float32))[None, None, :, None]
    rollout_l1 = jnp.mean(diff * discounts)

    # 2. Auxiliary IDM cross-entropy loss
    t_idm = idm_logits.shape[1]
    if actions.shape[-1] != t_idm:
        idm_actions = actions[:, :t_idm]
    else:
        idm_actions = actions

    idm_ce = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(idm_logits, idm_actions)
    )

    # 3. VICReg regularization on predicted future latents
    loss_var, loss_cov = vicreg_var_cov_loss(pred_future)

    total_loss = rollout_l1 + omega * idm_ce + alpha * loss_var + beta * loss_cov
    metrics = {
        "loss": total_loss,
        "rollout_l1": rollout_l1,
        "idm_loss": idm_ce,
        "var_loss": loss_var,
        "cov_loss": loss_cov,
    }
    return total_loss, metrics


def cosine_ema_momentum(
    step: int | jax.Array,
    total_steps: int,
    start: float = 0.996,
    end: float = 1.0,
) -> float:
    """Compute cosine schedule for EMA target momentum from start -> end.

    Args:
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        start: Starting momentum at step 0 (default 0.996).
        end: Final momentum at total_steps (default 1.0).

    Returns:
        Scalar momentum value in [start, end].
    """
    progress = jnp.clip(step / max(total_steps, 1), 0.0, 1.0)
    m = start + (end - start) * (0.5 * (1.0 - jnp.cos(jnp.pi * progress)))
    return float(m)
