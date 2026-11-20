"""Frozen-Backbone PPO Policy Trainer for 3D Voxel Navigation.

Trains an ActorCriticHead MLP on top of strictly frozen perception representations:
  - Perception backbone is strictly frozen (zero gradients, stop_gradient / inference only).
  - Only ActorCriticHead (~50K params) is updated.
  - Standard clipped PPO with GAE advantages and AdamW optimizer.
  - Optional Behavioral Cloning (BC) warm-start for rapid initial convergence.
  - Evaluates on VoxelNavEnv courses.

Strictly follows docs/EVAL_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Any, Dict, List, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from eval.sim_env import VoxelNavEnv, make_debug_bridge_course
from models.baselines import Generative3DMAE, SingleLODJEPA
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import MultiLODBackbone, SpatialJEPA
from train.checkpoint import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# ActorCriticHead Architecture (~50K params)
# ---------------------------------------------------------------------------

class ActorCriticHead(nnx.Module):
    """Actor-Critic MLP head operating on frozen perception representations.

    Takes concat([frozen_feat, goal_vec_norm]) (feat_dim + 3) ->
    Linear(in, 128) -> GELU -> Linear(128, 128) -> GELU ->
    Actor Linear(128, 12) logits + Critic Linear(128, 1) value (~50K params for feat_dim=256).
    """

    def __init__(
        self,
        feat_dim: int,
        num_actions: int = 12,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_actions = num_actions
        in_dim = feat_dim + 3

        self.fc1 = nnx.Linear(in_dim, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 128, rngs=rngs)
        self.actor = nnx.Linear(128, num_actions, rngs=rngs)
        self.critic = nnx.Linear(128, 1, rngs=rngs)

    def __call__(
        self,
        feat: jax.Array,
        goal_vec: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass returning action logits and state value estimates.

        Args:
            feat: (B, feat_dim) or (feat_dim,) feature vector, or pre-concatenated with goal_vec_norm.
            goal_vec: Optional (B, 3) or (3,) 3D goal direction vector.

        Returns:
            Tuple of (action_logits, values):
              - action_logits: (B, num_actions)
              - values: (B,)
        """
        # Ensure 2D batch dimension
        if feat.ndim == 1:
            feat = jnp.expand_dims(feat, axis=0)

        if goal_vec is not None:
            if goal_vec.ndim == 1:
                goal_vec = jnp.expand_dims(goal_vec, axis=0)
            norm = jnp.linalg.norm(goal_vec, axis=-1, keepdims=True)
            goal_dir = goal_vec / (norm + 1e-6)
            x = jnp.concatenate([feat, goal_dir], axis=-1)
        else:
            x = feat

        h = nnx.gelu(self.fc1(x))
        h = nnx.gelu(self.fc2(h))
        logits = self.actor(h)
        values = jnp.squeeze(self.critic(h), axis=-1)
        return logits, values


# ---------------------------------------------------------------------------
# Frozen Feature Extraction Helper
# ---------------------------------------------------------------------------

def extract_step_feature(
    backbone: Any,
    model_type: str,
    shells: np.ndarray | jax.Array,
    state: Optional[np.ndarray | jax.Array] = None,
    action: Optional[int] = None,
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None,
    temporal_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Optional[Any]]:
    """Extract single-step frozen perception features with zero gradients.

    Args:
        backbone: Initialized/trained perception backbone module.
        model_type: In {"m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae"}.
        shells: (3, 16, 16, 16, 16) or (1, 3, 16, 16, 16, 16) float32 shells array in [0, 1].
        state: Optional (12,) proprioceptive state.
        action: Optional scalar action taken at previous step.
        temporal_model: Optional DualLSTMTemporalIntegrator for m1_full.
        temporal_hidden: Optional hidden state tuple for temporal model.

    Returns:
        Tuple of (feat, updated_temporal_hidden):
          - feat: (1, feat_dim) JAX array
          - updated_temporal_hidden: updated hidden state or None
    """
    shells_arr = jnp.asarray(shells, dtype=jnp.float32)
    if shells_arr.ndim == 5:
        shells_arr = jnp.expand_dims(shells_arr, axis=0)

    # 1. Static Feature Extraction (zero gradient stop_gradient)
    if model_type in ("m2_static", "m1_full"):
        encoder = backbone.context_encoder if hasattr(backbone, "context_encoder") else backbone
        tokens = jax.lax.stop_gradient(encoder(shells_arr, visible_idx=None))
        z_static = jnp.mean(tokens, axis=1)  # (1, 256)
    elif model_type == "b1_lod0":
        encoder = backbone.context_encoder if hasattr(backbone, "context_encoder") else backbone
        tokens = jax.lax.stop_gradient(encoder(shells_arr[:, 0:1], visible_idx=None))
        z_static = jnp.mean(tokens, axis=1)  # (1, 256)
    elif model_type == "b2_lod2":
        encoder = backbone.context_encoder if hasattr(backbone, "context_encoder") else backbone
        tokens = jax.lax.stop_gradient(encoder(shells_arr[:, 2:3], visible_idx=None))
        z_static = jnp.mean(tokens, axis=1)  # (1, 256)
    elif model_type == "b3_mae":
        encoder = backbone.encoder if hasattr(backbone, "encoder") else backbone
        tokens = jax.lax.stop_gradient(encoder(shells_arr, visible_idx=None))
        z_static = jnp.mean(tokens, axis=1)  # (1, 256)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 2. Temporal Fusion for m1_full
    if model_type == "m1_full":
        if temporal_model is None:
            raise ValueError("temporal_model is required for model_type='m1_full'")
        if temporal_hidden is None:
            temporal_hidden = temporal_model.init_hidden(batch_size=1)

        state_arr = jnp.zeros((1, 12), dtype=jnp.float32) if state is None else jnp.asarray(state, dtype=jnp.float32).reshape((1, 12))
        act_arr = jnp.zeros((1,), dtype=jnp.int32) if action is None else jnp.asarray([action], dtype=jnp.int32)

        _, c_t, next_hidden = temporal_model.step(tokens, state_arr, act_arr, temporal_hidden)
        c_t = jax.lax.stop_gradient(c_t)  # (1, 192)
        feat = jnp.concatenate([z_static, c_t], axis=-1)  # (1, 448)
        return feat, next_hidden

    return z_static, None


# ---------------------------------------------------------------------------
# GAE Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and discounted targets.

    Args:
        rewards: (T,) array of step rewards.
        values: (T,) array of critic value estimates.
        dones: (T,) boolean array of episode termination flags.
        next_value: Scalar critic estimate for bootstrap state at T.
        gamma: Discount factor (default 0.99).
        gae_lambda: GAE smoothing parameter (default 0.95).

    Returns:
        Tuple of (advantages, returns) both shaped (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO & BC Loss & Optimization Steps
# ---------------------------------------------------------------------------

@nnx.jit
def ppo_update_step(
    policy: ActorCriticHead,
    optimizer: nnx.Optimizer,
    feats: jax.Array,
    goal_vecs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Execute one clipped PPO mini-batch update step on ActorCriticHead.

    Backbone is NOT involved here; strictly updating policy parameters only.
    """

    def loss_fn(model: ActorCriticHead):
        logits, values = model(feats, goal_vecs)
        log_probs_all = jax.nn.log_softmax(logits)
        log_probs = jnp.take_along_axis(
            log_probs_all, jnp.expand_dims(actions, -1), axis=-1
        ).squeeze(-1)

        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs_all, axis=-1).mean()

        ratio = jnp.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -jnp.minimum(surr1, surr2).mean()

        value_loss = 0.5 * jnp.mean((values - returns) ** 2)
        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (policy_loss, value_loss, entropy)

    (total_loss, (p_loss, v_loss, ent)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(policy)
    optimizer.update(policy, grads)
    return total_loss, p_loss, v_loss, ent


@nnx.jit
def bc_update_step(
    policy: ActorCriticHead,
    optimizer: nnx.Optimizer,
    feats: jax.Array,
    goal_vecs: jax.Array,
    target_actions: jax.Array,
) -> jax.Array:
    """Execute one Behavioral Cloning mini-batch cross-entropy update step."""

    def loss_fn(model: ActorCriticHead):
        logits, _ = model(feats, goal_vecs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, target_actions
        ).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(policy)
    optimizer.update(policy, grads)
    return loss


# ---------------------------------------------------------------------------
# Behavioral Cloning Warm-Start
# ---------------------------------------------------------------------------

def train_bc_warmstart(
    policy: ActorCriticHead,
    optimizer: nnx.Optimizer,
    dataset: Optional[Dict[str, np.ndarray]] = None,
    env: Optional[VoxelNavEnv] = None,
    backbone: Optional[Any] = None,
    model_type: str = "m2_static",
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> List[float]:
    """Train ActorCriticHead via Behavioral Cloning cross-entropy minimization.

    Args:
        policy: ActorCriticHead instance.
        optimizer: Optax optimizer wrapping policy.
        dataset: Optional dict with {"feats": (N, D), "goal_vecs": (N, 3), "actions": (N,)}.
        env: Optional environment to generate expert demos if dataset is None.
        backbone: Perception backbone used if collecting demos.
        model_type: Model type identifier.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.

    Returns:
        List of epoch cross-entropy losses.
    """
    if dataset is None:
        if env is None or backbone is None:
            return []
        # Generate simple expert trajectory (straight walk on bridge, action 0)
        feats_list = []
        goals_list = []
        acts_list = []
        obs = env.reset()
        for _ in range(35):
            feat, _ = extract_step_feature(backbone, model_type, obs["shells"], obs["state"])
            feats_list.append(np.asarray(feat[0]))
            goals_list.append(np.asarray(obs["goal_vec"]))
            acts_list.append(0)  # Action 0: Forward
            obs, _, done, _ = env.step(action=0)
            if done:
                break
        dataset = {
            "feats": np.stack(feats_list, axis=0),
            "goal_vecs": np.stack(goals_list, axis=0),
            "actions": np.array(acts_list, dtype=np.int32),
        }

    feats = jnp.asarray(dataset["feats"], dtype=jnp.float32)
    goals = jnp.asarray(dataset["goal_vecs"], dtype=jnp.float32)
    actions = jnp.asarray(dataset["actions"], dtype=jnp.int32)
    n_samples = len(actions)

    rng = np.random.default_rng(seed)
    epoch_losses = []

    for ep in range(epochs):
        indices = rng.permutation(n_samples)
        batch_losses = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            b_idx = indices[start_idx:end_idx]
            b_f = feats[b_idx]
            b_g = goals[b_idx]
            b_a = actions[b_idx]
            loss = bc_update_step(policy, optimizer, b_f, b_g, b_a)
            batch_losses.append(float(loss))
        mean_loss = float(np.mean(batch_losses))
        epoch_losses.append(mean_loss)

    return epoch_losses


# ---------------------------------------------------------------------------
# Clipped PPO Training Loop
# ---------------------------------------------------------------------------

def train_ppo(
    env: VoxelNavEnv,
    backbone: Any,
    policy: ActorCriticHead,
    optimizer: nnx.Optimizer,
    model_type: str = "m2_static",
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None,
    total_steps: int = 5000,
    rollout_steps: int = 128,
    ppo_epochs: int = 4,
    batch_size: int = 32,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    save_dir: str = "checkpoints/ppo_nav",
    smoke_test: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train ActorCriticHead via clipped PPO with strictly frozen perception backbone.

    Args:
        env: Voxel navigation simulation environment.
        backbone: Frozen perception backbone (no gradients).
        policy: Trainable ActorCriticHead.
        optimizer: Optax optimizer wrapping policy.
        model_type: Model identifier.
        temporal_model: Optional temporal model for m1_full.
        total_steps: Total environment steps.
        rollout_steps: Transition steps per PPO rollout buffer.
        ppo_epochs: Optimization epochs per rollout.
        batch_size: PPO mini-batch size.
        clip_eps: PPO clipping epsilon (default 0.2).
        gamma: Discount factor (default 0.99).
        gae_lambda: GAE parameter (default 0.95).
        ent_coef: Policy entropy bonus coefficient (default 0.01).
        vf_coef: Value function loss coefficient (default 0.5).
        save_dir: Path to directory for saving checkpoint and metrics.
        smoke_test: If True, executes exactly 2 updates with small buffer for test.
        seed: Random seed.

    Returns:
        Summary dictionary containing training diagnostics.
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "ppo_metrics.jsonl")

    # Clear previous metrics if exists
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    key = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    if smoke_test:
        total_steps = 32
        rollout_steps = 16
        ppo_epochs = 2
        batch_size = 16

    num_updates = max(1, total_steps // rollout_steps)
    if smoke_test:
        num_updates = 2

    current_obs = env.reset()
    temporal_hidden = None
    step_count = 0
    best_reward = -float("inf")
    metrics_history = []

    print(f"Starting PPO Training: {num_updates} updates, {rollout_steps} steps/rollout, model={model_type}")

    for update_idx in range(num_updates):
        # 1. Collect Rollout
        b_feats = []
        b_goals = []
        b_actions = []
        b_log_probs = []
        b_rewards = []
        b_dones = []
        b_values = []

        episode_rewards = []
        current_ep_reward = 0.0

        for step in range(rollout_steps):
            step_count += 1
            feat, temporal_hidden = extract_step_feature(
                backbone=backbone,
                model_type=model_type,
                shells=current_obs["shells"],
                state=current_obs["state"],
                temporal_model=temporal_model,
                temporal_hidden=temporal_hidden,
            )
            goal_v = current_obs["goal_vec"]

            logits, value = policy(feat, goal_v)
            key, subkey = jax.random.split(key)
            action = int(jax.random.categorical(subkey, logits[0]))
            log_prob = float(jax.nn.log_softmax(logits[0])[action])
            v_est = float(value[0])

            next_obs, reward, done, info = env.step(action)
            current_ep_reward += reward

            b_feats.append(np.asarray(feat[0]))
            b_goals.append(np.asarray(goal_v))
            b_actions.append(action)
            b_log_probs.append(log_prob)
            b_rewards.append(reward)
            b_dones.append(done)
            b_values.append(v_est)

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                current_obs = env.reset()
                temporal_hidden = None
            else:
                current_obs = next_obs

        # Compute next_value for GAE bootstrap
        if b_dones[-1]:
            next_value = 0.0
        else:
            final_feat, _ = extract_step_feature(
                backbone=backbone,
                model_type=model_type,
                shells=current_obs["shells"],
                state=current_obs["state"],
                temporal_model=temporal_model,
                temporal_hidden=temporal_hidden,
            )
            _, final_val = policy(final_feat, current_obs["goal_vec"])
            next_value = float(final_val[0])

        # Compute Advantages and Returns
        arr_rewards = np.array(b_rewards, dtype=np.float32)
        arr_values = np.array(b_values, dtype=np.float32)
        arr_dones = np.array(b_dones, dtype=bool)

        advantages, returns = compute_gae(
            rewards=arr_rewards,
            values=arr_values,
            dones=arr_dones,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        # Convert buffer to JAX arrays
        j_feats = jnp.asarray(b_feats, dtype=jnp.float32)
        j_goals = jnp.asarray(b_goals, dtype=jnp.float32)
        j_actions = jnp.asarray(b_actions, dtype=jnp.int32)
        j_old_log_probs = jnp.asarray(b_log_probs, dtype=jnp.float32)
        j_advantages = jnp.asarray(norm_advantages, dtype=jnp.float32)
        j_returns = jnp.asarray(returns, dtype=jnp.float32)

        # 2. PPO Optimization Epochs
        epoch_losses = []
        epoch_p_losses = []
        epoch_v_losses = []
        epoch_ents = []

        n_samples = rollout_steps
        for _ in range(ppo_epochs):
            perm = np_rng.permutation(n_samples)
            for s_idx in range(0, n_samples, batch_size):
                e_idx = min(s_idx + batch_size, n_samples)
                mb_idx = perm[s_idx:e_idx]

                t_loss, p_loss, v_loss, ent = ppo_update_step(
                    policy=policy,
                    optimizer=optimizer,
                    feats=j_feats[mb_idx],
                    goal_vecs=j_goals[mb_idx],
                    actions=j_actions[mb_idx],
                    old_log_probs=j_old_log_probs[mb_idx],
                    advantages=j_advantages[mb_idx],
                    returns=j_returns[mb_idx],
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                )
                epoch_losses.append(float(t_loss))
                epoch_p_losses.append(float(p_loss))
                epoch_v_losses.append(float(v_loss))
                epoch_ents.append(float(ent))

        mean_loss = float(np.mean(epoch_losses))
        mean_p_loss = float(np.mean(epoch_p_losses))
        mean_v_loss = float(np.mean(epoch_v_losses))
        mean_ent = float(np.mean(epoch_ents))
        mean_rew = float(np.mean(arr_rewards))

        record = {
            "update": update_idx + 1,
            "total_steps": step_count,
            "loss": mean_loss,
            "policy_loss": mean_p_loss,
            "value_loss": mean_v_loss,
            "entropy": mean_ent,
            "mean_step_reward": mean_rew,
        }
        metrics_history.append(record)

        # Append to ppo_metrics.jsonl
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Update {update_idx + 1}/{num_updates} | "
            f"Steps: {step_count} | Loss: {mean_loss:.4f} | "
            f"Policy: {mean_p_loss:.4f} | Value: {mean_v_loss:.4f} | "
            f"Entropy: {mean_ent:.4f} | MeanRew: {mean_rew:.3f}"
        )

        # Save policy_best.pkl
        if mean_rew >= best_reward or update_idx == num_updates - 1:
            best_reward = mean_rew
            save_checkpoint(
                os.path.join(save_dir, "policy_best.pkl"),
                policy,
                metadata={"model_type": model_type, "update": update_idx + 1, "mean_reward": mean_rew},
            )

    return {
        "final_loss": mean_loss,
        "best_reward": best_reward,
        "total_steps": step_count,
        "metrics_history": metrics_history,
    }


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entry Point
# ---------------------------------------------------------------------------

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for train_ppo_nav."""
    parser = argparse.ArgumentParser(
        description="Frozen-Backbone PPO Navigation Trainer for Squared Sense"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="m2_static",
        choices=["m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae"],
        help="Frozen perception backbone architecture.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to trained backbone checkpoint (.pkl).",
    )
    parser.add_argument(
        "--temporal_ckpt",
        type=str,
        default=None,
        help="Path to trained Dual-LSTM checkpoint (.pkl) for m1_full.",
    )
    parser.add_argument(
        "--courses_dir",
        type=str,
        default=None,
        help="Path to directory containing custom course volumes.",
    )
    parser.add_argument(
        "--region_dir",
        type=str,
        default=None,
        help="Path to directory containing baked Minecraft region memmaps (.mmap).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/ppo_nav",
        help="Directory to save checkpoints and training logs.",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=5000,
        help="Total environment interaction steps.",
    )
    parser.add_argument(
        "--bc_epochs",
        type=int,
        default=0,
        help="Number of Behavioral Cloning warm-start epochs (0 to skip).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke test with 2 PPO updates on debug bridge course.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Target acceleration device (auto, gpu, cpu).",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """CLI entrypoint for train_ppo_nav."""
    opts = parse_args(args)

    # Configure GPU memory and platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if opts.device != "auto":
        jax.config.update("jax_platform_name", opts.device)

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"\n=======================================================")
    print(f"Squared Sense World Model — PPO Policy Trainer")
    print(f"JAX Backend: {backend.upper()} | Devices: {devices}")
    print(f"=======================================================\n")

    rng = nnx.Rngs(opts.seed)

    # 1. Instantiate Perception Backbone
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None
    if opts.model_type == "m2_static":
        backbone = SpatialJEPA(num_shells=3, rngs=rng)
        feat_dim = 256
    elif opts.model_type == "m1_full":
        backbone = SpatialJEPA(num_shells=3, rngs=rng)
        temporal_model = DualLSTMTemporalIntegrator(rngs=rng)
        feat_dim = 448
    elif opts.model_type == "b1_lod0":
        backbone = SingleLODJEPA(lod_index=0, rngs=rng)
        feat_dim = 256
    elif opts.model_type == "b2_lod2":
        backbone = SingleLODJEPA(lod_index=2, rngs=rng)
        feat_dim = 256
    elif opts.model_type == "b3_mae":
        backbone = Generative3DMAE(rngs=rng)
        feat_dim = 256
    else:
        raise ValueError(f"Unknown model_type: {opts.model_type}")

    if opts.ckpt and os.path.exists(opts.ckpt):
        print(f"Loading backbone checkpoint from {opts.ckpt}...")
        load_checkpoint(opts.ckpt, backbone)

    if temporal_model and opts.temporal_ckpt and os.path.exists(opts.temporal_ckpt):
        print(f"Loading temporal checkpoint from {opts.temporal_ckpt}...")
        load_checkpoint(opts.temporal_ckpt, temporal_model)

    # 2. Instantiate Environment
    if opts.region_dir and os.path.exists(opts.region_dir):
        print(f"Loading real Minecraft environment from: {opts.region_dir}...")
        from env.minecraft_env import MinecraftNavEnv
        env = MinecraftNavEnv(region_dir=opts.region_dir)
    else:
        volume, start, waypoints = make_debug_bridge_course(seed=opts.seed)
        env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)

    # 3. Instantiate Trainable ActorCriticHead & Optax Optimizer
    policy = ActorCriticHead(feat_dim=feat_dim, num_actions=12, rngs=rng)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(learning_rate=3e-4, weight_decay=1e-4),
    )
    optimizer = nnx.Optimizer(policy, tx, wrt=nnx.Param)

    # 4. Optional BC Warm-Start
    if opts.bc_epochs > 0:
        print(f"Running BC Warm-Start for {opts.bc_epochs} epochs...")
        bc_losses = train_bc_warmstart(
            policy=policy,
            optimizer=optimizer,
            env=env,
            backbone=backbone,
            model_type=opts.model_type,
            epochs=opts.bc_epochs,
            seed=opts.seed,
        )
        print(f"BC Warm-Start completed: final loss = {bc_losses[-1]:.4f}")

    # 5. PPO Fine-Tuning
    results = train_ppo(
        env=env,
        backbone=backbone,
        policy=policy,
        optimizer=optimizer,
        model_type=opts.model_type,
        temporal_model=temporal_model,
        total_steps=opts.total_steps,
        save_dir=opts.save_dir,
        smoke_test=opts.smoke_test,
        seed=opts.seed,
    )
    print(f"PPO Training successfully finished. Results: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
