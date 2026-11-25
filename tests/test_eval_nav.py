"""Tests for Frozen-Backbone PPO Trainer and Zero-Shot Evaluator.

Strictly follows docs/EVAL_CONTRACT.md and docs/MODEL_CONTRACT.md.
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

from eval.eval_navigation import evaluate_navigation, main as eval_nav_main
from eval.sim_env import VoxelNavEnv, make_debug_bridge_course
from eval.train_ppo_nav import (
    ActorCriticHead,
    main as train_ppo_main,
    train_bc_warmstart,
    train_ppo,
)
from models.transformer import SpatialJEPA
from train.checkpoint import load_checkpoint


def test_actor_critic_head_architecture() -> None:
    """Verify ActorCriticHead parameter count (~50K) and output shapes."""
    rng = nnx.Rngs(42)
    # For m2_static: feat_dim=256
    policy = ActorCriticHead(feat_dim=256, num_actions=12, rngs=rng)

    # Count parameters
    state = nnx.state(policy, nnx.Param)
    pure_dict = nnx.to_pure_dict(state)
    n_params = sum(x.size for x in jax.tree.leaves(pure_dict))

    # Expect ~51,469 parameters (target ~50K)
    assert 50_000 <= n_params <= 53_000, f"Expected ~50K params, got {n_params}"

    # Forward pass test with batch size 4
    batch_feat = jnp.ones((4, 256), dtype=jnp.float32)
    batch_goal = jnp.array([[0.0, 0.0, 36.0]] * 4, dtype=jnp.float32)

    logits, values = policy(batch_feat, batch_goal)
    assert logits.shape == (4, 12)
    assert values.shape == (4,)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.all(jnp.isfinite(values))


def test_bc_warmstart_loss_decrease() -> None:
    """Verify train_bc_warmstart runs and cross-entropy loss decreases."""
    rng = nnx.Rngs(42)
    policy = ActorCriticHead(feat_dim=256, num_actions=12, rngs=rng)
    tx = optax.adamw(learning_rate=1e-3)
    optimizer = nnx.Optimizer(policy, tx, wrt=nnx.Param)

    # Generate synthetic target dataset
    np_rng = np.random.default_rng(42)
    dataset = {
        "feats": np_rng.normal(size=(64, 256)).astype(np.float32),
        "goal_vecs": np.array([[0.0, 0.0, 1.0]] * 64, dtype=np.float32),
        "actions": np.zeros(64, dtype=np.int32),  # Target action 0
    }

    losses = train_bc_warmstart(
        policy=policy,
        optimizer=optimizer,
        dataset=dataset,
        epochs=4,
        batch_size=16,
        seed=42,
    )
    assert len(losses) == 4
    for l in losses:
        assert math.isfinite(l)
    assert losses[-1] < losses[0], f"Loss did not decrease: start={losses[0]}, end={losses[-1]}"


def test_train_ppo_smoke_and_bit_identical_backbone(tmp_path) -> None:
    """Run train_ppo_nav.py --smoke_test into tmp_path:

    - Assert policy_best.pkl is created
    - Assert losses in ppo_metrics.jsonl are finite
    - Assert frozen backbone parameters are bit-identical before and after PPO updates
    """
    save_dir = str(tmp_path / "ppo_smoke")
    seed = 42
    rng = nnx.Rngs(seed)

    # 1. Instantiate reference backbone and record exact snapshot of parameter arrays
    backbone = SpatialJEPA(num_shells=3, rngs=rng)
    state_before = nnx.to_pure_dict(nnx.state(backbone, nnx.Param))
    leaves_before = [np.copy(np.asarray(x)) for x in jax.tree.leaves(state_before)]

    # 2. Run train_ppo directly with this backbone
    volume, start, waypoints = make_debug_bridge_course(seed=seed)
    env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)
    policy = ActorCriticHead(feat_dim=256, num_actions=12, rngs=rng)
    optimizer = nnx.Optimizer(policy, optax.adamw(learning_rate=3e-4), wrt=nnx.Param)

    results = train_ppo(
        env=env,
        backbone=backbone,
        policy=policy,
        optimizer=optimizer,
        model_type="m2_static",
        total_steps=32,
        rollout_steps=16,
        ppo_epochs=2,
        batch_size=16,
        save_dir=save_dir,
        smoke_test=True,
        seed=seed,
    )

    # 3. Assert policy_best.pkl is created
    policy_pkl = os.path.join(save_dir, "policy_best.pkl")
    assert os.path.isfile(policy_pkl), f"Expected checkpoint at {policy_pkl}"

    # 4. Assert losses in ppo_metrics.jsonl are finite
    metrics_jsonl = os.path.join(save_dir, "ppo_metrics.jsonl")
    assert os.path.isfile(metrics_jsonl), f"Expected metrics at {metrics_jsonl}"
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 2, f"Expected at least 2 updates, got {len(lines)}"
    for record in lines:
        for key in ["loss", "policy_loss", "value_loss", "entropy", "mean_step_reward"]:
            assert key in record
            assert math.isfinite(record[key]), f"Non-finite {key}: {record[key]}"

    # 5. Assert frozen backbone parameters are bit-identical before and after PPO updates
    state_after = nnx.to_pure_dict(nnx.state(backbone, nnx.Param))
    leaves_after = [np.asarray(x) for x in jax.tree.leaves(state_after)]
    assert len(leaves_before) == len(leaves_after)
    for idx, (b_arr, a_arr) in enumerate(zip(leaves_before, leaves_after)):
        diff = np.max(np.abs(b_arr - a_arr))
        assert diff == 0.0, f"Backbone parameter leaf {idx} modified during PPO! Max diff: {diff}"


def test_eval_navigation_gating_comparison(tmp_path) -> None:
    """Run eval_navigation.py on saved smoke checkpoint with --use_gating ON and OFF:

    - Assert both produce valid JSON metrics
    - Assert required metric keys exist with finite values
    - Assert gating reduces empirical MFLOPs
    """
    save_dir = str(tmp_path / "ppo_smoke")
    policy_pkl = os.path.join(save_dir, "policy_best.pkl")

    # If policy doesn't exist, create it via smoke CLI
    if not os.path.isfile(policy_pkl):
        ret = train_ppo_main([
            "--smoke_test",
            "--model_type", "m2_static",
            "--save_dir", save_dir,
        ])
        assert ret == 0

    json_dense = str(tmp_path / "nav_dense.json")
    json_gated = str(tmp_path / "nav_gated.json")

    # 1. Run eval_navigation with gating OFF
    ret_dense = eval_nav_main([
        "--policy_ckpt", policy_pkl,
        "--model_type", "m2_static",
        "--smoke_test",
        "--out_json", json_dense,
    ])
    assert ret_dense == 0
    assert os.path.isfile(json_dense)

    # 2. Run eval_navigation with gating ON
    ret_gated = eval_nav_main([
        "--policy_ckpt", policy_pkl,
        "--model_type", "m2_static",
        "--use_gating",
        "--tau_gate", "0.01",
        "--smoke_test",
        "--out_json", json_gated,
    ])
    assert ret_gated == 0
    assert os.path.isfile(json_gated)

    # 3. Read both JSON results and check metric keys
    with open(json_dense, "r", encoding="utf-8") as f:
        metrics_dense = json.load(f)
    with open(json_gated, "r", encoding="utf-8") as f:
        metrics_gated = json.load(f)

    required_keys = [
        "success_rate_pct",
        "spl",
        "spl_pct",
        "waypoint_completion_pct",
        "hazard_collision_rate_pct",
        "empirical_mflops_per_step",
        "latency_ms_per_step",
        "use_gating",
        "total_episodes",
        "total_steps",
    ]
    for key in required_keys:
        assert key in metrics_dense, f"Missing key {key} in dense metrics"
        assert key in metrics_gated, f"Missing key {key} in gated metrics"
        assert math.isfinite(float(metrics_dense[key])), f"Non-finite value for {key}"
        assert math.isfinite(float(metrics_gated[key])), f"Non-finite value for {key}"

    # 4. Verify gating strictly reduces empirical MFLOPs
    mflops_dense = metrics_dense["empirical_mflops_per_step"]
    mflops_gated = metrics_gated["empirical_mflops_per_step"]

    print(f"Dense MFLOPs/step: {mflops_dense:.1f} | Gated MFLOPs/step: {mflops_gated:.1f}")
    assert mflops_dense == pytest.approx(1770.06, rel=1e-3)
    assert mflops_gated < mflops_dense, f"Gated MFLOPs ({mflops_gated}) must be strictly less than dense ({mflops_dense})"
