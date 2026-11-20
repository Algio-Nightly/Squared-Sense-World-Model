"""Zero-Shot Course Evaluator & Latency / Gating Profiler for 3D Voxel Navigation.

Evaluates a trained (backbone, policy) over test navigation courses:
  - Measures: Success Rate (SR %), SPL, Waypoint Completion %, Hazard Collision Rate %
  - Profiles: Empirical MFLOPs/step and wall-clock Latency (ms/step)
  - Tests Event-Driven Shell Gating acceleration (--use_gating) against dense baseline
  - Outputs structured JSON metrics summary table.

Strictly follows docs/EVAL_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from eval.sim_env import VoxelNavEnv, make_debug_bridge_course
from eval.train_ppo_nav import ActorCriticHead, extract_step_feature
from models.baselines import Generative3DMAE, SingleLODJEPA
from models.gating import DENSE_MFLOPS_PER_STEP, EventDrivenShellGate
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import MultiLODBackbone, SpatialJEPA
from train.checkpoint import load_checkpoint


def load_courses_from_dir(
    courses_dir: str,
) -> List[Tuple[np.ndarray, Tuple[int, int, int], List[Tuple[int, int, int]], float]]:
    """Load evaluation courses from .npz files in directory."""
    course_files = sorted(glob.glob(os.path.join(courses_dir, "*.npz")))
    courses = []
    for fpath in course_files:
        data = np.load(fpath)
        vol = data["volume"]
        start = tuple(map(int, data["start"]))
        waypoints = [tuple(map(int, wp)) for wp in data["waypoints"]]
        # Compute reference shortest path length
        l_ref = float(data.get("shortest_path_len", 0.0))
        if l_ref <= 0.0:
            curr = np.array(start, dtype=float)
            total_d = 0.0
            for wp in waypoints:
                total_d += float(np.linalg.norm(np.array(wp, dtype=float) - curr))
                curr = np.array(wp, dtype=float)
            l_ref = max(1.0, total_d)
        courses.append((vol, start, waypoints, l_ref))
    return courses


def evaluate_navigation(
    backbone: Any,
    policy: ActorCriticHead,
    model_type: str = "m2_static",
    temporal_model: Optional[DualLSTMTemporalIntegrator] = None,
    courses: Optional[List[Tuple[np.ndarray, Tuple[int, int, int], List[Tuple[int, int, int]], float]]] = None,
    num_episodes: int = 5,
    max_steps: int = 250,
    use_gating: bool = False,
    tau_gate: float = 0.05,
    greedy: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate trained policy on navigation courses and profile compute / latency.

    Args:
        backbone: Frozen perception backbone module.
        policy: Trained ActorCriticHead policy.
        model_type: In {"m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae"}.
        temporal_model: Optional DualLSTMTemporalIntegrator for m1_full.
        courses: Optional list of (volume, start, waypoints, shortest_path_len) tuples.
        num_episodes: Number of episodes to evaluate.
        max_steps: Maximum steps allowed per episode.
        use_gating: If True, uses EventDrivenShellGate to conditionally skip compute.
        tau_gate: Threshold for EventDrivenShellGate.
        greedy: If True, takes argmax action; otherwise samples.
        seed: Random seed.

    Returns:
        Structured evaluation metrics dictionary.
    """
    key = jax.random.PRNGKey(seed)

    # 1. Prepare Courses
    if not courses:
        # Generate debug bridge courses with different seeds
        courses = []
        for i in range(num_episodes):
            vol, start, wps = make_debug_bridge_course(seed=seed + i)
            # Distance from (64, 20, 20) to (64, 20, 56) is 36.0 blocks
            l_ref = float(np.linalg.norm(np.array(wps[-1], dtype=float) - np.array(start, dtype=float)))
            courses.append((vol, start, wps, max(1.0, l_ref)))

    actual_episodes = min(num_episodes, len(courses))
    success_list = []
    spl_list = []
    waypoint_reached_list = []
    total_waypoint_list = []
    hazard_collision_list = []
    episode_lengths = []

    total_infer_time = 0.0
    total_steps = 0

    # 2. Setup Gating Wrapper if enabled
    gate: Optional[EventDrivenShellGate] = None
    if use_gating:
        if model_type in ("m2_static", "m1_full"):
            enc = backbone.context_encoder if hasattr(backbone, "context_encoder") else backbone
            gate = EventDrivenShellGate(backbone=enc, tau_gate=tau_gate)
        else:
            # Gating on other models defaults to dense if not multi-LOD
            gate = None

    # 3. Evaluation Loop
    for ep_idx in range(actual_episodes):
        vol, start, waypoints, l_ref = courses[ep_idx]
        env = VoxelNavEnv(volume=vol, start=start, waypoints=waypoints, max_steps=max_steps)
        obs = env.reset()

        if gate is not None:
            gate.reset()

        temporal_hidden = None
        done = False
        step_count = 0
        last_info: Dict[str, Any] = {}

        while not done and step_count < max_steps:
            step_count += 1
            total_steps += 1

            # Feature extraction with latency measurement
            t0 = time.perf_counter()
            if gate is not None:
                # Gated inference
                tokens = gate.step(obs["shells"])
                feat = jnp.mean(tokens, axis=1)  # (1, 256)
                if model_type == "m1_full" and temporal_model is not None:
                    if temporal_hidden is None:
                        temporal_hidden = temporal_model.init_hidden(1)
                    s_arr = jnp.asarray(obs["state"], dtype=jnp.float32).reshape((1, 12))
                    a_arr = jnp.zeros((1,), dtype=jnp.int32)
                    _, c_t, temporal_hidden = temporal_model.step(tokens, s_arr, a_arr, temporal_hidden)
                    feat = jnp.concatenate([feat, c_t], axis=-1)
            else:
                # Dense inference
                feat, temporal_hidden = extract_step_feature(
                    backbone=backbone,
                    model_type=model_type,
                    shells=obs["shells"],
                    state=obs["state"],
                    temporal_model=temporal_model,
                    temporal_hidden=temporal_hidden,
                )
            t_infer = time.perf_counter() - t0
            total_infer_time += t_infer

            # Policy forward pass
            goal_v = obs["goal_vec"]
            logits, _ = policy(feat, goal_v)

            if greedy:
                action = int(jnp.argmax(logits[0]))
            else:
                key, subkey = jax.random.split(key)
                action = int(jax.random.categorical(subkey, logits[0]))

            obs, reward, done, last_info = env.step(action)

        # Episode Metrics
        success = bool(last_info.get("success", False))
        path_len = float(last_info.get("path_length", 0.0))
        wp_reached = int(last_info.get("waypoints_reached", 0))
        tot_wp = int(last_info.get("total_waypoints", len(waypoints)))
        is_collision = bool(last_info.get("collision", False))
        # Hazard collision means catastrophic termination due to lava or fall >= 3
        is_hazard = is_collision and not success and done

        # SPL calculation: S_i * (l_i / max(p_i, l_i))
        if success:
            spl_i = float(l_ref / max(path_len, l_ref))
        else:
            spl_i = 0.0

        success_list.append(1.0 if success else 0.0)
        spl_list.append(spl_i)
        waypoint_reached_list.append(wp_reached)
        total_waypoint_list.append(tot_wp)
        hazard_collision_list.append(1.0 if is_hazard else 0.0)
        episode_lengths.append(step_count)

    # 4. Aggregate Metrics
    n = max(1, actual_episodes)
    sr_pct = (sum(success_list) / n) * 100.0
    spl_val = float(np.mean(spl_list))
    spl_pct = spl_val * 100.0
    tot_wp_count = max(1, sum(total_waypoint_list))
    wp_comp_pct = (sum(waypoint_reached_list) / tot_wp_count) * 100.0
    hazard_rate_pct = (sum(hazard_collision_list) / n) * 100.0

    # 5. Compute Empirical MFLOPs and Latency
    if gate is not None:
        stats = gate.get_stats()
        empirical_mflops = float(stats["mflops_per_step"])
    else:
        empirical_mflops = float(DENSE_MFLOPS_PER_STEP)

    latency_ms = (total_infer_time / max(1, total_steps)) * 1000.0

    results = {
        "success_rate_pct": float(sr_pct),
        "spl": float(spl_val),
        "spl_pct": float(spl_pct),
        "waypoint_completion_pct": float(wp_comp_pct),
        "hazard_collision_rate_pct": float(hazard_rate_pct),
        "empirical_mflops_per_step": float(empirical_mflops),
        "latency_ms_per_step": float(latency_ms),
        "use_gating": bool(use_gating),
        "total_episodes": int(actual_episodes),
        "total_steps": int(total_steps),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    return results


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entry Point
# ---------------------------------------------------------------------------

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for eval_navigation."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Navigation Course Evaluator & Latency Profiler"
    )
    parser.add_argument(
        "--policy_ckpt",
        type=str,
        default=None,
        help="Path to trained ActorCriticHead checkpoint (policy_best.pkl).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="m2_static",
        choices=["m1_full", "m2_static", "b1_lod0", "b2_lod2", "b3_mae"],
        help="Frozen perception backbone architecture.",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default=None,
        help="Path to trained perception backbone checkpoint (.pkl).",
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
        help="Path to directory containing custom course .npz files.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=250,
        help="Maximum steps allowed per episode.",
    )
    parser.add_argument(
        "--use_gating",
        action="store_true",
        help="Enable EventDrivenShellGate inference acceleration.",
    )
    parser.add_argument(
        "--tau_gate",
        type=float,
        default=0.05,
        help="Difference threshold for EventDrivenShellGate.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Path to save evaluation JSON summary.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke evaluation on debug bridge course.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """CLI entrypoint for eval_navigation."""
    opts = parse_args(args)
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

    if opts.backbone_ckpt and os.path.exists(opts.backbone_ckpt):
        print(f"Loading backbone from {opts.backbone_ckpt}...")
        load_checkpoint(opts.backbone_ckpt, backbone)

    if temporal_model and opts.temporal_ckpt and os.path.exists(opts.temporal_ckpt):
        print(f"Loading temporal model from {opts.temporal_ckpt}...")
        load_checkpoint(opts.temporal_ckpt, temporal_model)

    # 2. Instantiate Policy
    policy = ActorCriticHead(feat_dim=feat_dim, num_actions=12, rngs=rng)
    if opts.policy_ckpt and os.path.exists(opts.policy_ckpt):
        print(f"Loading policy checkpoint from {opts.policy_ckpt}...")
        load_checkpoint(opts.policy_ckpt, policy)

    # 3. Load Courses
    courses = None
    if opts.courses_dir and os.path.isdir(opts.courses_dir):
        courses = load_courses_from_dir(opts.courses_dir)

    num_episodes = 2 if opts.smoke_test else opts.num_episodes
    max_steps = 25 if opts.smoke_test else opts.max_steps

    # 4. Evaluate Navigation
    results = evaluate_navigation(
        backbone=backbone,
        policy=policy,
        model_type=opts.model_type,
        temporal_model=temporal_model,
        courses=courses,
        num_episodes=num_episodes,
        max_steps=max_steps,
        use_gating=opts.use_gating,
        tau_gate=opts.tau_gate,
        seed=opts.seed,
    )

    print("\n" + "=" * 60)
    print("Zero-Shot Navigation Evaluation Summary")
    print("=" * 60)
    print(f"Model Type:               {opts.model_type}")
    print(f"Use Gating:               {results['use_gating']}")
    print(f"Success Rate (SR %):      {results['success_rate_pct']:.2f}%")
    print(f"SPL:                      {results['spl']:.4f} ({results['spl_pct']:.2f}%)")
    print(f"Waypoint Completion %:    {results['waypoint_completion_pct']:.2f}%")
    print(f"Hazard Collision Rate %:  {results['hazard_collision_rate_pct']:.2f}%")
    print(f"Empirical MFLOPs/step:    {results['empirical_mflops_per_step']:.1f} MFLOPs")
    print(f"Latency (ms/step):        {results['latency_ms_per_step']:.2f} ms")
    print(f"Total Steps Evaluated:    {results['total_steps']}")
    print("=" * 60)

    # 5. Save JSON
    if opts.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(opts.out_json)), exist_ok=True)
        with open(opts.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation metrics to {opts.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
