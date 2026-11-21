"""
Unit & Integration Tests for MinecraftNavEnv & Inference Telemetry.
"""

from __future__ import annotations

import os
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env.minecraft_env import MinecraftNavEnv
from env.ipc_telemetry import InferenceTelemetry
from eval.train_ppo_nav import ActorCriticHead, extract_step_feature
from models.transformer import SpatialJEPA


@pytest.fixture
def test_regions_dir():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "data", "test_regions")
    if not os.path.exists(test_dir):
        pytest.skip(f"Test regions not found at {test_dir}")
    return test_dir


def test_minecraft_env_shapes_and_bounds(test_regions_dir):
    """Verify MinecraftNavEnv produces exact contract shapes and ranges."""
    env = MinecraftNavEnv(region_dir=test_regions_dir)
    obs = env.reset(seed=123)

    # 1. Shells: (3, 16, 16, 16, 16) float32 in [0.0, 1.0]
    shells = obs["shells"]
    assert isinstance(shells, np.ndarray)
    assert shells.shape == (3, 16, 16, 16, 16)
    assert shells.dtype == np.float32
    assert np.all(shells >= 0.0)
    assert np.all(shells <= 1.0)

    # 2. State: (12,) float32
    state = obs["state"]
    assert isinstance(state, np.ndarray)
    assert state.shape == (12,)
    assert state.dtype == np.float32

    # 3. Goal Vec: (3,) float32
    goal_vec = obs["goal_vec"]
    assert isinstance(goal_vec, np.ndarray)
    assert goal_vec.shape == (3,)
    assert goal_vec.dtype == np.float32

    # 4. Action Mask: (12,) bool
    action_mask = obs["action_mask"]
    assert isinstance(action_mask, np.ndarray)
    assert action_mask.shape == (12,)
    assert action_mask.dtype == bool
    assert np.any(action_mask), "At least one action must be valid"


def test_minecraft_env_step_all_actions(test_regions_dir):
    """Verify stepping through actions 0..11 produces valid rewards and states."""
    env = MinecraftNavEnv(region_dir=test_regions_dir)
    obs = env.reset(seed=42)

    for act in range(12):
        next_obs, reward, done, info = env.step(act)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "success" in info
        assert "target_distance" in info
        assert next_obs["shells"].shape == (3, 16, 16, 16, 16)
        if done:
            env.reset(seed=42 + act)


def test_inference_telemetry_pack():
    """Verify InferenceTelemetry packs TELM v1 header."""
    data = InferenceTelemetry.pack(
        target_pos=(100, 65, 200),
        world_delta=(10.0, 1.0, 5.0),
        target_dist=11.2,
        step_reward=-0.02,
        episode_return=5.4,
        critic_value=3.2,
        action_probabilities=[0.1] * 12,
        breadcrumbs=[(0, 0, 0), (1, 0, 0)],
        optimal_path=[(0, 0, 0), (0, 0, 1)],
        checkpoint_loss=0.045
    )
    assert len(data) <= 1024
    assert data[:4] == b"MLET"


def test_jax_policy_forward_with_minecraft_env(test_regions_dir):
    """Verify frozen SpatialJEPA + ActorCriticHead forward pass with real Minecraft observation."""
    rng = nnx.Rngs(42)
    backbone = SpatialJEPA(num_shells=3, rngs=rng)
    policy = ActorCriticHead(feat_dim=256, num_actions=12, rngs=rng)

    env = MinecraftNavEnv(region_dir=test_regions_dir)
    obs = env.reset(seed=42)

    feat, _ = extract_step_feature(
        backbone=backbone,
        model_type="m2_static",
        shells=obs["shells"],
        state=obs["state"],
    )
    assert feat.shape == (1, 256)

    logits, values = policy(feat, obs["goal_vec"])
    assert logits.shape == (1, 12)
    assert values.shape == (1,)
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.all(jnp.isfinite(values))


def test_minecraft_ipc_reader_interface():
    """Verify MinecraftIpcReader initializes and exposes expected API methods."""
    from env.mc_ipc_reader import MinecraftIpcReader, resolve_default_shm_path

    reader = MinecraftIpcReader(shm_path="test_nonexistent.bin")
    assert reader.shm_path == "test_nonexistent.bin"
    assert not reader.is_open()
    assert not reader.open()
    assert reader.read_header() is None
    assert reader.read_inference_telemetry() is None
    assert resolve_default_shm_path().endswith("squared_sense_mc.bin")


def test_record_episodes_e2e(tmp_path, test_regions_dir):
    """Verify record_episodes generates valid Option B .npz episodes loadable by dataset loader."""
    from env.record_episodes import record_episodes
    from data.dataset import load_episode_splits

    out_dir = str(tmp_path / "episodes")
    files = record_episodes(
        region_dir=test_regions_dir,
        out_dir=out_dir,
        num_episodes=3,
        ep_len=5,
        seed=101,
        policy="random",
    )
    assert len(files) == 3
    for f in files:
        assert os.path.isfile(f)

    # Verify loading via data/dataset.py:load_episode_splits
    splits = load_episode_splits(out_dir, seed=42)
    assert "train" in splits and "val" in splits and "test" in splits
    all_eps = splits["train"] + splits["val"] + splits["test"]
    assert len(all_eps) >= 3

    ep0 = all_eps[0]
    assert ep0["shells"].ndim == 6  # (T, 3, 16, 16, 16, 16)
    assert ep0["shells"].shape[1:] == (3, 16, 16, 16, 16)
    assert ep0["shells"].dtype == np.float32
    assert ep0["state"].shape[1] == 12
    assert ep0["action"].ndim == 1
    assert ep0["meta"].shape[1] == 6
