"""End-to-End Integration Test for Real-Format Episode and Course Data.

Verifies that all trainers and evaluators successfully ingest real-format disk data:
  - 4 synthetic uint8 .npz episodes (T=24, keys: 'action', 'meta', 'shells', 'state' in alphabetical order)
  - 2 .npz course volumes (16, 128, 64, 128) uint8
  - Real-mode execution (epochs=1, NO --smoke_test, NO --demo) for:
      1. train_spatial_jepa.py
      2. train_temporal_jepa.py
      3. train_baselines.py (b1_lod0)
      4. probe_suite.py (m2_static)
      5. train_ppo_nav.py (--courses_dir)
  - Validates zero crashes, finite losses/RankMe, and proper checkpoint/JSON outputs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np
import pytest

from data.schema import COURSE_SHAPE, Channel
from eval.probe_suite import run_probe_suite
from eval.train_ppo_nav import main as ppo_main
from train.train_baselines import train_baseline
from train.train_spatial_jepa import train_spatial_jepa
from train.train_temporal_jepa import train_temporal_jepa


@pytest.fixture
def real_data_env(tmp_path: Path):
    """Create a temporary directory with 4 real-format uint8 episodes and 2 course files."""
    data_dir = tmp_path / "episodes"
    courses_dir = tmp_path / "courses"
    save_dir = tmp_path / "checkpoints"
    data_dir.mkdir(parents=True)
    courses_dir.mkdir(parents=True)
    save_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)

    # 1. Create 4 uint8 episodes (T=24)
    # Save keys in alphabetical order: 'action', 'meta', 'shells', 'state'
    for i in range(4):
        T = 24
        shells = rng.integers(0, 256, size=(T, 3, 16, 16, 16, 16), dtype=np.uint8)
        # Ensure some hazards exist ahead for local hazard probing
        shells[:, 0, 7:10, 3:5, 5:7, Channel.HAZARD] = rng.choice(
            [0, 255], size=(T, 3, 2, 2), p=[0.5, 0.5]
        ).astype(np.uint8)

        state = np.zeros((T, 12), dtype=np.float32)
        state[:, 0:3] = rng.normal(0, 0.5, size=(T, 3))
        state[:, 3:6] = state[:, 0:3] * 0.2
        state[:, 6] = 0.0
        state[:, 7] = 1.0
        state[:, 8] = 0.0
        state[:, 9] = 1.0  # normalized health
        state[:, 10] = 1.0  # normalized food
        state[:, 11] = 1.0

        action = rng.integers(0, 12, size=(T,), dtype=np.int32)

        meta = np.zeros((T, 6), dtype=np.float32)
        meta[:, 0:3] = rng.uniform(-10, 10, size=(T, 3))
        meta[:, 3:6] = rng.uniform(30, 48, size=(T, 3))

        ep_path = data_dir / f"episode_{i:03d}.npz"
        # Save explicitly with keys
        np.savez_compressed(
            ep_path,
            action=action,
            meta=meta,
            shells=shells,
            state=state,
        )

    # 2. Create 2 course files (16, 128, 64, 128)
    for c_idx in range(2):
        vol = np.zeros(COURSE_SHAPE, dtype=np.uint8)
        vol[Channel.AIR] = 255
        vol[Channel.TERRAIN, 60:68, 19, 20:60] = 255
        vol[Channel.COLLISION_HEIGHT, 60:68, 19, 20:60] = 255
        start = np.array([64, 20, 20], dtype=np.int32)
        waypoints = np.array([[64, 20, 35], [64, 20, 50]], dtype=np.int32)

        c_path = courses_dir / f"course_{c_idx:02d}.npz"
        np.savez_compressed(
            c_path,
            volume=vol,
            start=start,
            waypoints=waypoints,
            shortest_path_len=30.0,
        )

    return {
        "data_dir": str(data_dir),
        "courses_dir": str(courses_dir),
        "save_dir": str(save_dir),
    }


def test_real_data_train_spatial_jepa(real_data_env):
    """Test 1-epoch real-mode execution of train_spatial_jepa.py."""
    data_dir = real_data_env["data_dir"]
    save_dir = os.path.join(real_data_env["save_dir"], "spatial")

    from train.train_spatial_jepa import parse_args

    args = parse_args(
        [
            "--data_dir", data_dir,
            "--save_dir", save_dir,
            "--epochs", "1",
            "--batch_size", "8",
            "--warmup_epochs", "0",
            "--seed", "42",
        ]
    )
    # Ensure smoke_test is False
    assert args.smoke_test is False

    model, best_val_loss = train_spatial_jepa(args)

    assert np.isfinite(best_val_loss)
    assert os.path.isfile(os.path.join(save_dir, "best.pkl"))
    assert os.path.isfile(os.path.join(save_dir, "last.pkl"))
    assert os.path.isfile(os.path.join(save_dir, "metrics.jsonl"))

    with open(os.path.join(save_dir, "metrics.jsonl"), "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    assert "val_loss" in lines[0]
    assert np.isfinite(lines[0]["val_loss"])


def test_real_data_train_temporal_jepa(real_data_env):
    """Test 1-epoch real-mode execution of train_temporal_jepa.py."""
    data_dir = real_data_env["data_dir"]
    save_dir = os.path.join(real_data_env["save_dir"], "temporal")

    from train.train_temporal_jepa import parse_args

    args = parse_args(
        [
            "--data_dir", data_dir,
            "--save_dir", save_dir,
            "--phase1_epochs", "1",
            "--phase2_epochs", "1",
            "--batch_size", "4",
            "--clip_len", "16",
            "--k_steps", "2",
            "--seed", "42",
        ]
    )
    assert args.smoke_test is False

    temp_model, spat_model = train_temporal_jepa(args)

    assert os.path.isfile(os.path.join(save_dir, "temporal_best.pkl"))
    assert os.path.isfile(os.path.join(save_dir, "metrics.jsonl"))

    with open(os.path.join(save_dir, "metrics.jsonl"), "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2  # 1 epoch Phase 1 + 1 epoch Phase 2
    assert "rollout_l1" in lines[0]
    assert np.isfinite(lines[0]["val_loss"])


def test_real_data_train_baselines(real_data_env):
    """Test 1-epoch real-mode execution of train_baselines.py (b1_lod0)."""
    data_dir = real_data_env["data_dir"]
    save_dir = os.path.join(real_data_env["save_dir"], "baselines_b1")

    from train.train_baselines import parse_args

    args = parse_args(
        [
            "--model", "b1_lod0",
            "--data_dir", data_dir,
            "--save_dir", save_dir,
            "--epochs", "1",
            "--batch_size", "8",
            "--warmup_epochs", "0",
            "--seed", "42",
        ]
    )
    assert args.smoke_test is False

    model, best_val_loss = train_baseline(args)

    assert np.isfinite(best_val_loss)
    assert os.path.isfile(os.path.join(save_dir, "best.pkl"))
    assert os.path.isfile(os.path.join(save_dir, "metrics.jsonl"))


def test_real_data_probe_suite(real_data_env):
    """Test real-mode execution of probe_suite.py on real episode files."""
    data_dir = real_data_env["data_dir"]
    out_json = os.path.join(real_data_env["save_dir"], "probe_summary.json")

    summary = run_probe_suite(
        model_type="m2_static",
        data_dir=data_dir,
        out_json=out_json,
        smoke_test=False,
    )

    assert os.path.isfile(out_json)
    assert "distant_goal" in summary
    assert "r2" in summary["distant_goal"]
    assert np.isfinite(summary["distant_goal"]["r2"])
    assert "local_hazard" in summary
    assert "f1" in summary["local_hazard"]
    assert "kinematic_and_occlusion" in summary
    assert "multistep_rollout" in summary
    assert "rankme" in summary["multistep_rollout"]
    assert np.isfinite(summary["multistep_rollout"]["rankme"])


def test_real_data_train_ppo_courses(real_data_env):
    """Test train_ppo_nav.py loading courses from --courses_dir."""
    courses_dir = real_data_env["courses_dir"]
    save_dir = os.path.join(real_data_env["save_dir"], "ppo")

    # Run 1 update with 16 rollout steps
    ret = ppo_main(
        [
            "--courses_dir", courses_dir,
            "--save_dir", save_dir,
            "--total_steps", "32",
            "--seed", "42",
        ]
    )
    assert ret == 0
    assert os.path.isfile(os.path.join(save_dir, "policy_best.pkl"))
    assert os.path.isfile(os.path.join(save_dir, "ppo_metrics.jsonl"))
