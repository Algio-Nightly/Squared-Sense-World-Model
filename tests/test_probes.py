"""Unit tests for eval/probe_suite.py representation and dynamics probe suite.

Strictly follows docs/EVAL_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import json
import math
import numpy as np
import pytest
import flax.nnx as nnx

from eval.probe_suite import (
    eval_multistep_rollout,
    extract_features,
    generate_synthetic_probe_data,
    probe_distant_goal,
    probe_kinematic_and_occlusion,
    probe_local_hazard,
    run_probe_suite,
)
from models.baselines import Generative3DMAE, SingleLODJEPA
from models.temporal_lstm import DualLSTMTemporalIntegrator
from models.transformer import SpatialJEPA


def test_extract_features_shapes() -> None:
    """Verify extract_features produces (N, 256) for static models and (N, 448) for m1_full."""
    rng = nnx.Rngs(0)
    n_samples = 4
    shells = np.random.uniform(0.0, 1.0, size=(n_samples, 3, 16, 16, 16, 16)).astype(np.float32)
    state = np.zeros((n_samples, 12), dtype=np.float32)
    action = np.zeros((n_samples,), dtype=np.int32)

    # 1. m2_static: (N, 256)
    m2 = SpatialJEPA(num_shells=3, rngs=rng)
    feat_m2 = extract_features(m2, "m2_static", shells, batch_size=2)
    assert feat_m2.shape == (n_samples, 256)
    assert feat_m2.dtype == np.float32
    assert np.all(np.isfinite(feat_m2))

    # 2. b1_lod0: (N, 256)
    b1 = SingleLODJEPA(lod_index=0, rngs=rng)
    feat_b1 = extract_features(b1, "b1_lod0", shells, batch_size=2)
    assert feat_b1.shape == (n_samples, 256)
    assert feat_b1.dtype == np.float32
    assert np.all(np.isfinite(feat_b1))

    # 3. b2_lod2: (N, 256)
    b2 = SingleLODJEPA(lod_index=2, rngs=rng)
    feat_b2 = extract_features(b2, "b2_lod2", shells, batch_size=2)
    assert feat_b2.shape == (n_samples, 256)
    assert feat_b2.dtype == np.float32
    assert np.all(np.isfinite(feat_b2))

    # 4. b3_mae: (N, 256)
    b3 = Generative3DMAE(rngs=rng)
    feat_b3 = extract_features(b3, "b3_mae", shells, batch_size=2)
    assert feat_b3.shape == (n_samples, 256)
    assert feat_b3.dtype == np.float32
    assert np.all(np.isfinite(feat_b3))

    # 5. m1_full: (N, 448) concatenating z_t (256) and c_t (192)
    temporal_model = DualLSTMTemporalIntegrator(rngs=rng)
    feat_m1 = extract_features(
        m2, "m1_full", shells, state=state, action=action, temporal_model=temporal_model, batch_size=2
    )
    assert feat_m1.shape == (n_samples, 448)
    assert feat_m1.dtype == np.float32
    assert np.all(np.isfinite(feat_m1))


def test_individual_probes() -> None:
    """Verify individual probe functions return expected metric keys with finite values."""
    n_train, n_test, d = 48, 16, 64
    rng = np.random.default_rng(123)

    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    X_test = rng.normal(size=(n_test, d)).astype(np.float32)

    # 1. Distant goal regression
    y_train_goal = rng.normal(size=(n_train, 3)).astype(np.float32)
    y_test_goal = rng.normal(size=(n_test, 3)).astype(np.float32)
    res_goal = probe_distant_goal(X_train, y_train_goal, X_test, y_test_goal)
    for key in ("r2", "r2_std", "mean_cosine_sim", "mean_cosine_sim_std"):
        assert key in res_goal
        assert math.isfinite(res_goal[key])

    # 2. Local hazard classification
    y_train_haz = rng.choice([0, 1], size=n_train).astype(int)
    y_test_haz = rng.choice([0, 1], size=n_test).astype(int)
    res_haz = probe_local_hazard(X_train, y_train_haz, X_test, y_test_haz)
    for key in ("f1", "f1_std", "auroc", "auroc_std"):
        assert key in res_haz
        assert math.isfinite(res_haz[key])

    # Single-class edge case handling
    res_single_class = probe_local_hazard(
        X_train, np.zeros(n_train, dtype=int), X_test, np.zeros(n_test, dtype=int)
    )
    assert math.isfinite(res_single_class["f1"])
    assert math.isfinite(res_single_class["auroc"])

    # 3. Kinematic and occlusion probe
    v_train = rng.normal(size=(n_train, 3)).astype(np.float32)
    v_test = rng.normal(size=(n_test, 3)).astype(np.float32)
    occ_train = rng.choice([0, 1], size=n_train).astype(int)
    occ_test = rng.choice([0, 1], size=n_test).astype(int)
    res_kin_occ = probe_kinematic_and_occlusion(
        X_train, v_train, occ_train, X_test, v_test, occ_test
    )
    for key in ("velocity_r2", "velocity_r2_std", "occlusion_f1", "occlusion_f1_std"):
        assert key in res_kin_occ
        assert math.isfinite(res_kin_occ[key])

    # 4. Multi-step rollout
    b, T = 2, 16
    tok_seq = rng.normal(size=(b, T, 192, 256)).astype(np.float32)
    st_seq = rng.normal(size=(b, T, 12)).astype(np.float32)
    act_seq = rng.integers(0, 12, size=(b, T)).astype(np.int32)
    ema_seq = rng.normal(size=(b, T, 256)).astype(np.float32)
    res_rollout = eval_multistep_rollout(
        temporal_model=None,
        tokens_seq=tok_seq,
        state_seq=st_seq,
        action_seq=act_seq,
        ema_targets_seq=ema_seq,
        horizons=(1, 2, 4, 8),
    )
    for h in (1, 2, 4, 8):
        assert f"cosine_sim_h{h}" in res_rollout
        assert math.isfinite(res_rollout[f"cosine_sim_h{h}"])
        assert f"l1_error_h{h}" in res_rollout
        assert math.isfinite(res_rollout[f"l1_error_h{h}"])
    assert "rankme" in res_rollout
    assert math.isfinite(res_rollout["rankme"])


@pytest.mark.parametrize("model_type", ["m2_static", "m1_full", "b1_lod0", "b3_mae"])
def test_smoke_test_all_models(tmp_path, model_type: str) -> None:
    """Run --smoke_test for m2_static, m1_full, b1_lod0, and b3_mae into tmp_path

    and assert the output JSON contains all required metric keys with finite values.
    """
    out_json = tmp_path / f"probe_results_{model_type}.json"
    summary = run_probe_suite(
        model_type=model_type,
        out_json=str(out_json),
        smoke_test=True,
    )

    # Verify file was written
    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required metric keys
    required_keys = [
        "model_type",
        "feature_dim",
        "r2",
        "mean_cosine_sim",
        "f1",
        "auroc",
        "velocity_r2",
        "occlusion_f1",
        "rankme",
        "cosine_sim_h1",
        "cosine_sim_h2",
        "cosine_sim_h4",
        "cosine_sim_h8",
        "l1_error_h1",
        "l1_error_h2",
        "l1_error_h4",
        "l1_error_h8",
    ]

    for key in required_keys:
        assert key in data, f"Missing required key '{key}' in {out_json}"
        if key not in ("model_type", "feature_dim"):
            val = data[key]
            assert isinstance(val, (float, int)), f"Key '{key}' value {val} is not numeric"
            assert math.isfinite(val), f"Key '{key}' value {val} is not finite"

    # Verify feature dimension matches expected architecture
    if model_type == "m1_full":
        assert data["feature_dim"] == 448
    else:
        assert data["feature_dim"] == 256
