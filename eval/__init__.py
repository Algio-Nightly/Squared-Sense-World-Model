"""Evaluation and simulation suite for Squared Sense World Model.

Modules:
  - probe_suite: Frozen representation probes and dynamics evaluation.
  - sim_env: Deterministic 3D voxel navigation simulator and test environments.
"""

from eval.eval_navigation import evaluate_navigation
from eval.probe_suite import (
    eval_multistep_rollout,
    extract_features,
    probe_distant_goal,
    probe_kinematic_and_occlusion,
    probe_local_hazard,
)
from eval.sim_env import (
    CardinalSnapper,
    VoxelNavEnv,
    extract_shells,
    make_debug_bridge_course,
)
from eval.train_ppo_nav import (
    ActorCriticHead,
    train_bc_warmstart,
    train_ppo,
)

__all__ = [
    "extract_features",
    "probe_distant_goal",
    "probe_local_hazard",
    "probe_kinematic_and_occlusion",
    "eval_multistep_rollout",
    "VoxelNavEnv",
    "make_debug_bridge_course",
    "CardinalSnapper",
    "extract_shells",
    "ActorCriticHead",
    "train_bc_warmstart",
    "train_ppo",
    "evaluate_navigation",
]
