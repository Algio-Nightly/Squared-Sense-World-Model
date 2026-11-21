"""Unit tests for eval/sim_env.py 3D Voxel Navigation Simulator.

Strictly follows docs/EVAL_CONTRACT.md and docs/CONTRACT.md.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from eval.sim_env import CardinalSnapper, VoxelNavEnv, make_debug_bridge_course


def test_cardinal_snapper_hysteresis() -> None:
    """Verify CardinalSnapper snaps with ±15° rotational hysteresis deadband."""
    snapper = CardinalSnapper(initial_yaw=0.0, deadband=15.0)
    assert snapper.snap(0.0) == 0.0

    # Small departure < 60° retains current snapped cardinal (0°)
    assert snapper.snap(45.0) == 0.0
    assert snapper.snap(59.0) == 0.0

    # Crossing > 60° switches to 90°
    assert snapper.snap(61.0) == 90.0

    # Hysteresis prevents switching back to 0° until departure from 90° exceeds 60°
    assert snapper.snap(45.0) == 90.0
    assert snapper.snap(31.0) == 90.0
    assert snapper.snap(29.0) == 0.0


def test_reset_shapes_and_ranges() -> None:
    """Verify reset() returns exact channel-last (3, 16, 16, 16, 16) float32 in [0, 1],

    state (12,), and goal_vec (3,).
    """
    volume, start, waypoints = make_debug_bridge_course()
    env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)
    obs = env.reset()

    # 1. Shells shape and range
    shells = obs["shells"]
    assert isinstance(shells, np.ndarray)
    assert shells.shape == (3, 16, 16, 16, 16)
    assert shells.dtype == np.float32
    assert np.all(shells >= 0.0)
    assert np.all(shells <= 1.0)

    # 2. State shape and range
    state = obs["state"]
    assert isinstance(state, np.ndarray)
    assert state.shape == (12,)
    assert state.dtype == np.float32
    # Verify health, food, grounded are reasonable
    assert state[9] == 1.0  # health normalized to 1.0
    assert state[10] == 1.0  # food normalized to 1.0
    assert state[11] == 1.0  # standing on ground

    # 3. Goal vector
    goal_vec = obs["goal_vec"]
    assert isinstance(goal_vec, np.ndarray)
    assert goal_vec.shape == (3,)
    assert goal_vec.dtype == np.float32
    # Distance to waypoint 36 blocks ahead along +Z
    expected_goal_vec = np.array([0.0, 0.0, 36.0], dtype=np.float32)
    np.testing.assert_allclose(goal_vec, expected_goal_vec, atol=1e-5)


def test_walk_straight_debug_bridge() -> None:
    """Verify walking straight across the debug bridge to the waypoint yields

    info['success'] == True and positive total reward.
    """
    volume, start, waypoints = make_debug_bridge_course()
    env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)
    env.reset()

    total_reward = 0.0
    done = False
    info = {}

    # Bridge waypoint is 36 blocks ahead along +Z. Action 0: Forward
    for step_i in range(40):
        obs, reward, done, info = env.step(action=0)
        total_reward += reward
        if done:
            break

    assert done is True
    assert info["success"] is True
    assert info["waypoints_reached"] == 1
    assert total_reward > 0.0, f"Expected positive total reward, got {total_reward}"
    assert info["collision"] is False


def test_step_into_lava() -> None:
    """Verify stepping off the bridge into lava terminates immediately with

    reward <= -5.0 and info['collision'] == True.
    """
    volume, start, waypoints = make_debug_bridge_course()
    env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)
    env.reset()

    # Step off bridge to the right: Action 4 (Strafe-Right)
    obs, reward, done, info = env.step(action=4)

    assert done is True
    assert reward <= -5.0, f"Expected reward <= -5.0, got {reward}"
    assert info["collision"] is True
    assert info["success"] is False


def test_walk_into_wall() -> None:
    """Verify walking into a solid wall gives -0.1 wall-bump penalty without moving

    through the wall.
    """
    volume, start, waypoints = make_debug_bridge_course()
    env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)
    env.reset()

    initial_pos = np.copy(env.pos)

    # In make_debug_bridge_course, solid wall is placed behind start at z=19.
    # Action 5: Backward moves towards z=19
    obs, reward, done, info = env.step(action=5)

    # Position must NOT change
    np.testing.assert_array_equal(env.pos, initial_pos)
    # Wall-bump gives -0.1 penalty (plus per-step -0.02 time penalty)
    assert reward <= -0.1
    assert info["collision"] is True
    assert done is False


def test_speed_200_steps() -> None:
    """Verify 200 steps execute in < 1.0s on CPU."""
    volume, start, waypoints = make_debug_bridge_course()
    # Test with salience pooling (default)
    env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints, max_steps=500)
    env.reset()

    t0 = time.perf_counter()
    for _ in range(200):
        # Action 11: Stay / Idle to step without terminating
        obs, reward, done, info = env.step(action=11)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"200 steps took {elapsed:.4f}s, exceeding 1.0s limit"
