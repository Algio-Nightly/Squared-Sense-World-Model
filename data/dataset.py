"""Shared dataset loading and temporal clip extraction for Squared Sense.

Provides robust, episode-level dataset splitting, uint8 -> float32 normalization,
and online temporal clip generation strictly matching docs/CONTRACT.md.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from data.schema import Channel, normalize_shells


def load_episode_splits(
    data_dir: str,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Load episode files from directory and split strictly by episode file ID.

    Never frame-shuffles across splits. Resolves 'shells' key first before 'voxels',
    and ensures shells are float32 in [0.0, 1.0].

    Args:
        data_dir: Directory containing episode .npz files.
        seed: Random seed for deterministic episode splitting.
        ratios: (train, val, test) split ratios (default 70% / 15% / 15%).

    Returns:
        Dictionary with keys 'train', 'val', 'test', each mapping to a list of episode dicts:
          {
              'shells': (T, 3, 16, 16, 16, 16) float32 in [0, 1],
              'state':  (T, 12) float32,
              'action': (T,) int32,
              'meta':   (T, 6) float32,
              'file':   str filepath,
          }
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Episode directory '{data_dir}' does not exist.")

    files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    )
    if not files:
        raise FileNotFoundError(f"No .npz episode files found in '{data_dir}'.")

    rng = np.random.default_rng(seed)
    shuffled_files = list(rng.permutation(files))
    n = len(shuffled_files)

    # Episode-level splitting logic
    if n == 1:
        train_files = shuffled_files
        val_files = shuffled_files
        test_files = shuffled_files
    elif n == 2:
        train_files = [shuffled_files[0]]
        val_files = [shuffled_files[1]]
        test_files = [shuffled_files[1]]
    else:
        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        n_test = max(1, n - n_train - n_val)
        train_files = shuffled_files[:n_train]
        val_files = shuffled_files[n_train : n_train + n_val]
        test_files = shuffled_files[n_train + n_val :]
        if not test_files:
            test_files = val_files

    def _load_single_episode(fpath: str) -> Dict[str, np.ndarray]:
        with np.load(fpath) as archive:
            # 1. Resolve shells array
            if "shells" in archive:
                raw_shells = archive["shells"]
            elif "voxels" in archive:
                raw_shells = archive["voxels"]
            else:
                raise KeyError(
                    f"Episode file '{fpath}' missing both 'shells' and 'voxels' keys. Found: {archive.files}"
                )

            # Ensure uint8 [0..255] is normalized to float32 [0.0..1.0]
            if raw_shells.dtype == np.uint8 or raw_shells.max() > 1.5:
                shells = normalize_shells(raw_shells)
            else:
                shells = np.asarray(raw_shells, dtype=np.float32)

            T = shells.shape[0]

            # 2. State
            if "state" in archive:
                state = np.asarray(archive["state"], dtype=np.float32)
            else:
                state = np.zeros((T, 12), dtype=np.float32)

            # 3. Action
            if "action" in archive:
                action = np.asarray(archive["action"], dtype=np.int32)
            else:
                action = np.zeros((T,), dtype=np.int32)

            # 4. Meta
            if "meta" in archive:
                meta = np.asarray(archive["meta"], dtype=np.float32)
            else:
                meta = np.zeros((T, 6), dtype=np.float32)

        return {
            "shells": shells,
            "state": state,
            "action": action,
            "meta": meta,
            "file": fpath,
        }

    return {
        "train": [_load_single_episode(f) for f in train_files],
        "val": [_load_single_episode(f) for f in val_files],
        "test": [_load_single_episode(f) for f in test_files],
    }


def concat_episode_shells(episodes: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    """Concatenate shell arrays across multiple episodes along time/batch axis.

    Returns:
        (sum_T, 3, 16, 16, 16, 16) float32 array in [0.0, 1.0].
    """
    if not episodes:
        return np.zeros((0, 3, 16, 16, 16, 16), dtype=np.float32)
    return np.concatenate([ep["shells"] for ep in episodes], axis=0)


def extract_temporal_clips(
    episodes: Sequence[Dict[str, np.ndarray]],
    clip_len: int = 16,
    stride: int = 4,
) -> Dict[str, np.ndarray]:
    """Slice continuous 6D episode trajectories into 7D trajectory clips.

    Never crosses episode boundaries. Slices online with specified clip length and stride.
    Gracefully pads short episodes (T < clip_len) with the final frame.

    Args:
        episodes: Sequence of episode dictionaries containing 'shells', 'state', 'action'.
        clip_len: Length of output trajectory clips in steps (default 16).
        stride: Stride between successive clip window start indices (default 4).

    Returns:
        Dictionary with keys:
          'shells':  (N_clips, clip_len, 3, 16, 16, 16, 16) float32 in [0, 1],
          'states':  (N_clips, clip_len, 12) float32,
          'actions': (N_clips, clip_len) int32.
    """
    all_shells = []
    all_states = []
    all_actions = []

    for ep in episodes:
        shells = ep["shells"]
        state = ep["state"]
        action = ep["action"]
        T = len(shells)

        if T < clip_len:
            # Pad short episode by repeating last frame
            pad_len = clip_len - T
            last_shell = shells[-1:]
            last_state = state[-1:]
            last_act = action[-1:]

            sh_padded = np.concatenate([shells, np.repeat(last_shell, pad_len, axis=0)], axis=0)
            st_padded = np.concatenate([state, np.repeat(last_state, pad_len, axis=0)], axis=0)
            ac_padded = np.concatenate([action, np.repeat(last_act, pad_len, axis=0)], axis=0)

            all_shells.append(sh_padded)
            all_states.append(st_padded)
            all_actions.append(ac_padded)
        else:
            for start_idx in range(0, T - clip_len + 1, stride):
                end_idx = start_idx + clip_len
                all_shells.append(shells[start_idx:end_idx])
                all_states.append(state[start_idx:end_idx])
                all_actions.append(action[start_idx:end_idx])

    if not all_shells:
        return {
            "shells": np.zeros((0, clip_len, 3, 16, 16, 16, 16), dtype=np.float32),
            "states": np.zeros((0, clip_len, 12), dtype=np.float32),
            "actions": np.zeros((0, clip_len), dtype=np.int32),
        }

    return {
        "shells": np.stack(all_shells, axis=0),
        "states": np.stack(all_states, axis=0),
        "actions": np.stack(all_actions, axis=0),
    }


def compute_local_hazard_labels(shells: np.ndarray) -> np.ndarray:
    """Derive binary local hazard target from LOD-0 shell channel 2 (HAZARD).

    Checks whether any voxel in the 4-block region ahead of the agent has ch 2 >= 0.5.
    In LOD-0 (16, 16, 16, 16), agent is at (x=8, y=4, z=4).
    The 4-block forward region is z in [4..8], lateral x in [6..10], vertical y in [2..6].

    Args:
        shells: (N, 3, 16, 16, 16, 16) or (N, 16, 16, 16, 16) voxel shells.

    Returns:
        (N,) integer binary labels in {0, 1}.
    """
    lod0 = shells[:, 0] if shells.ndim == 6 else shells
    # Region ahead of agent: x in [6..10], y in [2..5], z in [4..8]
    ahead_region = lod0[:, 6:11, 2:6, 4:9, Channel.HAZARD]
    has_hazard = (ahead_region >= 0.5).any(axis=(1, 2, 3))
    return has_hazard.astype(int)


def compute_occlusion_labels(hazard: np.ndarray, lag: int = 5) -> np.ndarray:
    """Derive binary occlusion memory target.

    1 if a local hazard was present lag steps earlier (t - lag) and is no longer visible at t.

    Args:
        hazard: (N,) binary hazard labels for a continuous trajectory.
        lag: Step lookback window (default 5 steps).

    Returns:
        (N,) integer binary labels in {0, 1}.
    """
    T = len(hazard)
    occ = np.zeros(T, dtype=int)
    for t in range(lag, T):
        if hazard[t - lag] == 1 and hazard[t] == 0:
            occ[t] = 1
    return occ
