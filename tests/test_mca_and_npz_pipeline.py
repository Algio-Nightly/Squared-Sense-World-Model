"""Comprehensive Unit & Integration Test for MCA-to-Baked-Region Pipeline & Standardized NPZ Format.

Tests:
1. NPZ episode creation and ingestion via load_episode_splits.
2. NPZ course volume creation and ingestion via load_courses_from_dir.
3. Curriculum SpawnSampler loading from .npz.
4. LodExtractor slicing baked terrain into multi-LOD tensors (3, 16, 16, 16, 16) + metadata.
5. MinecraftNavEnv and record_episodes generating valid .npz episodes from baked region memmaps.
6. Execution of training scripts with --region_dir pointing directly to baked terrain.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import pytest

from data.dataset import load_episode_splits
from data.schema import COURSE_SHAPE, Channel
from env.lod_extractor import LodExtractor
from env.mca_baker import REGION_BLOCKS, WORLD_HEIGHT, WORLD_MIN_Y
from env.minecraft_env import MinecraftNavEnv
from env.palette import get_palette
from env.record_episodes import record_episodes
from env.spawn_indexer import SpawnSampler, index_region_spawns
from eval.eval_navigation import load_courses_from_dir
from eval.train_ppo_nav import main as ppo_main
from train.train_spatial_jepa import parse_args as parse_spatial_args, train_spatial_jepa


def create_synthetic_baked_region(output_dir: str, reg_x: int = 0, reg_z: int = 0):
    """Helper to create a synthetic baked region (.mmap, heightmap.npz, spawns.npz, json)."""
    os.makedirs(output_dir, exist_ok=True)
    mmap_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.mmap")
    h_npz_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.heightmap.npz")
    spawns_npz_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.spawns.npz")
    json_path = os.path.join(output_dir, f"r.{reg_x}.{reg_z}.json")
    manifest_path = os.path.join(output_dir, "world_manifest.json")

    shape = (REGION_BLOCKS, WORLD_HEIGHT, REGION_BLOCKS)
    mmap = np.memmap(mmap_path, dtype=np.uint16, mode="w+", shape=shape)

    # Fill ground from Y=0 to Y=127 with stone (ID 1), air above
    palette = get_palette()
    stone_id = palette.block_to_id.get("minecraft:stone", 1)
    grass_id = palette.block_to_id.get("minecraft:grass_block", 2)
    mmap[:, :127, :] = stone_id
    mmap[:, 127, :] = grass_id
    mmap[:, 128:, :] = 0
    mmap.flush()
    del mmap

    # Heightmap: highest solid block world Y = 127 + WORLD_MIN_Y = 127 - 64 = 63
    world_y_heightmap = np.full((REGION_BLOCKS, REGION_BLOCKS), 127 + WORLD_MIN_Y, dtype=np.int16)
    np.savez_compressed(h_npz_path, heightmap=world_y_heightmap)

    # Spawns: 2 blocks above grass (feet at 128 -> world Y = 128 - 64 = 64)
    min_wx = reg_x * REGION_BLOCKS
    min_wz = reg_z * REGION_BLOCKS
    surface_spawns = np.array([
        [min_wx + 10, 64, min_wz + 10],
        [min_wx + 20, 64, min_wz + 20],
        [min_wx + 30, 64, min_wz + 30],
    ], dtype=np.int32)
    subterranean_spawns = np.array([
        [min_wx + 15, 30, min_wz + 15],
    ], dtype=np.int32)
    steep_spawns = np.array([
        [min_wx + 25, 64, min_wz + 25],
    ], dtype=np.int32)

    np.savez_compressed(
        spawns_npz_path,
        surface=surface_spawns,
        subterranean=subterranean_spawns,
        steep=steep_spawns,
    )

    meta = {
        "region_x": reg_x,
        "region_z": reg_z,
        "min_world_x": min_wx,
        "max_world_x": min_wx + REGION_BLOCKS - 1,
        "min_world_z": min_wz,
        "max_world_z": min_wz + REGION_BLOCKS - 1,
        "min_world_y": WORLD_MIN_Y,
        "max_world_y": WORLD_MIN_Y + WORLD_HEIGHT - 1,
        "shape": list(shape),
        "dtype": "uint16",
        "mmap_file": os.path.basename(mmap_path),
        "heightmap_file": os.path.basename(h_npz_path),
        "spawns_file": os.path.basename(spawns_npz_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"regions": [meta]}, f, indent=4)

    return output_dir


def test_npz_episode_loading(tmp_path: Path):
    """Verify load_episode_splits loads standard .npz formatted episodes cleanly."""
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    T = 10

    for i in range(3):
        shells = rng.integers(0, 256, size=(T, 3, 16, 16, 16, 16), dtype=np.uint8)
        state = rng.normal(0, 1, size=(T, 12)).astype(np.float32)
        action = rng.integers(0, 12, size=(T,), dtype=np.int32)
        meta = rng.normal(0, 1, size=(T, 6)).astype(np.float32)

        np.savez_compressed(
            episodes_dir / f"ep_{i:02d}.npz",
            shells=shells,
            state=state,
            action=action,
            meta=meta,
        )

    splits = load_episode_splits(str(episodes_dir), ratios=(0.5, 0.25, 0.25), seed=42)

    all_eps = splits["train"] + splits["val"] + splits["test"]
    assert len(all_eps) == 3

    for ep in all_eps:
        assert ep["shells"].shape == (T, 3, 16, 16, 16, 16)
        assert ep["shells"].dtype == np.float32
        assert 0.0 <= ep["shells"].min() and ep["shells"].max() <= 1.0
        assert ep["state"].shape == (T, 12)
        assert ep["action"].shape == (T,)
        assert ep["meta"].shape == (T, 6)


def test_npz_courses_loading(tmp_path: Path):
    """Verify load_courses_from_dir loads .npz course files."""
    courses_dir = tmp_path / "courses"
    courses_dir.mkdir(parents=True)

    vol = np.zeros(COURSE_SHAPE, dtype=np.uint8)
    vol[Channel.AIR] = 255
    start = np.array([64, 20, 20], dtype=np.int32)
    waypoints = np.array([[64, 20, 40]], dtype=np.int32)

    for i in range(2):
        np.savez_compressed(
            courses_dir / f"course_{i:02d}.npz",
            volume=vol,
            start=start,
            waypoints=waypoints,
            shortest_path_len=20.0,
        )

    courses = load_courses_from_dir(str(courses_dir))
    assert len(courses) == 2
    for c in courses:
        c_vol, c_start, c_wps, c_len = c
        assert c_vol.shape == COURSE_SHAPE
        assert c_start == (64, 20, 20)
        assert c_wps == [(64, 20, 40)]
        assert c_len == 20.0


def test_spawn_sampler_npz(tmp_path: Path):
    """Verify SpawnSampler can be initialized directly from .npz file."""
    region_dir = create_synthetic_baked_region(str(tmp_path / "region"))
    npz_path = os.path.join(region_dir, "r.0.0.spawns.npz")

    sampler = SpawnSampler.from_npz(npz_path)
    coord = sampler.sample()

    assert len(coord) == 3


def test_lod_extractor_and_record_episodes_e2e(tmp_path: Path):
    """Test LodExtractor + MinecraftNavEnv + record_episodes in .npz format."""
    region_dir = create_synthetic_baked_region(str(tmp_path / "baked_region"))
    out_episodes_npz = str(tmp_path / "episodes_npz")

    # 1. Verify LodExtractor directly
    extractor = LodExtractor(region_dir=region_dir)
    shells, meta = extractor.extract_multi_lod(player_pos=(10.0, 64.0, 10.0), yaw=0.0)

    assert shells.shape == (3, 16, 16, 16, 16)
    assert shells.dtype == np.float32
    assert 0.0 <= shells.min() and shells.max() <= 1.0

    # 2. Record episodes in .npz format
    npz_files = record_episodes(
        region_dir=region_dir,
        out_dir=out_episodes_npz,
        num_episodes=2,
        ep_len=10,
        seed=42,
    )
    assert len(npz_files) == 2
    for f in npz_files:
        assert f.endswith(".npz")
        assert os.path.exists(f)

    # 3. Ingest via load_episode_splits
    splits_npz = load_episode_splits(out_episodes_npz)
    assert len(splits_npz["train"]) >= 1


def test_train_spatial_jepa_with_region_dir(tmp_path: Path):
    """Test that train_spatial_jepa takes --region_dir directly and auto-generates/loads multi-LOD episodes."""
    region_dir = create_synthetic_baked_region(str(tmp_path / "baked_for_training"))
    save_dir = str(tmp_path / "ckpt_spatial")

    args = parse_spatial_args([
        "--region_dir", region_dir,
        "--save_dir", save_dir,
        "--epochs", "1",
        "--batch_size", "4",
        "--warmup_epochs", "0",
        "--seed", "42",
    ])
    assert args.smoke_test is False
    assert args.region_dir == region_dir

    model, best_val_loss = train_spatial_jepa(args)
    assert np.isfinite(best_val_loss)
    assert os.path.isfile(os.path.join(save_dir, "best.pkl"))


def test_mca_baker_limit_sorting(tmp_path: Path):
    """Verify that bake_world with limit filters MCA files and prioritizes closest to origin (0, 0)."""
    from env.mca_baker import bake_world

    raw_dir = tmp_path / "raw_regions"
    out_dir = tmp_path / "baked_limited"
    raw_dir.mkdir()

    # Create dummy empty .mca files with coords (5, 5), (0, 0), (1, 0), (-1, -1)
    # Distances squared: (0,0)->0, (1,0)->1, (-1,-1)->2, (5,5)->50
    filenames = ["r.5.5.mca", "r.0.0.mca", "r.1.0.mca", "r.-1.-1.mca"]
    for fn in filenames:
        (raw_dir / fn).write_bytes(b"\x00" * 4096)  # Minimal 4KB header

    manifest = bake_world(str(raw_dir), str(out_dir), limit=2)
    # Manifest should only have 2 regions baked, specifically r.0.0 and r.1.0 (closest to 0,0)
    assert manifest["total_regions"] == 2
    assert "0,0" in manifest["regions"]
    assert "1,0" in manifest["regions"]
    assert "5,5" not in manifest["regions"]


