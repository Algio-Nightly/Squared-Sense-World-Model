"""Tests for data/schema.py single source of truth."""

import numpy as np
import pytest

from data.schema import (
    AVG_POOL_CHANNELS,
    COURSE_SHAPE,
    LOD_SPECS,
    MAX_POOL_CHANNELS,
    META_DIM,
    NUM_CHANNELS,
    NUM_LODS,
    STATE_DIM,
    Channel,
    LODConfig,
    normalize_shells,
)


def test_channel_enum() -> None:
    """Verify Channel enum defines all 16 channels from 0 to 15."""
    values = {c.value for c in Channel}
    assert values == set(range(NUM_CHANNELS))
    assert len(values) == 16

    # Verify specific key channels from docs/CONTRACT.md
    assert Channel.TERRAIN == 0
    assert Channel.AIR == 1
    assert Channel.HAZARD == 2
    assert Channel.LIQUID == 3
    assert Channel.CLIMBABLE == 4
    assert Channel.FOLIAGE == 5
    assert Channel.WORLD_Y == 6
    assert Channel.COLLISION_HEIGHT == 7
    assert Channel.WAYPOINT == 8
    assert Channel.RESOURCE == 9
    assert Channel.WALKABLE_SURFACE == 10
    assert Channel.LIGHT_LEVEL == 11
    assert Channel.LETHAL_DROP == 12
    assert Channel.SURFACE_SHELL == 13
    assert Channel.BLOCK_HARDNESS == 14
    assert Channel.OUT_OF_BOUNDS == 15


def test_pooling_channels_disjoint_and_exhaustive() -> None:
    """Verify MAX_POOL_CHANNELS and AVG_POOL_CHANNELS are disjoint and cover all 16 channels."""
    assert isinstance(MAX_POOL_CHANNELS, tuple)
    assert isinstance(AVG_POOL_CHANNELS, tuple)

    max_set = set(MAX_POOL_CHANNELS)
    avg_set = set(AVG_POOL_CHANNELS)

    # Disjoint check
    assert max_set.isdisjoint(avg_set), f"Overlap found: {max_set & avg_set}"

    # Coverage check across all 16 channels
    all_channels = max_set | avg_set
    assert all_channels == set(range(16)), f"Missing or extra channels: {all_channels ^ set(range(16))}"
    assert len(MAX_POOL_CHANNELS) + len(AVG_POOL_CHANNELS) == 16


def test_lod_specs_spans_and_scales() -> None:
    """Verify LOD configs: all LOD spans equal 16 * scale and match geometric contracts."""
    assert isinstance(LOD_SPECS, tuple)
    assert len(LOD_SPECS) == NUM_LODS == 3

    expected_scales = (1, 2, 4)
    for idx, (lod, expected_scale) in enumerate(zip(LOD_SPECS, expected_scales)):
        assert isinstance(lod, LODConfig)
        assert lod.level == idx
        assert lod.scale == expected_scale
        # Explicit contract requirement: all LOD spans equal 16 * scale
        assert lod.span == 16 * lod.scale

        # Geometric span consistency checks
        assert (lod.forward - lod.back) == 16 * lod.scale
        assert (lod.lateral * 2) == 16 * lod.scale
        assert (lod.base_y_max - lod.base_y_min) == 16 * lod.scale
        assert (lod.y_max - lod.y_min) == 16 * lod.scale

        # Output voxel grid shape is invariant across LODs
        assert lod.output_shape == (16, 16, 16, 16)


def test_normalize_shells() -> None:
    """Verify normalize_shells maps 0 -> 0.0 and 255 -> 1.0 with float32 output."""
    raw = np.array([0, 51, 255], dtype=np.uint8)
    norm = normalize_shells(raw)

    assert norm.dtype == np.float32
    assert norm[0] == pytest.approx(0.0)
    assert norm[1] == pytest.approx(0.2)
    assert norm[2] == pytest.approx(1.0)

    # Multi-dimensional tensor test
    shape = (3, 16, 16, 16, 16)
    zeros = np.zeros(shape, dtype=np.uint8)
    ones = np.full(shape, 255, dtype=np.uint8)

    norm_zeros = normalize_shells(zeros)
    norm_ones = normalize_shells(ones)

    assert norm_zeros.shape == shape
    assert norm_zeros.dtype == np.float32
    assert np.all(norm_zeros == 0.0)

    assert norm_ones.shape == shape
    assert norm_ones.dtype == np.float32
    assert np.all(norm_ones == 1.0)


def test_constants() -> None:
    """Verify state, meta, and course shape constants."""
    assert STATE_DIM == 12
    assert META_DIM == 6
    assert COURSE_SHAPE == (16, 128, 64, 128)

