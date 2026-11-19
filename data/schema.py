"""Squared Sense Phase 1 schema definitions and constants.

Single source of truth for channel definitions, pooling configurations,
LOD geometry specifications, and tensor dimensions matching docs/CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import numpy as np

# Total number of channels and LOD levels
NUM_CHANNELS: int = 16
NUM_LODS: int = 3

# Dimensions
STATE_DIM: int = 12
META_DIM: int = 6
COURSE_SHAPE: tuple[int, int, int, int] = (16, 128, 64, 128)
SHELL_VOXEL_SHAPE: tuple[int, int, int, int] = (16, 16, 16, 16)


class Channel(IntEnum):
    """16-channel voxel schema as specified in docs/CONTRACT.md.
    
    Each voxel is uint8 [0..255] on disk and normalized to float32 [0.0..1.0] in model.
    """
    TERRAIN = 0             # Solid opaque walkable terrain [AVG pool]
    AIR = 1                 # Passable air / empty space [AVG pool]
    HAZARD = 2              # Hazard (lava, fire, magma) [MAX pool - salience]
    LIQUID = 3              # Liquid (water) [AVG pool]
    CLIMBABLE = 4           # Climbable / vertical step (stairs/ladders) [MAX pool - salience]
    FOLIAGE = 5             # Foliage / fragile / breakable [AVG pool]
    WORLD_Y = 6             # World Y Level (norm [-64..320] -> [0..255]) [AVG pool]
    COLLISION_HEIGHT = 7    # Collision height (0=air, 128=slab, 255=full/fence) [MAX pool - salience]
    WAYPOINT = 8            # Waypoint / goal beacon marker [MAX pool - salience]
    RESOURCE = 9            # Resource / interactive block [MAX pool - salience]
    WALKABLE_SURFACE = 10   # Walkable top surface (solid + >=2 air) [MAX pool - salience]
    LIGHT_LEVEL = 11        # Combined light level (max(sky, block)/15) [AVG pool]
    LETHAL_DROP = 12        # Lethal drop / cliff indicator (>=3 air below) [MAX pool - salience]
    SURFACE_SHELL = 13      # Surface shell flag (solid bordering air) [MAX pool - salience]
    BLOCK_HARDNESS = 14     # Normalized block hardness / diggability [AVG pool]
    OUT_OF_BOUNDS = 15      # Out-of-bounds / void padding mask [MAX pool]

    # Semantic aliases
    SOLID = 0
    WATER = 3
    GOAL = 8
    CLIFF = 12
    VOID = 15


# Pooling channel partition (disjoint and exhaustive across channels 0..15)
MAX_POOL_CHANNELS: tuple[int, ...] = (2, 4, 7, 8, 9, 10, 12, 13, 15)
AVG_POOL_CHANNELS: tuple[int, ...] = (0, 1, 3, 5, 6, 11, 14)


@dataclass(frozen=True)
class LODConfig:
    """Configuration for concentric multi-LOD shell geometry."""
    level: int
    scale: int
    span: int
    forward: int
    back: int
    lateral: int
    base_y_min: int
    base_y_max: int
    output_shape: tuple[int, int, int, int] = (16, 16, 16, 16)

    @property
    def y_min(self) -> int:
        """Alias for base_y_min."""
        return self.base_y_min

    @property
    def y_max(self) -> int:
        """Alias for base_y_max."""
        return self.base_y_max


LOD_SPECS: tuple[LODConfig, ...] = (
    LODConfig(
        level=0,
        scale=1,
        span=16,
        forward=12,
        back=-4,
        lateral=8,
        base_y_min=-4,
        base_y_max=12,
    ),
    LODConfig(
        level=1,
        scale=2,
        span=32,
        forward=24,
        back=-8,
        lateral=16,
        base_y_min=-8,
        base_y_max=24,
    ),
    LODConfig(
        level=2,
        scale=4,
        span=64,
        forward=48,
        back=-16,
        lateral=32,
        base_y_min=-16,
        base_y_max=48,
    ),
)


def normalize_shells(uint8_array: np.ndarray) -> np.ndarray:
    """Normalize uint8 voxel shells ([0..255]) to float32 ([0.0..1.0]).

    Args:
        uint8_array: Array containing uint8 voxel values in range [0, 255].

    Returns:
        float32 array scaled by 1.0 / 255.0 with values in range [0.0, 1.0].
    """
    return (np.asarray(uint8_array, dtype=np.float32) / 255.0).astype(np.float32)

