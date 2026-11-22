"""
Deterministic Movement Physics Kernel for Minecraft 3D Pathfinding.
Enforces 1x2x1 player hitbox, 1-block step-up, gravity falling, and fatal hazard/fall detection.
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np

from .mca_baker import WORLD_MIN_Y
from .palette import BlockPalette, get_palette


class MoveOutcome(Enum):
    SUCCESS = 0         # Standard flat move (same Y)
    STEP_UP = 1         # Stepped up 1 block onto solid obstacle
    FALL = 2            # Stepped off ledge and landed safely (fall distance <= max_safe_fall)
    BLOCKED = 3         # Move blocked by wall, low ceiling, or lack of step-up clearance
    FATAL_FALL = 4      # Fell > max_safe_fall blocks
    FATAL_HAZARD = 5    # Landed on or walked into lava/fire/hazard
    FATAL_VOID = 6      # Fell into void below world minimum Y


class VoxelCourseAdapter:
    """Adapts a 16-channel uint8 (16, X, Y, Z) course volume into a block query interface."""

    def __init__(self, volume: np.ndarray, palette_obj: Optional[BlockPalette] = None):
        self.volume = volume
        self.palette = palette_obj or get_palette()
        self.c_dim, self.x_dim, self.y_dim, self.z_dim = volume.shape
        self.air_id = self.palette.block_to_id.get("minecraft:air", 0)
        self.stone_id = self.palette.block_to_id.get("minecraft:stone", 2)
        self.lava_id = self.palette.block_to_id.get("minecraft:lava", 478)

    def get_block(self, x: int, y: int, z: int) -> int:
        if x < 0 or x >= self.x_dim or y < 0 or y >= self.y_dim or z < 0 or z >= self.z_dim:
            return self.stone_id  # Bounds barrier
        # Channel 2: HAZARD
        if self.volume[2, x, y, z] >= 128:
            return self.lava_id
        # Channel 0: TERRAIN, Channel 7: COLLISION_HEIGHT
        if self.volume[0, x, y, z] >= 128 or self.volume[7, x, y, z] >= 128:
            return self.stone_id
        return self.air_id

    def is_solid(self, x: int, y: int, z: int) -> bool:
        b = self.get_block(x, y, z)
        return bool(self.palette.lut_is_solid[b])

    def is_hazard(self, x: int, y: int, z: int) -> bool:
        b = self.get_block(x, y, z)
        return bool(self.palette.lut_hazard[b] > 0.0)


class MovementPhysicsKernel:
    """
    Simulates discrete 1x2x1 player locomotion in Minecraft voxel worlds.
    Supports both WorldVolume memmaps and 16-channel course volumes (via VoxelCourseAdapter).
    """

    def __init__(self, world_volume, palette_obj: Optional[BlockPalette] = None, max_safe_fall: int = 5):
        self.palette = palette_obj or get_palette()
        if isinstance(world_volume, np.ndarray) and world_volume.ndim == 4:
            self.world = VoxelCourseAdapter(world_volume, self.palette)
        else:
            self.world = world_volume
        self.max_safe_fall = max_safe_fall

    def is_passable_air(self, block_id: int) -> bool:
        """Returns True if block can be occupied by player body (air, grass, flowers)."""
        return self.palette.lut_passability[block_id] >= 0.5 and self.palette.lut_hazard[block_id] == 0.0

    def is_solid_floor(self, block_id: int) -> bool:
        """Returns True if block can support player feet without hazard."""
        return self.palette.lut_is_solid[block_id] and self.palette.lut_hazard[block_id] == 0.0

    def is_hazard(self, block_id: int) -> bool:
        """Returns True if block is dangerous (lava, fire, magma, cactus)."""
        return self.palette.lut_hazard[block_id] > 0.0

    def is_valid_standing_position(self, x: int, y: int, z: int) -> bool:
        """
        Validates 1x2x1 player hitbox clearance at feet position (x, y, z):
        - Floor at y-1 must be solid and non-hazard.
        - Feet at y must be passable and non-hazard.
        - Head at y+1 must be passable and non-hazard.
        """
        floor_b = self.world.get_block(x, y - 1, z)
        if not self.is_solid_floor(floor_b):
            return False

        feet_b = self.world.get_block(x, y, z)
        if not self.is_passable_air(feet_b):
            return False

        head_b = self.world.get_block(x, y + 1, z)
        if not self.is_passable_air(head_b):
            return False

        return True

    def step(
        self,
        x: int,
        y: int,
        z: int,
        dx: int,
        dz: int
    ) -> Tuple[int, int, int, MoveOutcome, int]:
        """
        Simulates moving 1 block horizontally from (x, y, z) along (dx, dz).
        Returns: (next_x, next_y, next_z, outcome, fall_distance).
        """
        nx, nz = x + dx, z + dz

        # Query relevant blocks
        target_feet_b = self.world.get_block(nx, y, nz)

        # -------------------------------------------------------------
        # 1. Step-UP check (if obstacle in front is solid)
        # -------------------------------------------------------------
        if self.palette.lut_is_solid[target_feet_b]:
            # Can step up 1 block if:
            # - Current position has headroom to jump: (x, y+2, z) is passable
            # - Target position has headroom for feet: (nx, y+1, nz) is passable
            # - Target position has headroom for head: (nx, y+2, nz) is passable
            # - Target floor (nx, y, nz) is non-hazard
            current_headroom = self.world.get_block(x, y + 2, z)
            target_step_feet = self.world.get_block(nx, y + 1, nz)
            target_step_head = self.world.get_block(nx, y + 2, nz)

            if (
                self.is_passable_air(current_headroom)
                and self.is_passable_air(target_step_feet)
                and self.is_passable_air(target_step_head)
                and not self.is_hazard(target_feet_b)
            ):
                return nx, y + 1, nz, MoveOutcome.STEP_UP, 0
            else:
                return x, y, z, MoveOutcome.BLOCKED, 0

        # -------------------------------------------------------------
        # 2. Hazard check at entry level
        # -------------------------------------------------------------
        if self.is_hazard(target_feet_b):
            return nx, y, nz, MoveOutcome.FATAL_HAZARD, 0

        # Check head clearance at entry level
        target_head_b = self.world.get_block(nx, y + 1, nz)
        if self.is_hazard(target_head_b):
            return nx, y, nz, MoveOutcome.FATAL_HAZARD, 0

        if not self.is_passable_air(target_head_b):
            # Blocked by low ceiling
            return x, y, z, MoveOutcome.BLOCKED, 0

        if not self.is_passable_air(target_feet_b):
            # Target feet block is not passable (e.g. non-solid impeding wall)
            return x, y, z, MoveOutcome.BLOCKED, 0

        # -------------------------------------------------------------
        # 3. Flat Move Check (Floor at y-1)
        # -------------------------------------------------------------
        target_floor_b = self.world.get_block(nx, y - 1, nz)
        if self.is_hazard(target_floor_b):
            return nx, y, nz, MoveOutcome.FATAL_HAZARD, 0

        if self.palette.lut_is_solid[target_floor_b]:
            # Solid floor -> successful flat move
            return nx, y, nz, MoveOutcome.SUCCESS, 0

        # -------------------------------------------------------------
        # 4. Gravity & Fall Resolution
        # -------------------------------------------------------------
        fall_dist = 0
        curr_y = y

        while True:
            fall_dist += 1
            drop_feet_y = curr_y - 1
            drop_floor_y = drop_feet_y - 1

            if drop_feet_y < WORLD_MIN_Y:
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_VOID, fall_dist

            drop_feet_b = self.world.get_block(nx, drop_feet_y, nz)
            if self.is_hazard(drop_feet_b):
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_HAZARD, fall_dist

            drop_floor_b = self.world.get_block(nx, drop_floor_y, nz)
            if self.is_hazard(drop_floor_b):
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_HAZARD, fall_dist

            if self.palette.lut_is_solid[drop_floor_b]:
                # Landed on solid ground
                if fall_dist > self.max_safe_fall:
                    return nx, drop_feet_y, nz, MoveOutcome.FATAL_FALL, fall_dist
                else:
                    return nx, drop_feet_y, nz, MoveOutcome.FALL, fall_dist

            # Still falling through air
            curr_y = drop_feet_y
            if fall_dist > 15:
                # Terminal void / fatal fall
                return nx, drop_feet_y, nz, MoveOutcome.FATAL_FALL, fall_dist

