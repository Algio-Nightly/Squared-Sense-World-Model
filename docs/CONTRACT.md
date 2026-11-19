# Squared Sense — Phase 1 Data & Interface Contract (IMMUTABLE)

## 1. Repository Layout
- `mod_export/`   : Java mod patch (`/sqsense export <name>`)
- `data/`         : `schema.py`, `shell_extractor.py`, `astar_oracle.py`, `dataset.py`
- `models/`       : Phase 2-3 neural modules
- `train/`        : Phase 2-4 training & diagnostics
- `eval/`         : Phase 5 probes & simulator
- `paper_figs/`   : Vector PDF/SVG plots (NO Minecraft screenshots)
- `tests/`        : Pytest unit tests for every module

## 2. 16-Channel Voxel Schema (`uint8` [0..255] on disk -> `float32` [0.0..1.0] in model)
- ch 0  : Solid opaque walkable terrain             [AVG pool]
- ch 1  : Passable air / empty space                [AVG pool]
- ch 2  : Hazard (lava, fire, magma)                [MAX pool - salience]
- ch 3  : Liquid (water)                            [AVG pool]
- ch 4  : Climbable / vertical step (stairs/ladders)[MAX pool - salience]
- ch 5  : Foliage / fragile / breakable             [AVG pool]
- ch 6  : World Y Level (norm [-64..320] -> [0..255])[AVG pool]
- ch 7  : Collision height (0=air, 128=slab, 255=full/fence) [MAX pool - salience]
- ch 8  : Waypoint / goal beacon marker             [MAX pool - salience]
- ch 9  : Resource / interactive block              [MAX pool - salience]
- ch 10 : Walkable top surface (solid + >=2 air)    [MAX pool - salience]
- ch 11 : Combined light level (max(sky, block)/15) [AVG pool]
- ch 12 : Lethal drop / cliff indicator (>=3 air below) [MAX pool - salience]
- ch 13 : Surface shell flag (solid bordering air)  [MAX pool - salience]
- ch 14 : Normalized block hardness / diggability   [AVG pool]
- ch 15 : Out-of-bounds / void padding mask         [MAX pool]

MAX_POOL_CHANNELS = [2, 4, 7, 8, 9, 10, 12, 13, 15]
AVG_POOL_CHANNELS = [0, 1, 3, 5, 6, 11, 14]

MAX_POOL_CHANNELS = [2, 4, 6, 7, 8, 9, 10, 15]
AVG_POOL_CHANNELS = [0, 1, 3, 5, 11, 12, 13, 14]

## 3. Concentric Multi-LOD Shell Geometry (All output shapes: `(16, 16, 16, 16)` = C, X, Y, Z)
- LOD-0 (1x): span 16x16x16 | forward +12 / back -4  | lateral ±8  | base Y -4 / +12
- LOD-1 (2x): span 32x32x32 | forward +24 / back -8  | lateral ±16 | base Y -8 / +24
- LOD-2 (4x): span 64x64x64 | forward +48 / back -16 | lateral ±32 | base Y -16 / +48
- Pitch vertical shift: `shift_y = round(4 * sin(pitch_rad)) * scale` where `scale in {1, 2, 4}`.
- Cardinal snapping: snap yaw to nearest 90° cardinal axis with ±15° rotational hysteresis deadband.

## 4. Option B Episode Format (5 Hz trajectories)
- `shells`: `(T, 3, 16, 16, 16, 16)` uint8   # (Time, LOD, C, X, Y, Z)
- `state` : `(T, 12)` float32                # [vx, vy, vz, dx, dy, dz, sin_yaw, cos_yaw, pitch, health, food, grounded]
- `action`: `(T,)` int64                     # discrete movement primitive
- `meta`  : `(T, 6)` float32                 # [world_x, world_y, world_z, goal_dx, goal_dy, goal_dz]
- Splits  : 70% train / 15% val / 15% test strictly by episode ID (NEVER frame shuffle).

## 5. Option A Full-Course Volume Format
- Binary/NPZ volume: `(16, 128, 64, 128)` uint8 (C, X, Y, Z)
  centered around export origin: x in [cx-64, cx+63], y in [cy-16, cy+47], z in [cz-64, cz+63]
- Waypoints JSON: `courses/<course_name>_waypoints.json` containing ordered `[x, y, z]` local coordinates.