# Squared Sense — Project Progress Checklist

Tracking implementation progress against `docs/CONTRACT.md`.

## Phase 0: Repository Setup & Scaffolding
- [x] **T0.1**: Repository structure initialized with package directories and `__init__.py` files:
  - `mod_export/` (Java mod patch)
  - `data/` (Data pipeline package)
  - `models/` (Neural network models package)
  - `train/` (Training & diagnostics package)
  - `eval/` (Evaluation probes & simulator package)
  - `paper_figs/` (Vector PDF/SVG plots)
  - `tests/` (Pytest test suite package)

## Phase 1: Data & Interface Contract
- [x] **T1.1**: Core schema implementation (`data/schema.py`) & unit tests (`tests/test_schema.py`)
  - 16-channel `Channel` IntEnum
  - `MAX_POOL_CHANNELS` and `AVG_POOL_CHANNELS` disjoint pooling tuples
  - `LODConfig` dataclass and `LOD_SPECS` for LOD-0, LOD-1, LOD-2
  - `normalize_shells(uint8_array) -> float32_array`
  - Dimensions: `STATE_DIM = 12`, `META_DIM = 6`, `COURSE_SHAPE = (16, 128, 64, 128)`
- [ ] **T1.2**: Shell extractor module (`data/shell_extractor.py`)
  - Multi-LOD voxel crop with cardinal snapping (yaw deadband ±15°)
  - Pitch vertical shift
  - Average/max channel pooling
- [ ] **T1.3**: A* oracle baseline (`data/astar_oracle.py`)
  - 3D jump-aware grid search for ground-truth pathfinding
- [ ] **T1.4**: Dataset loader & validation (`data/dataset.py`)
  - Episode parsing (`shells`, `state`, `action`, `meta`)
  - Strict episode ID train/val/test splits (70/15/15)
- [ ] **T1.5**: Minecraft Java mod export integration (`mod_export/`)
  - `/sqsense export <name>` integration and volume exporter

## Phase 2: Neural World Model Architecture
- [ ] **T2.1**: Multi-LOD 3D CNN encoder (`models/`)
  - Concentric shell fusion
- [ ] **T2.2**: State & action conditioning modules
- [ ] **T2.3**: Dynamics transition model

## Phase 3: Trajectory Modeling
- [ ] **T3.1**: Autoregressive sequence model
- [ ] **T3.2**: Closed-loop rollout simulation

## Phase 4: Training & Diagnostics
- [ ] **T4.1**: Training loops and composite loss functions (`train/`)
- [ ] **T4.2**: Checkpoint management and diagnostics

## Phase 5: Evaluation & Artifacts
- [ ] **T5.1**: Probes and simulator evaluation (`eval/`)
- [ ] **T5.2**: Publication vector figures (`paper_figs/`)

