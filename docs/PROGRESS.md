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
- [x] **T2.1**: Multi-LOD 3D CNN encoder (`models/cnn3d.py`, `tests/test_cnn3d.py`)
  - Siamese 3D ConvNeXt + SE attention + CoordConv3D
  - Layout: Channel-last `(..., X, Y, Z, C)`
  - Exact parameter count: **470,944** trainable parameters (target budget ~470,560, diff +384 / +0.08%)
  - Concentric shell fusion: `(B, 3, 16, 16, 16, 16) -> (B, 192, 256)`
- [x] **M2**: Transformer Encoder, EMA Target, JEPA Predictor & 3D Masking (`models/transformer.py`, `tests/test_transformer.py`)
  - `PreLNTransformerBlock` (Pre-LN Multi-Head Attention + MLP with residuals)
  - `MultiLODBackbone` (Siamese 3D CNN + Pos+Scale Embeddings + 4L Transformer Encoder)
  - `JEPAPredictor` (Input Proj + Mask Token + Pos+Scale Embeddings + 2L Transformer + Output Proj)
  - `sample_3d_jepa_masks`: Pure JAX JIT-safe 3D masking over 2x2x2 blocks with cross-scale complementary logic
  - `SpatialJEPA` with in-place EMA target update (`update_ema`) and stop-gradient EMA targets
  - Exact parameter counts:
    - Core Backbone (`context_encoder`): **3,647,648** params (target ~3,647,264, diff +384 / +0.01%)
    - Predictor: **1,001,920** params (target ~1,001,920, exact 0% diff)
    - Stage 1A Spatial Total (`context_encoder + predictor`): **4,649,568** params (target ~4,649,184, diff +384 / +0.008%)
- [x] **M3**: Event-Driven Multi-LOD Shell Gating (`models/gating.py`, `tests/test_gating.py`)
  - Host-side stateful cache: `prev_shells`, `prev_cnn_tokens`, `prev_z`
  - Zero learnable parameters (0 params)
  - Selective CNN evaluation when $\Delta_l > \tau_{\text{gate}}$ and conditional Transformer skip
  - Verified lossless equivalence at $\tau_{\text{gate}} = 0.0$ against dense evaluation ($\text{max diff} < 3 \times 10^{-6}$)
  - 5-step test counts: LOD-0 = 3/5, LOD-1 = 2/5, LOD-2 = 1/5, Transformer = 3/5
  - Empirical compute: **969.24 MFLOPs/step** vs 1743.6 MFLOPs/step dense baseline (44.4% compute reduction)
- [x] **M4**: Dual-LSTM Temporal Integrator & Ego-Motion 3D Shift (`models/temporal_lstm.py`, `tests/test_temporal_lstm.py`)
  - `ego_motion_shift_3d`: differentiable, JIT-compatible 3D coordinate shift with trilinear interpolation (`map_coordinates`) vmapped across batch and channels
  - `LSTMCellNNX`: standard LSTM cell in Flax NNX
  - `QueryCrossAttentionBlock`: Pre-LN Cross-Attention fusing proprioceptive query with 24 coarse 3D spatial memory slots
  - `DualLSTMTemporalIntegrator`:
    - Spatial 2x2x2 block average pooling: `(B, 192, 256) -> (B, 24, 256)`
    - Ego-motion shifting per LOD shell scale `(1.0, 2.0, 4.0)` before spatial LSTM step
    - Proprioceptive state LSTM (`12 -> 64`) with projection producing query token (`(B, 1, 192)`)
    - Action-conditioned future predictor (`action_embed(12, 64)` + MLP `256 -> 192 -> 256`) predicting pooled `z_hat_next`
    - Auxiliary Inverse Dynamics Model (IDM) head (`[z_t, z_{t+1}] -> 64 -> 12`)
    - Autoregressive multi-step `rollout(k_steps=4)`
  - Exact parameter count: **807,564** parameters (~0.808M; target ~724,864, including auxiliary IDM head)
  - Verified finite non-zero gradients via `nnx.value_and_grad` and `@nnx.jit` execution
- [x] **M5**: Baselines Architecture (`models/baselines.py`, `tests/test_baselines.py`)
  - **Baseline A (Single-LOD JEPA)**:
    - Slices single LOD shell (`x[:, lod_index:lod_index+1]`, 64 tokens)
    - Wraps `SpatialJEPA(num_shells=1)` with EMA target updating
    - Trainable parameter count: **4,648,672** params (target ~4,648,672, exact 0.000% difference)
    - Verified forward pass and finite non-zero gradients via `nnx.value_and_grad` inside `@nnx.jit` for both LOD-0 and LOD-2
    - Verified zero gradients on target encoder
  - **Baseline B (Generative 3D-MAE)**:
    - Reuses `MultiLODBackbone(num_shells=3)` context encoder (**3,647,648** params)
    - Lightweight 3D voxel decoder (**1,150,144** params): `Linear(256, 192)` + `mask_token` + pos/scale embeds + `2 × PreLNTransformerBlock(192, 6, mlp_ratio=4)` + `LayerNorm(192)` + `Linear(192, 1024)` reconstructing 4x4x4x16 voxel patches
    - Total parameters: **4,797,792** params (target ~5.126M params, within ~6.4%)
    - `patchify`: Verified coordinate matching against direct sub-volume slices
    - Computes `mse_loss` strictly over masked target patches with scalar loss and finite non-zero gradients inside `@nnx.jit`
- [ ] **T2.2**: State & action conditioning modules
- [ ] **T2.3**: Dynamics transition model

## Phase 3: Trajectory Modeling
- [ ] **T3.1**: Autoregressive sequence model
- [ ] **T3.2**: Closed-loop rollout simulation

## Phase 4: Training & Diagnostics
- [x] **T4.1**: Training losses and regularization utilities (`train/losses.py`, `tests/test_train_utils.py`)
  - `smooth_l1_loss(pred, target, beta=1.0)`
  - `vicreg_var_cov_loss(z, gamma=1.0, eps=1e-4)` variance and covariance penalties
  - `spatial_jepa_loss(pred_targets, norm_ema_targets, context_tokens, alpha=0.1, beta=0.04)`
  - `temporal_jepa_loss(pred_future, ema_future_targets, idm_logits, actions, gamma_discount=0.9, alpha=0.1, beta=0.04, omega=0.2)`
  - `cosine_ema_momentum(step, total_steps, start=0.996, end=1.0)`
- [x] **T4.2**: Checkpoint management and representation diagnostics (`train/checkpoint.py`, `train/diagnostics.py`, `tests/test_train_utils.py`)
  - `compute_rankme(embeddings, eps=1e-7)`: uncentered continuous effective rank metric (>100 on isotropic random, <2 on rank-1 collapsed)
  - `append_metrics_jsonl(path, record)`: structured epoch logging
  - `save_checkpoint(path, module, metadata)` and `load_checkpoint(path, module)`: zero-dependency, cross-platform pure-dict serialization via pickle with exact bit-level roundtrip equality (`atol=0`)
- [ ] **T4.3**: End-to-end training loops (`train_spatial_jepa.py`, `train_temporal_jepa.py`, `train_baselines.py`)

## Phase 5: Evaluation & Artifacts
- [ ] **T5.1**: Probes and simulator evaluation (`eval/`)
- [ ] **T5.2**: Publication vector figures (`paper_figs/`)

