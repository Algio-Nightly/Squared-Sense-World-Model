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
- [x] **T1.4**: Unified dataset loader & validation (`data/dataset.py`, `tests/test_real_data_e2e.py`)
  - Shared episode loader: `load_episode_splits` strictly splitting by episode file ID (70/15/15) without frame-shuffling, reading `"shells"` key first with fallback to `"voxels"`, and enforcing `normalize_shells()` uint8 -> float32 $[0.0, 1.0]$.
  - Temporal clip extraction: `extract_temporal_clips(clip_len, stride)` cleanly slicing continuous 6D episode trajectories without crossing episode boundaries.
  - Dynamic probe label derivation: `compute_local_hazard_labels` (4-block lookahead over lava hazard channel 2 $\ge 0.5$) and `compute_occlusion_labels` (local hazard present at $t-5$ and no longer visible at $t$).
  - End-to-end real-format test suite: `tests/test_real_data_e2e.py` verifying real `.npz` execution across spatial JEPA, temporal JEPA, baselines, probe suite, and PPO navigation.
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
- [x] **T4.3a**: Stage 1A Spatial JEPA training pipeline (`train/train_spatial_jepa.py`, `tests/test_train_spatial.py`)
  - Full CLI with `argparse`: `--data_dir`, `--save_dir`, `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--warmup_epochs`, `--alpha`, `--beta`, `--seed`, `--smoke_test`
  - Optimizer: `optax.adamw` with warmup cosine decay schedule and gradient norm clipping
  - Optimizer isolation: strictly updates ONLY `context_encoder` and `predictor` parameters, with `target_encoder` updated solely via cosine scheduled EMA (`0.996 -> 1.0`)
  - JIT-compiled optimization step (`@nnx.jit train_step`) and evaluation step (`@nnx.jit val_step`)
  - Checkpoint management: saves `last.pkl` and `best.pkl` (lowest val loss) via zero-dependency pure-dict serialization
  - Representation diagnostics: uncentered RankMe computed on pooled validation embeddings every 5 epochs and final epoch
  - Metrics logging: records train/val loss, embedding std, EMA momentum, LR, and RankMe to `metrics.jsonl`
  - Verified via smoke test and optimizer exclusion unit tests (2 passed in `tests/test_train_spatial.py`)
- [x] **T4.3b**: Stage 1B Dual-LSTM Temporal JEPA training pipeline (`train/train_temporal_jepa.py`, `tests/test_train_temporal.py`)
  - Two-phase anchored optimization pipeline with CLI (`--spatial_ckpt`, `--data_dir`, `--save_dir`, `--phase1_epochs`, `--phase2_epochs`, `--batch_size`, `--clip_len`, `--k_steps`, `--lr`, `--omega`, `--alpha`, `--beta`, `--anchor_weight`, `--seed`, `--smoke_test`)
  - Phase 1 (Frozen Spatial Backbone): Trains `DualLSTMTemporalIntegrator` on stop-gradient representations with K=4 rollout discounted L1 loss and auxiliary IDM cross-entropy; leaves spatial parameters completely untouched
  - Phase 2 (Unfreeze Top 2 Blocks + Spatial Anchor): Dual optimizer targeting `blocks[-2:]` and final `norm` of context encoder at 0.1x LR (3e-5) alongside `DualLSTMTemporalIntegrator` at 3e-4, anchored by Stage 1A masked prediction loss and spatial EMA updates (0.999)
  - Selective parameter isolation: leaves `cnn` and `blocks[:2]` strictly untouched during Phase 2
  - Evaluates continuous effective rank (RankMe) on fused validation context states $c_t$
  - Saves `temporal_best.pkl`, `spatial_finetuned_best.pkl`, and structured records in `metrics.jsonl`
  - Verified via smoke test, Phase 1 weight freeze assertion, and Phase 2 selective update assertion (3 passed in `tests/test_train_temporal.py`)
- [x] **T4.3c**: Parameter-Matched Baselines training pipeline (`train/train_baselines.py`, `tests/test_train_baselines.py`)
  - Unified CLI with `--model` choices `["b1_lod0", "b2_lod2", "b3_mae"]`, `--data_dir`, `--save_dir`, `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--warmup_epochs`, `--seed`, `--smoke_test`
  - SingleLODJEPA (`b1_lod0` & `b2_lod2`): 64-token single-shell training via `sample_3d_jepa_masks(num_shells=1)`, `spatial_jepa_loss` (Smooth-L1 + VICReg var/cov), optimizer updating only context encoder & predictor, cosine EMA update (`0.996 -> 1.0`), and uncentered RankMe diagnostics on validation embeddings
  - Generative3DMAE (`b3_mae`): 192-token 3D masked autoencoding via `sample_3d_jepa_masks(num_shells=3)`, patch MSE loss minimization over masked $4\times 4\times 4\times 16$ voxel patches, end-to-end AdamW optimization, and uncentered RankMe diagnostics on encoder representations
  - Saves `best.pkl`, `last.pkl`, and structured epoch logging in `metrics.jsonl`
  - Verified across all three baseline variants in smoke test mode (3 passed in `tests/test_train_baselines.py`)

## Phase 5: Evaluation & Artifacts
- [x] **T5.1**: Probes and simulator evaluation (`eval/`, `tests/test_probes.py`, `tests/test_sim_env.py`)
  - **Ticket E1 — Frozen Representation & Dynamics Probe Suite (`eval/probe_suite.py`, `tests/test_probes.py`)**:
    - `extract_features(model, model_type, shells, state=None, action=None, temporal_model=None, batch_size=32) -> np.ndarray`: Frozen JAX inference (`@nnx.jit`, no gradients) returning `(N, 256)` float32 features for static models (`m2_static`, `b1_lod0`, `b2_lod2`, `b3_mae`) and `(N, 448)` concatenating $z_t$ and $c_t$ for `m1_full`.
    - Four lightweight scikit-learn linear/logistic probes evaluated across 3 seeds `[0, 1, 2]` reporting mean and std:
      - `probe_distant_goal`: Ridge regression ($R^2$, mean cosine similarity) to distant waypoint coordinates.
      - `probe_local_hazard`: Logistic regression with balanced class weights (macro $F_1$, AUROC) detecting lava hazard; gracefully handles single-class edge cases in small smoke tests.
      - `probe_kinematic_and_occlusion`: Ridge regression for forward velocity ($R^2$) and logistic regression for binary occlusion ($F_1$).
      - `eval_multistep_rollout`: Closed-loop dynamics rollout over horizons $h \in \{1, 2, 4, 8\}$ reporting per-horizon cosine similarity, $L_1$ error, and uncentered RankMe continuous rank metric.
    - CLI supporting `--model_type`, `--ckpt`, `--temporal_ckpt`, `--data_dir`, `--out_json`, and `--smoke_test` with structured JSON output.
    - Verified via unit and smoke tests across all models in `tests/test_probes.py` (6 passed).
  - **Ticket E2 — Deterministic 3D Voxel Navigation Simulator (`eval/sim_env.py`, `tests/test_sim_env.py`)**:
    - `CardinalSnapper`: Yaw angle discretization with $\pm 15^\circ$ deadband hysteresis preventing flip-flop at boundaries.
    - Built-in `extract_shells`: Standalone multi-LOD voxel crop with channel-last layout `(3, 16, 16, 16, 16)` float32 in $[0, 1]$, forward offset $+12/-4$, pitch vertical shift, salience max-pooling on `MAX_POOL_CHANNELS`, and average pooling on `AVG_POOL_CHANNELS`.
    - `VoxelNavEnv`: Deterministic navigation simulator operating on $(16, 128, 64, 128)$ courses with waypoint beacon injection into channel 8 (255) for LOD-2 visibility. Pre-padded volume slicing and pairwise SIMD salience pooling executing 200 steps in $\sim 0.15$s on CPU ($< 1.0$s limit).
    - 12 discrete actions, exact voxel collision with 1-block step-up, drop physics ($\Delta y \le -3$ fall penalty), and hazard check (channel 2 $\ge 128$).
    - Exact potential-based reward function: $+10.0$ waypoint reach, $+0.25 \times (d_{\text{prev}} - d_{\text{curr}})$, $-5.0$ lava hazard, $-3.0$ fall $\ge 3$, $-0.1$ wall bump, $-0.02$ step penalty.
    - `make_debug_bridge_course`: Built-in test arena with 1-block wide bridge over lava trench and distant waypoint.
    - Verified in `tests/test_sim_env.py` covering shapes, ranges, bridge traversal, lava hazard termination, wall bumps, and speed (6 passed).
  - **Ticket E3 — Frozen-Backbone PPO Trainer & Zero-Shot Course Evaluator (`eval/train_ppo_nav.py`, `eval/eval_navigation.py`, `tests/test_eval_nav.py`)**:
    - `ActorCriticHead(feat_dim, num_actions=12)`: MLP head mapping $\text{concat}([z_t, \hat{g}_t])$ (`feat_dim + 3`) $\to \text{Linear}(128) \to \text{GELU} \to \text{Linear}(128) \to \text{GELU} \to$ Actor $\text{Linear}(12) +$ Critic $\text{Linear}(1)$ (51,469 parameters for `feat_dim=256`, target $\sim 50\text{K}$).
    - `train_bc_warmstart`: Behavioral Cloning warm-start minimizing cross-entropy on frozen feature representations for rapid policy bootstrapping.
    - `train_ppo`: Clipped PPO ($\epsilon=0.2, \gamma=0.99, \lambda=0.95, c_{\text{ent}}=0.01, c_{\text{vf}}=0.5$, AdamW with lr=3e-4, grad clip 0.5) where the perception backbone is 100% frozen (zero gradients, `stop_gradient`) and ONLY `ActorCriticHead` parameters are updated.
    - Exact parameter preservation: Verified backbone parameters are bit-identical before and after PPO updates ($\Delta = 0.0$).
    - `evaluate_navigation`: Zero-shot course evaluator computing Success Rate (SR %), SPL ($\frac{1}{N}\sum S_i \frac{l_i}{\max(p_i, l_i)}$), Waypoint Completion %, Hazard Collision Rate %, Empirical MFLOPs/step, and Latency (ms/step).
    - Verified `EventDrivenShellGate` acceleration: Evaluated gating ON vs OFF, confirming gating strictly reduces empirical compute from 1743.6 MFLOPs/step while preserving valid outputs.
    - Verified in `tests/test_eval_nav.py` (4 passed).
- [x] **T5.2**: Publication vector figures (`paper_figs/make_figures.py`, `tests/test_figures.py`)
  - **Ticket F1 — Springer Nature LNEE Vector Figures Generator (`paper_figs/make_figures.py`, `tests/test_figures.py`)**:
    - Compliant with Springer Nature LNEE publication standards: vector PDF (`.pdf`), high-resolution 600 DPI PNG (`.png`), TrueType font embedding (`pdf.fonttype=42`), grayscale-safe colorblind palette, distinct markers/linestyles, and zero raw Minecraft screenshots.
    - Implemented 4 modular plotting functions with `--demo` synthetic mode and CLI:
      1. `plot_training_and_rankme`: 2-panel figure showing Stage 1A/1B validation loss convergence and continuous effective rank trajectories ($\mathcal{R}_{\mathrm{exp}} \gg 100$ vs collapsed baseline $r < 2$).
      2. `plot_3d_patch_similarity`: DINO-style 3D voxel patch cosine self-similarity maps rendered across LOD-0 ($1\times$), LOD-1 ($2\times$), and LOD-2 ($4\times$) shells.
      3. `plot_gating_efficiency`: Grouped bar chart with grayscale-safe hatched bars comparing dense (1743.6 MFLOPs) vs. Event-Driven Shell Gating across movement regimes (Stationary, Straight Walking, Bridge Crossing, Obstacle Dodging) with speedup annotations (up to 100% compute savings).
      4. `plot_navigation_trajectories`: Clean 2D top-down and 3D isometric Matplotlib visualizer rendered directly from NumPy voxel arrays, comparing A* oracle path vs. M1 (Squared Sense), B1 (LOD-0 Myopic), and B2 (LOD-2 Coarse).
    - Verified in `tests/test_figures.py` asserting all 4 vector PDF files (>5 KB) and PNG previews are generated cleanly (2 passed).

## Phase 6: Code-First Data & I/O Audit & Unified Loader (Ticket P1)
- [x] **T6.1**: Data & I/O Master Reference (`docs/DATA_AND_IO_REFERENCE.md`)
  - Complete code-grounded audit of all loaders, models, trainers, probes, simulators, and figures.
  - Identified Discrepancies 4.1 through 4.13 across disk formats, axis ordering, pooling modes, and state normalization.
- [x] **T6.2**: Unified Real-Data Loader & Bugfix Patch (`data/dataset.py`, `tests/test_real_data_e2e.py`)
  - **Resolved 4.3, 4.4, 4.5, 4.7**: Introduced `data/dataset.py` with `load_episode_splits` and `extract_temporal_clips`, wired into `train/train_spatial_jepa.py`, `train/train_temporal_jepa.py`, `train/train_baselines.py`, and `eval/probe_suite.py`.
  - **Resolved 4.1 & 4.5**: Replaced `probe_suite.py` `NotImplementedError` with dynamic hazard/occlusion label generators and on-the-fly EMA target computation.
  - **Resolved 4.2**: Wired `--courses_dir` and multi-course cycling into `eval/train_ppo_nav.py`.
  - **Resolved 4.8**: Replaced strided subsampling on `_AVG_CH` in `eval/sim_env.py` with true $2\times 2\times 2$ and $4\times 4\times 4$ block average pooling while maintaining max-pooling on `_MAX_CH` and $< 1.0$s 200-step CPU execution.
  - **Resolved 4.9**: Normalized proprioceptive state vector in `eval/sim_env.py` (`pitch / 90.0`, `health / 20.0`, `food / 20.0`) while preserving metric displacement in `state[0:6]` for 3D ego-motion shift calibration.
  - **Resolved 4.10**: Added real experimental input flags (`--shells_npz`, `--gating_json`, `--nav_json`, `--course_npz`) and nested metrics directory loading to `paper_figs/make_figures.py`.
  - **Cleaned Contract**: Removed duplicate conflicting lines 33-34 in `docs/CONTRACT.md`.
  - **End-to-End Verified**: 11/11 tests green across `tests/test_sim_env.py` and `tests/test_real_data_e2e.py`.


