# Squared Sense World Model — Master Data & I/O Reference

> **Document Type:** Ground-Truth Code & I/O Audit  
> **Repository:** `Squared-Sense-World-Model`  
> **Source Verification:** Fully audited against implemented Python code in `data/schema.py`, `models/`, `train/`, `eval/`, and `paper_figs/`.  
> **Scope:** Input data specifications, tensor schemas, module-by-module I/O tables, publication metric keys, and discrepancy/bug audit.

---

## 1. Master Input Data Formats (Disk $\rightarrow$ Memory)

This section documents every external data format loaded from disk across training, evaluation, and probing pipelines when running on real datasets (i.e. outside `--smoke_test` / `--demo`).

### 1.1 Episode Trajectory Files (`--data_dir`)

Episode trajectory files represent recorded agent interactions in the 3D voxel world (5 Hz control loop).

#### Accepted File Extensions & Loader Behaviors
| Script | Accepted Extensions | Code Location | Key Resolution Logic |
| :--- | :--- | :--- | :--- |
| `train/train_spatial_jepa.py` | `.npy`, `.npz` | [train_spatial_jepa.py](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_spatial_jepa.py#L222-L236) | If `.npy`: `np.load(f, mmap_mode="r")`<br>If `.npz`: `archive["voxels"]` if present, else `archive[archive.files[0]]` |
| `train/train_baselines.py` | `.npy`, `.npz` | [train_baselines.py](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_baselines.py#L31) | Delegates directly to `train_spatial_jepa.load_dataset` |
| `train/train_temporal_jepa.py` | `.npz` only | [train_temporal_jepa.py](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_temporal_jepa.py#L425-L439) | Explicit keys: `archive["shells"]`, `archive["state"]`, `archive["action"]` |
| `eval/probe_suite.py` | **None** (Crashes) | [probe_suite.py](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/probe_suite.py#L560-L562) | **Not implemented**: raises `NotImplementedError` if `--data_dir` is passed without `--smoke_test` |

#### Exact Array Specifications
| Key Name | Array Shape | Dtype on Disk | Model Dtype & Range | Axis Order & Convention | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `shells` (or `voxels`) | `(T, 3, 16, 16, 16, 16)` or `(N_clips, 16, 3, 16, 16, 16, 16)` | `uint8` (`[0..255]`) | `float32` (`[0.0..1.0]`) | Channel-Last per shell:<br>`(Time, LOD=3, X=16, Y=16, Z=16, C=16)` | Multi-LOD concentric voxel shells centered at agent position. |
| `state` | `(T, 12)` | `float32` | `float32` | `[0..11]` flat vector | Kinematic & proprioceptive measurements (velocity, displacement, orientation, health, ground contact). |
| `action` | `(T,)` | `int64` / `int32` | `int32` (`[0..11]`) | Discrete scalar | Index of movement primitive executed at step $t$. |
| `meta` | `(T, 6)` | `float32` | `float32` | `[0..5]` flat vector | Absolute world coordinates and relative 3D vector to distant goal. |

#### Dataset Splitting Mechanisms
* **`docs/CONTRACT.md` Specification:** Strict 70% train / 15% val / 15% test partitioned strictly by episode ID (never shuffle across frames).
* **`train/train_spatial_jepa.py` & `train/train_baselines.py`:** Flat frame-level split on concatenated array: `split_idx = int(0.9 * len(all_data))` $\rightarrow$ **90% train / 10% val / 0% test**.
* **`train/train_temporal_jepa.py`:** Flat clip-level split on concatenated clips: `split = int(0.85 * n_total)` $\rightarrow$ **85% train / 15% val / 0% test**.
* **`eval/probe_suite.py`:** Synthetic episodes split at `train_n = 3 * clip_len` out of 4 episodes $\rightarrow$ **75% train / 25% test / 0% val**.

---

### 1.2 Course / Arena Files (`--courses_dir`)

Course files represent static 3D navigation environments loaded by `eval/eval_navigation.py` (and accepted via CLI in `eval/train_ppo_nav.py`).

#### Exact File Format & Keys
* **File Format:** Compressed NumPy Archive (`.npz`).
* **Search Pattern:** `glob.glob(os.path.join(courses_dir, "*.npz"))` ([eval_navigation.py:41](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/eval_navigation.py#L41)).
* **Dictionary Keys:**
  1. `volume`: `(16, 128, 64, 128)` `uint8` in range `[0..255]`. **Axis Order: Channel-First `(C, X, Y, Z)`**.
  2. `start`: `(3,)` integer array/tuple specifying agent spawn `(x, y, z)` in volume coordinates.
  3. `waypoints`: `(N_wp, 3)` integer array specifying ordered checkpoint coordinates `[(x, y, z), ...]`.
  4. `shortest_path_len` *(optional)*: Float representing oracle reference path length $L_{\mathrm{ref}}$. If missing or $\le 0.0$, `eval_navigation.py` automatically computes piecewise Euclidean distances between sequential waypoints ([eval_navigation.py:49-56](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/eval_navigation.py#L49-L56)).

#### How `VoxelNavEnv` Consumes Course Files
In [eval/sim_env.py:272-280](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/sim_env.py#L272-L280):
1. **Padded Volume Creation:** To avoid out-of-bounds crop branches during shell extraction, `VoxelNavEnv` embeds the `(16, 128, 64, 128)` arena into a 64-voxel margin padded volume of shape `(16, 256, 192, 256)`.
2. **Void Masking:** `padded_volume[Channel.OUT_OF_BOUNDS = 15]` is initialized to `255`. The inner volume is written to `padded_volume[:, 64:192, 64:128, 64:192]`.
3. **Active Waypoint Beacon Injection:** In `_inject_waypoint_beacon()` ([sim_env.py:297-312](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/sim_env.py#L297-L312)), all existing voxels in `Channel.WAYPOINT (8)` are cleared to `0`, and a vertical pillar `wy - 2 : wy + 4` at `(wx, wz)` is set to `255` in both `volume` and `padded_volume`.

---

### 1.3 Complete Schema Tables

#### Table 1.1: 16-Channel Voxel Schema (`data/schema.py`)
All channels are `uint8` (`0..255`) on disk and normalized to `float32` (`0.0..1.0`) via `normalize_shells()` ($x / 255.0$).

| Channel ID | Enum Constant | Pooling Mode | Semantic Definition | Disk Values | Where Explicitly Read in Codebase |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **0** | `TERRAIN` (`SOLID`) | **AVG** | Solid opaque walkable terrain | `0`=passable, `255`=solid | `sim_env.py:218,232,238,330`, `make_figures.py:578` |
| **1** | `AIR` | **AVG** | Passable air / empty space | `0`=occupied, `255`=clear air | `sim_env.py:211,221,224,231,234,240` |
| **2** | `HAZARD` | **MAX** | Hazard (lava, fire, magma) | `0`=safe, `255`=lethal hazard | `sim_env.py:230,338`, `make_figures.py:579` |
| **3** | `LIQUID` (`WATER`) | **AVG** | Liquid (water) | `0`=none, `255`=submerged | *Never read in implemented code* |
| **4** | `CLIMBABLE` | **MAX** | Climbable / vertical steps | `0`=none, `255`=ladder/vine | *Never read in implemented code* |
| **5** | `FOLIAGE` | **AVG** | Foliage / breakable plants | `0`=none, `255`=leaves/grass | *Never read in implemented code* |
| **6** | `WORLD_Y` | **AVG** | Normalized world height | `[-64..320]` mapped to `[0..255]` | *Never read in implemented code* |
| **7** | `COLLISION_HEIGHT`| **MAX** | Collision bounding height | `0`=air, `128`=slab, `255`=full | `sim_env.py:219,239,331` |
| **8** | `WAYPOINT` (`GOAL`) | **MAX** | Active waypoint beacon marker | `0`=none, `255`=active beacon | `sim_env.py:300,301,305,307` |
| **9** | `RESOURCE` | **MAX** | Resource / interactive block | `0`=none, `255`=ore/chest | *Never read in implemented code* |
| **10** | `WALKABLE_SURFACE`| **MAX** | Solid top surface with $\ge 2$ air | `0`=no, `255`=walkable floor | `sim_env.py:220` |
| **11** | `LIGHT_LEVEL` | **AVG** | Combined light $\max(\text{sky}, \text{blk})/15$ | `[0..255]` | *Never read in implemented code* |
| **12** | `LETHAL_DROP` (`CLIFF`)| **MAX** | Cliff indicator ($\ge 3$ air below)| `0`=safe, `255`=cliff edge | *Never read in implemented code* |
| **13** | `SURFACE_SHELL` | **MAX** | Solid block bordering air | `0`=interior, `255`=boundary | *Never read in implemented code* |
| **14** | `BLOCK_HARDNESS` | **AVG** | Normalized break hardness | `[0..255]` | *Never read in implemented code* |
| **15** | `OUT_OF_BOUNDS` (`VOID`)| **MAX** | Out-of-bounds boundary padding | `0`=in bounds, `255`=void pad | `sim_env.py:140,275` |

> [!NOTE]
> Pooling partitions defined in `data/schema.py`:
> * `MAX_POOL_CHANNELS = (2, 4, 7, 8, 9, 10, 12, 13, 15)` (9 channels)
> * `AVG_POOL_CHANNELS = (0, 1, 3, 5, 6, 11, 14)` (7 channels)

---

#### Table 1.2: 12-Float Proprioceptive `state` Vector Layout
Passed into `models/temporal_lstm.py:DualLSTMTemporalIntegrator` and generated by `eval/sim_env.py:VoxelNavEnv`.

| Index | Field Name | Contract Specification | Expected by `temporal_lstm.py` | Actually Emitted by `sim_env.py` | Discrepancy / Risk |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | `vx` | Velocity X (m/s) | Fed into `state_lstm` (12 $\rightarrow$ 64) | `dx` (blocks/step, float32) | Velocity approximated as discrete displacement $\Delta x$. |
| **1** | `vy` | Velocity Y (m/s) | Fed into `state_lstm` (12 $\rightarrow$ 64) | `dy` (blocks/step, float32) | Velocity approximated as discrete displacement $\Delta y$. |
| **2** | `vz` | Velocity Z (m/s) | Fed into `state_lstm` (12 $\rightarrow$ 64) | `dz` (blocks/step, float32) | Velocity approximated as discrete displacement $\Delta z$. |
| **3** | `dx` | Displacement X (m) | **Sliced as `delta_pos[:, 0]`** in `ego_motion_shift_3d` | `dx = pos_x - prev_pos_x` | Assumes metric displacement in blocks. |
| **4** | `dy` | Displacement Y (m) | **Sliced as `delta_pos[:, 1]`** in `ego_motion_shift_3d` | `dy = pos_y - prev_pos_y` | Assumes metric displacement in blocks. |
| **5** | `dz` | Displacement Z (m) | **Sliced as `delta_pos[:, 2]`** in `ego_motion_shift_3d` | `dz = pos_z - prev_pos_z` | Assumes metric displacement in blocks. |
| **6** | `sin_yaw` | $\sin(\text{yaw})$ in $[-1.0, 1.0]$ | Fed into `state_lstm` | $\sin(\text{yaw\_rad})$ | Exact match. |
| **7** | `cos_yaw` | $\cos(\text{yaw})$ in $[-1.0, 1.0]$ | Fed into `state_lstm` | $\cos(\text{yaw\_rad})$ | Exact match. |
| **8** | `pitch` | Pitch angle | Fed into `state_lstm` | `self.pitch` **in DEGREES** ($[-90.0..90.0]$) | **Unit Mismatch:** degrees rather than radians or normalized $[-1, 1]$. |
| **9** | `health` | Player health $[0.0..1.0]$ | Fed into `state_lstm` | Hardcoded `20.0` | **Unnormalized:** raw Minecraft hearts ($20.0$). |
| **10** | `food` | Player food $[0.0..1.0]$ | Fed into `state_lstm` | Hardcoded `20.0` | **Unnormalized:** raw Minecraft food points ($20.0$). |
| **11** | `grounded` | Ground contact flag | Fed into `state_lstm` | `1.0` if `_is_solid(pos_y - 1)` else `0.0` | Binary float indicator. |

---

#### Table 1.3: 12-Action Discrete Action Space (`eval/sim_env.py`)
Action index mapping implemented in `VoxelNavEnv.step()` ([eval/sim_env.py:431-457](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/sim_env.py#L431-L457)):

| Action ID | Semantic Action | Translation / Coordinate Effect | Yaw / Pitch Rotation | Collision & Environmental Constraints |
| :---: | :--- | :--- | :--- | :--- |
| **0** | Forward | $\Delta x = \mathrm{round}(\sin\psi), \Delta z = \mathrm{round}(\cos\psi)$ | None | Blocked if target or head solid; triggers drop physics if air below. |
| **1** | Forward-Left | $\Delta x = \mathrm{round}(f_x + l_x), \Delta z = \mathrm{round}(f_z + l_z)$ | None | Diagonal move; blocked if target or overhead solid. |
| **2** | Forward-Right | $\Delta x = \mathrm{round}(f_x + r_x), \Delta z = \mathrm{round}(f_z + r_z)$ | None | Diagonal move; blocked if target or overhead solid. |
| **3** | Strafe-Left | $\Delta x = \mathrm{round}(l_x), \Delta z = \mathrm{round}(l_z)$ | None | Orthogonal left lateral translation ($l_x = -\cos\psi, l_z = \sin\psi$). |
| **4** | Strafe-Right | $\Delta x = \mathrm{round}(r_x), \Delta z = \mathrm{round}(r_z)$ | None | Orthogonal right lateral translation ($r_x = \cos\psi, r_z = -\sin\psi$). |
| **5** | Backward | $\Delta x = \mathrm{round}(-f_x), \Delta z = \mathrm{round}(-f_z)$ | None | Reversal move along active view vector. |
| **6** | Turn-Left | None | $\psi \leftarrow (\psi - 45^\circ) \bmod 360^\circ$ | Pure rotation; triggers cardinal deadband update if $\Delta\psi > 60^\circ$. |
| **7** | Turn-Right | None | $\psi \leftarrow (\psi + 45^\circ) \bmod 360^\circ$ | Pure rotation; triggers cardinal deadband update if $\Delta\psi > 60^\circ$. |
| **8** | Step-Up-Forward | Forward $\Delta x, \Delta z$ with step-up request | None | Requires: `solid(tgt, y)` AND `air(tgt, y+1)` AND `air(tgt, y+2)`. Lands at `y+1`. |
| **9** | Pitch-Up | None | $\theta \leftarrow \min(90^\circ, \theta + 15^\circ)$ | Affects vertical LOD shell pitch shift: $\mathrm{round}(4\sin\theta)\cdot\mathrm{scale}$. |
| **10** | Pitch-Down | None | $\theta \leftarrow \max(-90^\circ, \theta - 15^\circ)$ | Downward pitch shift. |
| **11** | Stay / Idle | None | None | Consumes 1 environment step; incurs $-0.02$ step penalty. |

---

#### Table 1.4: 6-Float `meta` Vector Layout
Defined in `docs/CONTRACT.md:47` and sliced in `eval/probe_suite.py:586`:

| Index | Field Name | Coordinate Reference | Unit / Range | Usage in `probe_suite.py` |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `world_x` | World origin coordinates | Blocks (float32) | Ignored during probing |
| **1** | `world_y` | World elevation | Blocks (float32) | Ignored during probing |
| **2** | `world_z` | World origin coordinates | Blocks (float32) | Ignored during probing |
| **3** | `goal_dx` | Relative vector to active goal: $x_{\mathrm{goal}} - x_{\mathrm{agent}}$ | Blocks ($[30.0..48.0]$) | **Target for Ridge Regression** (`y[:, 0]`) |
| **4** | `goal_dy` | Relative vector to active goal: $y_{\mathrm{goal}} - y_{\mathrm{agent}}$ | Blocks | **Target for Ridge Regression** (`y[:, 1]`) |
| **5** | `goal_dz` | Relative vector to active goal: $z_{\mathrm{goal}} - z_{\mathrm{agent}}$ | Blocks ($[30.0..48.0]$) | **Target for Ridge Regression** (`y[:, 2]`) |

---

## 2. Module-by-Module Data Consumption & Output Audit

### 2.1 High-Level Module I/O Matrix

| Directory | File | Primary Function / Class | Exact Input Tensors (Shape, Dtype, Range) | Exact Output Tensors / Disk Files | Critical Implicit Assumptions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/` | `cnn3d.py` | `SiameseVoxelCNN` | `x`: `(B, 3, 16, 16, 16, 16)` float32 $[0.0, 1.0]$ | `tokens`: `(B, 192, 256)` float32 | Channel-last layout `(..., X, Y, Z, C)`; CoordConv appends 3 channels. |
| `models/` | `transformer.py` | `SpatialJEPA` | `x`: `(B, 3, 16, 16, 16, 16)`<br>`vis_idx`: `(B, 96)`, `tgt_idx`: `(B, 96)` | `pred_targets`: `(B, 96, 256)`<br>`norm_ema_targets`: `(B, 96, 256)`<br>`context_tokens`: `(B, 96, 256)` | Fixed 2x2x2 block masking (4 target blocks per shell = 32 tokens/shell). |
| `models/` | `gating.py` | `EventDrivenShellGate` | `shells_single`: `(1, 3, 16, 16, 16, 16)` float32 | `z`: `(1, 192, 256)` float32 | Stateful host cache; threshold `tau_gate` skips CNN and Transformer. |
| `models/` | `temporal_lstm.py` | `DualLSTMTemporalIntegrator` | `tokens_seq`: `(B, T, 192, 256)`<br>`state_seq`: `(B, T, 12)`<br>`action_seq`: `(B, T)` int32 | `pred_future`: `(B, T-K+1, K, 256)`<br>`idm_logits`: `(B, T-1, 12)` | $T \ge K$; `state[:, 3:6]` must be block displacement $\Delta x, \Delta y, \Delta z$. |
| `models/` | `baselines.py` | `SingleLODJEPA`<br>`Generative3DMAE` | `x`: `(B, 3, 16, 16, 16, 16)`<br>`vis_idx`: `(B, 32)` or `(B, 96)` | `SingleLODJEPA`: `(B, 32, 256)`<br>`Generative3DMAE`: `(B, 96, 1024)`, `mse_loss` | SingleLOD slices shell at `lod_index`; 3D-MAE reconstructs $4\times 4\times 4\times 16=1024$ voxel patches. |
| `train/` | `losses.py` | `spatial_jepa_loss`<br>`temporal_jepa_loss` | Predictions & EMA targets | Scalar losses + metrics dictionary | VICReg variance $\gamma=1.0$, covariance decorrelation; Smooth-L1 $\beta=1.0$. |
| `train/` | `diagnostics.py` | `compute_rankme` | `embeddings`: `(N, D)` float32 | Scalar continuous effective rank $\mathcal{R}_{\mathrm{exp}} \in [1.0, D]$ | **Uncentered SVD** (never subtract mean). |
| `train/` | `checkpoint.py` | `save_checkpoint`<br>`load_checkpoint` | Flax NNX module + metadata dict | `.pkl` file: `{"state": np_pure_dict, "metadata": dict}` | Pure dict serialization converted to NumPy arrays via pickle. |
| `train/` | `train_spatial_jepa.py`| Training pipeline | `--data_dir` (.npy/.npz) | `checkpoints/m2_spatial_jepa/`<br>`best.pkl`, `last.pkl`, `metrics.jsonl` | Slices flat 90% train / 10% val; expects key `"voxels"`. |
| `train/` | `train_temporal_jepa.py`| Two-phase training pipeline| `--data_dir` (.npz), `--spatial_ckpt` | `checkpoints/m1_full_temporal/`<br>`temporal_best.pkl`, `spatial_finetuned_best.pkl`, etc. | Expects pre-chunked 7D clips `(N, 16, 3, 16, 16, 16, 16)`. |
| `train/` | `train_baselines.py` | Unified baseline pipeline | `--model`, `--data_dir` | `checkpoints/baselines/{model}/`<br>`best.pkl`, `last.pkl`, `metrics.jsonl` | Shares `load_dataset` with `train_spatial_jepa.py`. |
| `eval/` | `sim_env.py` | `VoxelNavEnv` | Discrete action `a \in [0..11]` | `obs`: `{"shells", "state", "goal_vec"}`<br>`reward`: float, `done`: bool, `info`: dict | Volume shape `(16, 128, 64, 128)` uint8; beacon injected into channel 8. |
| `eval/` | `train_ppo_nav.py` | Frozen PPO Trainer | Frozen backbone + `ActorCriticHead` | `checkpoints/ppo_nav/policy_best.pkl`<br>`ppo_metrics.jsonl` | Backbone 100% frozen (`stop_gradient`); ignores `--courses_dir`. |
| `eval/` | `eval_navigation.py` | Course Evaluator & Profiler | Course `.npz` files + checkpoints | Structured JSON to `--out_json` | Measures SR %, SPL %, Waypoint %, Collision %, MFLOPs, Latency. |
| `eval/` | `probe_suite.py` | Linear & Probe Suite | Synthetic/real episodes | Structured JSON to `--out_json` | Ridge & LogisticRegression across 3 seeds; crashes on real `--data_dir`. |
| `paper_figs/` | `make_figures.py` | Vector Figure Generator | Log files, checkpoints | `paper_figs/fig1`–`fig4` (`.pdf` + `.png`) | Springer LNEE compliant; TrueType font embedding (type 42); zero screenshots. |

---

### 2.2 Deep Code-Grounded Breakdown by File

#### `models/cnn3d.py`
* **Inputs:** Voxel tensors of shape `(N, 16, 16, 16, 16)` for single shells, or `(B, 3, 16, 16, 16, 16)` for 3-shell batches, float32 scaled to `[0.0, 1.0]`.
* **Internal Layers:**
  1. `CoordConv3D`: appends constant normalized meshgrid `(1, 16, 16, 16, 3)` $\in [-1, 1]$ along axis=-1 $\rightarrow$ channel dimension becomes $16 + 3 = 19$.
  2. Stem Conv: kernel `(2, 2, 2)`, stride `(2, 2, 2)`, VALID padding $\rightarrow$ `(N, 8, 8, 8, 64)`.
  3. Stage 1: 2 $\times$ `ConvNeXt3DSEBlock(64)` with depthwise conv, LayerNorm, pointwise MLP ($4\times$), and Squeeze-and-Excitation (reduction $4\times$).
  4. Downsample Conv: kernel `(2, 2, 2)`, stride `(2, 2, 2)` $\rightarrow$ `(N, 4, 4, 4, 128)`.
  5. Stage 2: 2 $\times$ `ConvNeXt3DSEBlock(128)`.
  6. Projection Conv: kernel `(1, 1, 1)` $\rightarrow$ `(N, 4, 4, 4, 256)`. Reshaped to `(N, 64, 256)`.
* **Outputs:** For 3 shells: `(B, 192, 256)` float32 tokens ($3 \times 64 = 192$ tokens). Trainable parameters: **~470,560**.

#### `models/transformer.py`
* **Inputs:**
  * Context encoder: `x` shape `(B, 3, 16, 16, 16, 16)` float32, optional `visible_idx` shape `(B, 96)` int32.
  * Predictor: `context_tokens` `(B, 96, 256)`, `visible_idx` `(B, 96)`, `target_idx` `(B, 96)`.
* **Masking Function (`sample_3d_jepa_masks`):** Partitions 4x4x4 grid into 8 disjoint 2x2x2 blocks of 8 tokens each. Selects 4 blocks per shell for target (32 tokens/shell $\rightarrow$ 96 total) and 4 blocks for context (32 tokens/shell $\rightarrow$ 96 total). With probability `cross_scale_prob = 0.3`, LOD-0 masks blocks that are visible in LOD-1/2.
* **Outputs:**
  * `pred_targets`: `(B, 96, 256)` float32.
  * `norm_ema_targets`: `(B, 96, 256)` float32, per-token LayerNorm normalized target representations from stop-gradient EMA encoder.
  * `context_tokens`: `(B, 96, 256)` float32.
* **Parameter Budget:** MultiLODBackbone (~3.647M) + Predictor (~1.002M) = **~4.649M params**.

#### `models/gating.py`
* **Inputs:** Single-step shells `shells_single`: `(1, 3, 16, 16, 16, 16)` float32.
* **Internal Mechanism:**
  * Computes temporal $L_1$ delta per shell: $\delta_l = \frac{1}{|V|} \sum |S_t^l - S_{t-1}^l|$.
  * If $\delta_l \le \tau_{\mathrm{gate}}$, reuses cached CNN tokens from step $t-1$.
  * If all 3 shells skip, reuses cached output representation $z_{t-1}$.
* **Outputs:** `z`: `(1, 192, 256)` float32. Learnable parameters: **0**.
* **Accounting Constants:** `MFLOPS_PER_SHELL = 128.2`, `MFLOPS_TRANSFORMER = 1359.0`, `DENSE_MFLOPS_PER_STEP = 1743.6`.

#### `models/temporal_lstm.py`
* **Inputs:**
  * `tokens_t`: `(B, 192, 256)` float32.
  * `state_t`: `(B, 12)` float32.
  * `action_t`: `(B,)` int32 $\in [0..11]$.
* **Sub-operations:**
  1. `pool_spatial`: average-pools 2x2x2 token blocks from `(B, 3, 4, 4, 4, 256)` $\rightarrow$ 24 coarse slots `(B, 3, 2, 2, 2, 256)`.
  2. `ego_motion_shift_3d`: applies trilinear coordinate shift to hidden memory $(h_{\mathrm{sp}}, c_{\mathrm{sp}})$ by `delta_pos / (8.0 * scale)` where `scale in (1.0, 2.0, 4.0)`.
  3. `spatial_lstm`: updates shifted slots with `LSTMCellNNX(256, 192)`.
  4. `state_lstm`: updates `state_t` with `LSTMCellNNX(12, 64)`, projected to query $q \in \mathbb{R}^{192}$.
  5. `cross_attn`: query $q$ attends over 24 spatial memory slots $\rightarrow$ fused context $c_t \in \mathbb{R}^{192}$.
  6. `action_embed`: embeds action into 64-D, concatenated with $c_t$ to predict next pooled latent $\hat{z}_{t+1} \in \mathbb{R}^{256}$.
* **Outputs:** `z_hat_next` `(B, 256)`, `c_t` `(B, 192)`, and rollout `pred_future` `(B, T-K+1, K, 256)`. Total params: **~724,864**.

#### `models/baselines.py`
* **`SingleLODJEPA`:** Wraps `SpatialJEPA(num_shells=1)` with input slicing at `lod_index` (0 or 2). Produces 64 tokens, 32 visible, 32 target tokens. Matches Stage 1A parameter budget (~4.649M).
* **`Generative3DMAE`:** Reuses `MultiLODBackbone(num_shells=3)` encoder (3.647M) and appends a lightweight 2-layer Transformer decoder (~1.150M) with projection head predicting patch dimension $4\times 4\times 4\times 16 = 1024$. Computes MSE loss strictly on masked target patches. Total params: **~5.126M**.

#### `train/losses.py`
* **`spatial_jepa_loss`:** $\mathcal{L}_{\mathrm{spatial}} = \text{SmoothL1}(\hat{y}, y) + \alpha \mathcal{L}_{\mathrm{var}}(z_{\mathrm{ctx}}) + \beta \mathcal{L}_{\mathrm{cov}}(z_{\mathrm{ctx}})$ with $\alpha=0.1, \beta=0.04$.
* **`temporal_jepa_loss`:** $\mathcal{L}_{\mathrm{temporal}} = \sum_{k=1}^K \gamma^{k-1} \|\hat{z}_{t+k} - z_{t+k}\|_1 + \omega \text{CrossEntropy}(\text{logits}_{\mathrm{idm}}, a_t) + \alpha \mathcal{L}_{\mathrm{var}} + \beta \mathcal{L}_{\mathrm{cov}}$ with $\gamma=0.9, \omega=0.2$.
* **`cosine_ema_momentum`:** $m_t = m_{\mathrm{start}} + (m_{\mathrm{end}} - m_{\mathrm{start}}) \cdot \frac{1}{2} [1 - \cos(\pi \cdot t / T)]$ from $0.996 \rightarrow 1.0$.

#### `train/diagnostics.py`
* **`compute_rankme(embeddings, eps=1e-7)`:**
  1. Flattens input to $Z \in \mathbb{R}^{N \times D}$.
  2. Computes singular values $s = \text{SVD}(Z)$ **without mean subtraction** (uncentered).
  3. Computes normalized singular spectrum: $p_k = s_k / (\sum s_j + \varepsilon)$.
  4. Returns $\exp(-\sum p_k \ln(p_k + \varepsilon))$.
* **`append_metrics_jsonl(path, record)`:** Serializes NumPy/JAX scalar and array values and appends valid JSON string with `\n`.

#### `train/checkpoint.py`
* **`save_checkpoint(path, module, metadata)`:** Extracts parameter tree using `nnx.state(module, nnx.Param)`, converts leaves to pure NumPy arrays via `jax.tree.map(np.asarray, ...)`, and saves via `pickle.dump(..., protocol=HIGHEST_PROTOCOL)`.
* **`load_checkpoint(path, module)`:** Restores serialized arrays to JAX device arrays and updates module in-place via `nnx.replace_by_pure_dict` and `nnx.update`.

#### `train/train_spatial_jepa.py`
* **CLI Arguments:** `--data_dir`, `--save_dir` (default `checkpoints/m2_spatial_jepa`), `--epochs` (80), `--batch_size` (32), `--lr` (3e-4), `--weight_decay` (0.05), `--warmup_epochs` (10), `--alpha` (0.1), `--beta` (0.04), `--seed` (42), `--smoke_test`.
* **Disk Output Schema:**
  * `best.pkl` / `last.pkl`: `{"state": pure_dict, "metadata": {"epoch", "global_step", "val_loss", "rankme"}}`.
  * `metrics.jsonl`: One JSON object per epoch with keys: `epoch`, `global_step`, `train_loss`, `train_pred_loss`, `val_loss`, `emb_std`, `ema_momentum`, `lr`, `epoch_sec`, and optional `rankme`.

#### `train/train_temporal_jepa.py`
* **CLI Arguments:** `--spatial_ckpt`, `--data_dir`, `--save_dir` (default `checkpoints/m1_full_temporal`), `--phase1_epochs` (15), `--phase2_epochs` (25), `--batch_size` (8), `--clip_len` (16), `--k_steps` (4), `--lr` (3e-4), `--omega` (0.2), `--alpha` (0.1), `--beta` (0.04), `--anchor_weight` (0.1), `--seed` (42), `--smoke_test`.
* **Phased Training:**
  * Phase 1 (15 epochs): spatial backbone frozen; trains `DualLSTMTemporalIntegrator`.
  * Phase 2 (25 epochs): unfreezes top 2 Transformer blocks (`blocks[2]`, `blocks[3]`) and final `norm` of context encoder at $0.1\times$ lr (3e-5) with spatial anchor loss on random frames.
* **Disk Output Schema:**
  * `temporal_best.pkl`, `spatial_finetuned_best.pkl`, `temporal_last.pkl`, `spatial_last.pkl`.
  * `metrics.jsonl`: Keys: `phase`, `epoch`, `global_step`, `train_loss`, `val_loss`, `rollout_l1`, `idm_loss`, `rankme`, `epoch_sec`.

#### `train/train_baselines.py`
* **CLI Arguments:** `--model` (`b1_lod0`, `b2_lod2`, `b3_mae`), `--data_dir`, `--save_dir` (default `checkpoints/baselines/{model}`), `--epochs` (80), `--batch_size` (32), `--lr` (3e-4), `--smoke_test`.
* **Disk Output Schema:** `best.pkl`, `last.pkl`, `metrics.jsonl` with baseline model metadata.

#### `eval/sim_env.py`
* **`VoxelNavEnv` Methods:**
  * `reset()`: Resets position to `start`, yaw=0.0, pitch=0.0, active waypoint=0. Returns observation dict: `shells` `(3, 16, 16, 16, 16)` float32, `state` `(12,)` float32, `goal_vec` `(3,)` float32.
  * `step(action)`: Executes action $\in [0..11]$. Returns `(obs, reward, done, info)` where `info` contains `{"success", "waypoints_reached", "total_waypoints", "collision", "path_length", "step"}`.

#### `eval/train_ppo_nav.py`
* **`ActorCriticHead`:** Takes `concat([feat, goal_vec_norm])` $\in \mathbb{R}^{\text{feat\_dim} + 3}$ through 2-layer MLP (128 $\rightarrow$ GELU $\rightarrow$ 128 $\rightarrow$ GELU) $\rightarrow$ Actor logits `(B, 12)` + Critic value `(B,)`. Total params: **~50K**.
* **Outputs:** Checkpoint `checkpoints/ppo_nav/policy_best.pkl` and `ppo_metrics.jsonl` with keys: `update`, `total_steps`, `loss`, `policy_loss`, `value_loss`, `entropy`, `mean_step_reward`.

#### `eval/eval_navigation.py`
* **Inputs:** Frozen checkpoints (`--policy_ckpt`, `--backbone_ckpt`, optional `--temporal_ckpt`) and `--courses_dir`.
* **Metrics Computed:**
  * $\text{SR \%} = 100 \times \frac{1}{N} \sum S_i$
  * $\text{SPL} = \frac{1}{N} \sum S_i \frac{L_i}{\max(P_i, L_i)}$
  * $\text{Waypoint Completion \%} = 100 \times \frac{\sum \text{wps\_reached}}{\sum \text{wps\_total}}$
  * $\text{Hazard Collision Rate \%} = 100 \times \frac{1}{N} \sum \mathbf{1}_{\text{hazard\_death}}$
  * Empirical MFLOPs/step and wall-clock latency (ms/step).
* **Disk Output:** Formatted JSON summary saved to `--out_json`.

#### `eval/probe_suite.py`
* **Probes Implemented:**
  1. Distant Goal Direction: Ridge regression on `meta[:, 3:6]` $\rightarrow$ $R^2$, mean cosine similarity.
  2. Local Hazard Detection: LogisticRegression on binary hazard $\rightarrow$ F1 score, AUROC.
  3. Kinematic Velocity & Occlusion Memory: Ridge on `state[:, 0:3]` ($R^2$) and LogisticRegression on past hazard visibility after turns (F1).
  4. Multi-step Rollout: Autoregressive predictions up to $H=8$ $\rightarrow$ cosine similarity, $L_1$ error, uncentered RankMe.
* **Disk Output:** Formatted JSON summary saved to `--out_json`.

#### `paper_figs/make_figures.py`
* **Output Plots:**
  * `fig1_training_and_rankme.pdf` / `.png`: Subplot (a) Validation loss log-scale curves; Subplot (b) RankMe effective rank trajectories with collapse threshold line ($r < 2$).
  * `fig2_3d_patch_similarity.pdf` / `.png`: 3-panel horizontal cross-sections of cosine similarity maps across LOD-0, LOD-1, and LOD-2 against query token #36.
  * `fig3_gating_efficiency.pdf` / `.png`: Grouped bar chart with hatched patterns comparing dense (1743.6 MFLOPs) vs. gated compute across 4 motion regimes.
  * `fig4_navigation_trajectories.pdf` / `.png`: Subplot (a) Top-down 2D floor plan rendering terrain, lava, A* path, and model tracks; Subplot (b) 3D isometric surface rendering.

---

## 3. Paper Figures & Metrics Data Requirements

### 3.1 Real Run CLI Flags vs. Synthetic Fallbacks

To generate publication figures from **genuine experimental outputs** rather than procedural fallbacks:

```bash
# 1. Generate Figure 1 from actual training logs
python paper_figs/make_figures.py \
  --out_dir paper_figs \
  --metrics_dir checkpoints/aggregated_metrics \
  --dpi 600

# 2. Generate Figure 2 using trained SpatialJEPA weights
python paper_figs/make_figures.py \
  --out_dir paper_figs \
  --ckpt checkpoints/m2_spatial_jepa/best.pkl \
  --dpi 600

# 3. Full Navigation Evaluation producing nav_eval.json
python eval/eval_navigation.py \
  --model_type m1_full \
  --backbone_ckpt checkpoints/m1_full_temporal/spatial_finetuned_best.pkl \
  --temporal_ckpt checkpoints/m1_full_temporal/temporal_best.pkl \
  --policy_ckpt checkpoints/ppo_nav/policy_best.pkl \
  --courses_dir data/courses \
  --use_gating \
  --tau_gate 0.05 \
  --out_json paper_figs/nav_m1_gated.json

# 4. Probe Suite Evaluation producing probe_eval.json
python eval/probe_suite.py \
  --model_type m1_full \
  --ckpt checkpoints/m1_full_temporal/spatial_finetuned_best.pkl \
  --temporal_ckpt checkpoints/m1_full_temporal/temporal_best.pkl \
  --data_dir data/episodes \
  --out_json paper_figs/probe_m1.json
```

---

### 3.2 Exact JSON & JSONL Key Mappings

#### Table 3.1: Metrics Log Keys (`metrics.jsonl`)
Read by `make_figures.py:plot_training_and_rankme`:

| Target Plot Element | Expected JSON Key | Source Training Script | Code Location |
| :--- | :--- | :--- | :--- |
| Epoch Index (X-axis) | `"epoch"` | `train_spatial_jepa.py`, `train_temporal_jepa.py`, `train_baselines.py` | [make_figures.py:161](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/paper_figs/make_figures.py#L161) |
| Loss Curve (Fig 1a) | `"val_loss"` | `train_spatial_jepa.py`, `train_temporal_jepa.py`, `train_baselines.py` | [make_figures.py:161](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/paper_figs/make_figures.py#L161) |
| Effective Rank (Fig 1b) | `"rankme"` | `train_spatial_jepa.py` (every 5 eps), `train_temporal_jepa.py` | [make_figures.py:162](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/paper_figs/make_figures.py#L162) |

#### Table 3.2: PPO Training Metrics (`ppo_metrics.jsonl`)
Produced by `train_ppo_nav.py`:

| Metric Name | JSON Key | Data Type | Description |
| :--- | :--- | :--- | :--- |
| PPO Update Number | `"update"` | `int` | 1-indexed rollout optimization epoch |
| Cumulative Steps | `"total_steps"` | `int` | Environment interaction steps |
| Total Surrogate Loss | `"loss"` | `float` | $\mathcal{L}_{\mathrm{policy}} + 0.5 \mathcal{L}_{\mathrm{vf}} - 0.01 \mathcal{S}$ |
| Policy Clipped Loss | `"policy_loss"` | `float` | Clipped PPO surrogate objective |
| Value Function Loss | `"value_loss"` | `float` | Mean squared error against GAE returns |
| Policy Entropy | `"entropy"` | `float` | Categorical policy distribution entropy |
| Mean Step Reward | `"mean_step_reward"` | `float` | Average per-step environment reward in rollout |

#### Table 3.3: Probe Suite Summary Keys (`probes_*.json`)
Produced by `eval/probe_suite.py`:

| Probe Category | JSON Key Path | Expected Unit / Range | Contract Evaluation Criterion |
| :--- | :--- | :--- | :--- |
| Distant Goal Direction | `["distant_goal"]["r2"]`<br>`["distant_goal"]["mean_cosine_sim"]` | $R^2 \in (-\infty, 1.0]$<br>Cosine $\in [-1.0, 1.0]$ | $R^2 > 0.85$, Cosine Sim $> 0.90$ for 30–48 block distance |
| Local Hazard Detection | `["local_hazard"]["f1"]`<br>`["local_hazard"]["auroc"]` | F1 $\in [0.0, 1.0]$<br>AUROC $\in [0.5, 1.0]$ | $\text{F1} > 0.80$, $\text{AUROC} > 0.90$ within 4 blocks ahead |
| Kinematic Velocity | `["kinematic_and_occlusion"]["velocity_r2"]` | $R^2 \in (-\infty, 1.0]$ | Velocity regression $R^2 > 0.90$ on $(v_x, v_y, v_z)$ |
| Occlusion Memory | `["kinematic_and_occlusion"]["occlusion_f1"]` | F1 score $\in [0.0, 1.0]$ | TI-JEPA Kill Test: recall hazard 5 steps after turn |
| Latent Rollout Quality | `["multistep_rollout"]["cosine_sim_h{k}"]`<br>`["multistep_rollout"]["l1_error_h{k}"]` | $k \in \{1, 2, 4, 8\}$ | Latent drift comparison against persistence baseline |
| Uncentered RankMe | `["multistep_rollout"]["rankme"]` | $\in [1.0, 256.0]$ | Effective dimensional capacity; must be $> 100$ |

#### Table 3.4: Navigation Evaluation Keys (`nav_*.json`)
Produced by `eval/eval_navigation.py`:

| Evaluation Metric | JSON Key | Format | Benchmark Target |
| :--- | :--- | :--- | :--- |
| Success Rate | `"success_rate_pct"` | Percentage float | Target: $\ge 85\%$ on unseen test courses |
| Success weighted by Path Length | `"spl"` | Float $\in [0.0, 1.0]$ | Target: $\ge 0.70$ |
| Percentage SPL | `"spl_pct"` | Percentage float | $\text{SPL} \times 100.0$ |
| Waypoint Completion | `"waypoint_completion_pct"` | Percentage float | Intermediate waypoint achievement rate |
| Catastrophic Hazard Collision | `"hazard_collision_rate_pct"` | Percentage float | Percentage of episodes ending in lava or fall $\ge 3$ blocks |
| Computational Throughput | `"empirical_mflops_per_step"`| MFLOPs float | Dense: $1743.6$; Gated target: $\le 650.0$ MFLOPs/step |
| Step Latency | `"latency_ms_per_step"` | Milliseconds float | On RTX 3050 Mobile: target $\le 15.0$ ms/step |

---

### 3.3 3D External Visualizer Export Requirements

To export the raw scenes, trajectories, and DINO-style patch activations to external 3D engines (e.g., Blender, Three.js, Open3D, PyVista), the following data structures must be exported:

1. **Course Geometry & Voxel Meshes:**
   * Source: `volume` array of shape `(16, 128, 64, 128)` uint8.
   * Extraction: Extract boolean masks for `volume[Channel.TERRAIN] >= 128` (solid terrain, hex `#B0BEC5`) and `volume[Channel.HAZARD] >= 128` (lava, hex `#FF5722`). Run a 3D greedy mesher or marching cubes algorithm to produce standard `.obj` / `.gltf` meshes.
2. **Trajectory Paths & Waypoint Beacons:**
   * Source: Logged step coordinates `[(x_0, y_0, z_0), (x_1, y_1, z_1), ...]` from `VoxelNavEnv`.
   * Export: Format as 3D polyline splines with vertex color coding: A* Oracle (Black `#000000`), M1 Multi-LOD (Dark Teal `#004D40`), B1 LOD-0 (Pink `#D81B60`), B2 LOD-2 (Orange `#E65100`). Waypoint beacons rendered as glowing translucent cylinders at `(wx, wy, wz)`.
3. **Concentric Multi-LOD Patch Attention Volumes (`fig2`):**
   * Source: Tensor of shape `(3, 4, 4, 4)` containing cosine similarity values against the query token.
   * Spatial Bounding Box Transformation: Each shell $l \in \{0, 1, 2\}$ occupies an asymmetric world bounding box centered at agent position $(p_x, p_y, p_z)$ aligned with cardinal yaw $\psi$:
     * LOD-0 ($1\times$): span $16\times 16\times 16$, patch voxel size $4\times 4\times 4$ blocks.
     * LOD-1 ($2\times$): span $32\times 32\times 32$, patch voxel size $8\times 8\times 8$ blocks.
     * LOD-2 ($4\times$): span $64\times 64\times 64$, patch voxel size $16\times 16\times 16$ blocks.
   * Visualizer Rendering: Render as 3 concentric semi-transparent nested voxel grids where cell alpha and color are driven by $\mathrm{clamp}(\cos(z_i, z_{\mathrm{query}}), 0.0, 1.0)$ using the `viridis` colormap.

---

## 4. Code vs. Contract Discrepancy & Missing-Loader Audit

This section provides an itemized audit of every code discrepancy, unimplemented loader, key mismatch, and shortcut discovered during direct inspection of the implemented `.py` files.

---

### Discrepancy 4.1: `eval/probe_suite.py` Raises `NotImplementedError` on Real Data
* **File & Line:** [eval/probe_suite.py:560-562](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/probe_suite.py#L560-L562)
* **Implemented Code:**
  ```python
  if smoke_test or data_dir is None:
      num_episodes = 4
      clip_len = 24
      data = generate_synthetic_probe_data(num_episodes=num_episodes, clip_len=clip_len)
  else:
      raise NotImplementedError(f"Loading real dataset from '{data_dir}' not implemented.")
  ```
* **Contract Specification:** `docs/EVAL_CONTRACT.md:9` mandates running the probe suite on Option B episode files.
* **Impact:** Executing `python eval/probe_suite.py --data_dir data/episodes` immediately crashes with `NotImplementedError`. Real representation evaluation cannot execute without a real data loading routine.
* **Additional Sub-Discrepancy:** The synthetic generator expects pre-existing dictionary keys `"hazard"`, `"occlusion"`, and `"ema_targets"` ([probe_suite.py:458-465](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/probe_suite.py#L458-L465)), none of which exist in the raw episode format specified in `docs/CONTRACT.md:44-47`.

---

### Discrepancy 4.2: `eval/train_ppo_nav.py` Declares `--courses_dir` but Ignores It
* **File & Line:** [eval/train_ppo_nav.py:635-639, 707-708](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/train_ppo_nav.py#L635-L639)
* **Implemented Code:**
  ```python
  parser.add_argument("--courses_dir", type=str, default=None, ...)
  ...
  # Step 2 in main():
  volume, start, waypoints = make_debug_bridge_course(seed=opts.seed)
  env = VoxelNavEnv(volume=volume, start=start, waypoints=waypoints)
  ```
* **Contract Specification:** `docs/EVAL_CONTRACT.md:18` states policies should train and evaluate across procedural course volumes.
* **Impact:** Passing `--courses_dir path/to/courses` to `train_ppo_nav.py` does nothing; the training loop strictly trains on the synthetic 1-block debug bridge course. Contrast with `eval/eval_navigation.py:362-364` which correctly invokes `load_courses_from_dir`.

---

### Discrepancy 4.3: Dictionary Key Mismatch (`"voxels"` vs `"shells"`) in `train_spatial_jepa.py` & `train_baselines.py`
* **File & Line:** [train/train_spatial_jepa.py:231-233](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_spatial_jepa.py#L231-L233) (reused in [train/train_baselines.py:31](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_baselines.py#L31))
* **Implemented Code:**
  ```python
  with np.load(f) as archive:
      key = "voxels" if "voxels" in archive else archive.files[0]
      data_list.append(archive[key])
  ```
* **Contract Specification:** `docs/CONTRACT.md:44` defines the episode dictionary key as `"shells"`.
* **Impact:** In an `.npz` archive saved with standard keys `["action", "meta", "shells", "state"]`, `"voxels"` will not be found. `archive.files[0]` evaluates alphabetically to `"action"`, loading 1D integer action arrays instead of 6D voxel shells, causing an immediate shape crash during training.

---

### Discrepancy 4.4: Missing `uint8 -> float32` Normalization in All Training Scripts
* **File & Line:**
  * [train/train_spatial_jepa.py:226, 233, 305](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_spatial_jepa.py#L305)
  * [train/train_temporal_jepa.py:432, 536](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_temporal_jepa.py#L432)
  * [train/train_baselines.py:31, 316](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_baselines.py#L316)
* **Implemented Code:**
  ```python
  # In train_spatial_jepa.py:
  batch_x = jnp.asarray(train_x[batch_idx])
  # In train_temporal_jepa.py:
  all_shells.append(archive["shells"])
  ...
  b_shells = jnp.asarray(train_dict["shells"][idx])
  ```
* **Contract Specification:** `docs/CONTRACT.md:12` and `data/schema.py:117-126` dictate that uint8 `[0..255]` on disk MUST be normalized to float32 `[0.0..1.0]` in model via `normalize_shells()`.
* **Impact:** If unnormalized `uint8` data from disk is passed directly into JAX arrays, inputs are either `uint8` or unscaled float32 in range `[0..255]`. Passing `255.0` into `SiameseVoxelCNN` causes gradient explosion, numerical overflow, and `NaN` in LayerNorm layers. (Contrast with `eval/probe_suite.py:115` and `eval/sim_env.py:156` which correctly multiply by `1.0 / 255.0`).

---

### Discrepancy 4.5: `train_temporal_jepa.py` Assumes Pre-Chunked 7D Arrays (No Online Trajectory Slicing)
* **File & Line:** [train/train_temporal_jepa.py:432-440, 170](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_temporal_jepa.py#L432-L440)
* **Implemented Code:**
  ```python
  all_shells.append(archive["shells"])
  ...
  shells_arr = np.concatenate(all_shells, axis=0)
  ...
  b, T = shells.shape[0], shells.shape[1]
  ```
* **Contract Specification:** `docs/CONTRACT.md:44` defines episode trajectories as 6D arrays `(T_ep, 3, 16, 16, 16, 16)`.
* **Impact:** If an `.npz` file contains a continuous episode trajectory of length $T_{\mathrm{ep}}=500$, `np.concatenate(all_shells, axis=0)` stacks along the time axis, producing a 6D array `(total_T, 3, 16, 16, 16, 16)`. Slicing a batch of indices `shells_arr[idx]` yields shape `(B, 3, 16, 16, 16, 16)` rather than the 7D clip `(B, T_clip, 3, 16, 16, 16, 16)` expected by `train_step_phase1`. The loader lacks online window chunking into `--clip_len` (16 steps).

---

### Discrepancy 4.6: Inconsistent File Extension Filtering Across Loaders
* **File & Line:**
  * [train/train_spatial_jepa.py:222-223](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_spatial_jepa.py#L222-L223): filters for both `.npy` and `.npz`.
  * [train/train_temporal_jepa.py:425](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_temporal_jepa.py#L425): filters strictly for `.npz` and ignores `.npy`.
  * [eval/eval_navigation.py:41](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/eval_navigation.py#L41): filters strictly for `.npz`.
* **Impact:** Datasets stored in separate `.npy` memory maps cannot be loaded by `train_temporal_jepa.py`.

---

### Discrepancy 4.7: Dataset Splitting Violations
* **File & Line:**
  * [train/train_spatial_jepa.py:239-241](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_spatial_jepa.py#L239-L241): `split_idx = int(0.9 * len(all_data))` $\rightarrow$ 90% train / 10% val, split across contiguous frames.
  * [train/train_temporal_jepa.py:441-452](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/train/train_temporal_jepa.py#L441-L452): `split = int(0.85 * n_total)` $\rightarrow$ 85% train / 15% val, split across clips.
* **Contract Specification:** `docs/CONTRACT.md:48` strictly mandates: `"Splits: 70% train / 15% val / 15% test strictly by episode ID (NEVER frame shuffle)."`
* **Impact:** Neither script creates a test split; both perform arbitrary percentage splits that do not track episode boundary IDs.

---

### Discrepancy 4.8: Pooling Mode Shortcut in `eval/sim_env.py` (Strided Subsampling vs. True Average Pooling)
* **File & Line:** [eval/sim_env.py:167, 179](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/sim_env.py#L167)
* **Implemented Code:**
  ```python
  # Scale 2 (LOD-1):
  m[_AVG_CH] = crop[_AVG_CH, ::2, ::2, ::2]
  # Scale 4 (LOD-2):
  m[_AVG_CH] = crop[_AVG_CH, ::4, ::4, ::4]
  ```
* **Contract Specification:** `docs/CONTRACT.md:13-28` and `data/schema.py:56` define average pooling for channels `(0, 1, 3, 5, 6, 11, 14)`.
* **Impact:** Slicing `::2` and `::4` is point subsampling (decimation), not average pooling. Thin 1-block obstacles or paths that fall on odd coordinates are completely erased from LOD-1 and LOD-2 observations.

---

### Discrepancy 4.9: Proprioceptive State Normalization & Units in `sim_env.py`
* **File & Line:** [eval/sim_env.py:358-370](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/sim_env.py#L358-L370)
* **Implemented Code:**
  ```python
  state = np.array([
      dx, dy, dz,  # vx, vy, vz
      dx, dy, dz,  # dx, dy, dz
      math.sin(yaw_rad), math.cos(yaw_rad),
      self.pitch,  # pitch in degrees [-90.0..90.0]
      20.0,        # health unnormalized
      20.0,        # food unnormalized
      grounded,
  ], dtype=np.float32)
  ```
* **Contract Specification:** `docs/CONTRACT.md:45` specifies `health` and `food` in $[0.0..1.0]$ and angles normalized.
* **Impact:** Pitch is emitted in degrees ($[-90.0..90.0]$), and health/food are raw unnormalized Minecraft values ($20.0$). Meanwhile, `models/temporal_lstm.py:222` treats `state[:, 3:6]` as metric displacement for `ego_motion_shift_3d`. If a user normalizes `state[:, 3:6]` without adjusting `temporal_lstm.py`, the coordinate shift will be miscalibrated.

---

### Discrepancy 4.10: `paper_figs/make_figures.py` Lacks CLI Wiring for Experimental Data in Figures 2, 3, and 4
* **File & Line:** [paper_figs/make_figures.py:768, 771, 774](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/paper_figs/make_figures.py#L768)
* **Implemented Code:**
  ```python
  # Figure 2:
  plot_3d_patch_similarity(backbone=backbone, shells_single=shells, ...)  # shells is hardcoded None
  # Figure 3:
  plot_gating_efficiency(gating_stats=None, ...)  # hardcoded None
  # Figure 4:
  plot_navigation_trajectories(volume=None, astar_path=None, model_paths=None, ...)  # hardcoded None
  ```
* **Impact:**
  * In Figure 2, because `shells` is always `None`, line 276 evaluates to `False`, forcing Figure 2 to ALWAYS plot synthetic noise tokens even when `--ckpt` is supplied!
  * Figure 3 cannot ingest JSON outputs from `eval/eval_navigation.py` via CLI.
  * Figure 4 cannot ingest trajectory coordinates or real course files via CLI.
  * Figure 1 lines 750-753 expects `--metrics_dir` to contain specifically prefixed files (`m1_metrics.jsonl`, `m2_metrics.jsonl`, `b1_metrics.jsonl`, `b3_metrics.jsonl`), whereas the training scripts default to distinct folders (`checkpoints/m2_spatial_jepa/metrics.jsonl`, etc.).

---

### Discrepancy 4.11: Internal Typo / Conflict Within `docs/CONTRACT.md` for Pooling Channels
* **File & Line:** [docs/CONTRACT.md:30-34](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/docs/CONTRACT.md#L30-L34)
* **Contract Text:**
  ```python
  MAX_POOL_CHANNELS = [2, 4, 7, 8, 9, 10, 12, 13, 15]
  AVG_POOL_CHANNELS = [0, 1, 3, 5, 6, 11, 14]

  MAX_POOL_CHANNELS = [2, 4, 6, 7, 8, 9, 10, 15]
  AVG_POOL_CHANNELS = [0, 1, 3, 5, 11, 12, 13, 14]
  ```
* **Resolution in Code:** `data/schema.py:55-56` implements the FIRST definition: Channel 6 (`WORLD_Y`) is in `AVG_POOL_CHANNELS`, and Channels 12 (`LETHAL_DROP`) and 13 (`SURFACE_SHELL`) are in `MAX_POOL_CHANNELS`. This matches lines 19, 25, and 26 of `docs/CONTRACT.md`.

---

### Discrepancy 4.12: Unimplemented Files Listed in `docs/CONTRACT.md`
* **File & Line:** `docs/CONTRACT.md:3-4` lists:
  * `data/shell_extractor.py`
  * `data/astar_oracle.py`
  * `data/dataset.py`
  * `mod_export/` Java mod patch
* **Ground-Truth Code State:** None of these three Python files exist in `data/` (`schema.py` is the only file). `mod_export/` contains only an empty `.gitkeep`.
* **Resolution in Code:** `eval/sim_env.py:41-43` includes an `except ImportError` fallback that implements `CardinalSnapper` and `extract_shells` in `sim_env.py` directly.

---

### Discrepancy 4.13: Inactive / Unchecked Channels in Simulation Environment
* **File & Line:** [eval/sim_env.py:325-339](file:///c:/Squared%20Sense/Squared%20Sense%20World%20Model/eval/sim_env.py#L325-L339)
* **Implemented Code:** In `VoxelNavEnv`, `_is_solid()` checks only `Channel.TERRAIN (0)` and `Channel.COLLISION_HEIGHT (7)`. `_is_hazard()` checks only `Channel.HAZARD (2)`.
* **Impact:** 9 of the 16 channels defined in `data/schema.py` (`LIQUID (3)`, `CLIMBABLE (4)`, `FOLIAGE (5)`, `WORLD_Y (6)`, `RESOURCE (9)`, `LIGHT_LEVEL (11)`, `LETHAL_DROP (12)`, `SURFACE_SHELL (13)`, `BLOCK_HARDNESS (14)`) are neither checked by the physics engine nor populated by `make_debug_bridge_course()`.
