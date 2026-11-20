# Squared Sense — Full Pre-Training Architecture Blueprint & Specification

> **Framework**: JAX + Flax NNX (`flax.nnx`), Optax  
> **Layout**: Strictly Channel-Last `(..., X, Y, Z, C)`  
> **Core Pre-Training Paradigms**: Self-Supervised Joint-Embedding Predictive Architecture (JEPA), Ego-Motion Compensated Recurrence, and Masked Autoencoding (MAE).

---

## 1. High-Level Pre-Training Pipeline

The Squared Sense pre-training regimen splits into two complementary stages:
1. **Stage 1A — Spatial World Modeling (Spatial JEPA)**: Learns view-invariant, scale-consistent geometric representations of concentric 3D voxel shells by predicting masked spatial tokens in latent space without pixel/voxel reconstruction.
2. **Stage 1B — Temporal World Modeling (Dual-LSTM Dynamics)**: Integrates spatial representations across time with ego-motion coordinate compensation, fusing proprioceptive agent telemetry and predicting action-conditioned future latents.

```mermaid
flowchart TD
    subgraph Data["1. Input Data Pipeline (5 Hz Trajectories)"]
        RawVoxel["Raw Voxel Grid (16 Channels)"] --> ShellExt["Multi-LOD Shell Extractor (Cardinally Snapped)"]
        ShellExt --> VoxelShells["x: (B, 3, 16, 16, 16, 16)\nLOD-0 (1x), LOD-1 (2x), LOD-2 (4x)"]
        Telemetry["Proprioception"] --> StateTensor["state: (B, 12)\n[v, dp, yaw, pitch, health, ...]"]
        Actions["Primitives"] --> ActionTensor["action: (B,) int32\n(12 Discrete Actions)"]
    end

    subgraph Stage1A["Stage 1A: Spatial Pre-Training (Spatial JEPA)"]
        VoxelShells --> MaskGen["sample_3d_jepa_masks()\n(32 Vis / 32 Tgt per Shell)"]
        VoxelShells --> CtxEnc["Context Encoder (Backbone)\nSiamese 3D CNN + 4L Transformer"]
        MaskGen -.->|"visible_idx (96)"| CtxEnc
        CtxEnc --> VisTokens["Visible Context Tokens (B, 96, 256)"]
        
        VisTokens --> Predictor["JEPA Predictor (2L Transformer)\nInput Proj + Mask Tokens + Output Proj"]
        MaskGen -.->|"target_idx (96)"| Predictor
        Predictor --> PredLatents["Predicted Latents s_hat (B, 96, 256)"]

        VoxelShells --> TgtEnc["Target Encoder (Backbone, Frozen)\nUpdated via In-Place EMA (m=0.99..0.999)"]
        TgtEnc --> StopGrad["stop_gradient + Gather target_idx"]
        StopGrad --> NormTgt["LayerNorm per Token: s_tilde (B, 96, 256)"]

        PredLatents --> Loss1A["L_JEPA = MSE(s_hat, s_tilde)"]
        NormTgt --> Loss1A
    end

    subgraph Stage1B["Stage 1B: Temporal Pre-Training (Dual-LSTM & Dynamics)"]
        CtxEnc --> DenseTokens["Full Spatial Tokens (B, 192, 256)\n(Evaluated with visible_idx=None)"]
        DenseTokens --> PoolSlots["Average-Pool 2x2x2 Blocks\n-> 24 Coarse Slots (B, 24, 256)"]
        StateTensor --> StateShift["Ego-Motion Coordinates Shift\ndp / (8.0 * scale) via map_coordinates"]
        StateShift --> SpatialLSTM["Shared Spatial LSTM (256 -> 192)\n(24 Parallel Recurrent Cells)"]
        PoolSlots --> SpatialLSTM
        
        StateTensor --> StateLSTM["Proprioceptive LSTM (12 -> 64)"]
        StateLSTM --> StateProj["Linear Proj (64 -> 192) -> Query q_t (B, 1, 192)"]
        
        SpatialLSTM --> CrossAttn["Pre-LN Query Cross-Attention Block\nQuery: q_t | Keys/Values: 24 Slots"]
        StateProj --> CrossAttn
        CrossAttn --> FusedCtx["Fused Context c_t (B, 192)"]
        
        FusedCtx --> DynPred["Future Predictor MLP (256 -> 192 -> 256)"]
        ActionTensor --> ActEmb["Action Embedding (12, 64)"]
        ActEmb --> DynPred
        DynPred --> NextLatent["Predicted Next Latent z_hat_{t+1} (B, 256)"]
        
        DenseTokens --> MeanPool["Global Mean Pool -> z_t (B, 256)"]
        MeanPool --> IDM["Inverse Dynamics Head (512 -> 64 -> 12)\n[z_t, z_{t+1}] -> action logits"]
        NextLatent --> Loss1B["L_temporal = Rollout_MSE + lambda * L_IDM"]
        IDM --> Loss1B
    end

    Loss1A -.->|"Pre-trains spatial features"| CtxEnc
    Loss1B -.->|"Pre-trains temporal dynamics & physics"| SpatialLSTM
```

---

## 2. Stage 1A: Spatial JEPA Architecture

The spatial representation learner is formulated as a **Non-generative Joint-Embedding Predictive Architecture**. It operates on concentric shells with 3D block-structured masking.

```mermaid
flowchart LR
    subgraph Inputs["Inputs"]
        X["x: (B, 3, 16, 16, 16, 16)"]
        M_vis["visible_idx: (B, 96)"]
        M_tgt["target_idx: (B, 96)"]
    end

    subgraph CtxPath["Context Path (Trainable Gradient Flow)"]
        X --> CNN_ctx["Siamese 3D ConvNeXt\nCoordConv + 2x SE Stages\n(470,944 params)"]
        CNN_ctx --> RawTokens_ctx["Tokens: (B, 192, 256)"]
        RawTokens_ctx --> AddPos_ctx["Add Spatial Embed (64, 256)\n+ Scale Embed (3, 1, 256)"]
        AddPos_ctx --> GatherVis["jnp.take_along_axis(visible_idx)"]
        M_vis -.-> GatherVis
        GatherVis --> VisTok["(B, 96, 256)"]
        VisTok --> Trans_ctx["4 x PreLNTransformerBlock(256, H=8)\n+ Final LayerNorm(256)\n(3,159,552 params)"]
        Trans_ctx --> ContextOut["u_vis: (B, 96, 256)"]
    end

    subgraph PredictorBlock["JEPA Predictor (Trainable Gradient Flow)"]
        ContextOut --> InProj["nnx.Linear(256, 192)"]
        InProj --> AddPredPosCtx["Add Pred Pos+Scale Embed"]
        
        MaskTok["mask_token: (1, 1, 192)"] --> TgtQueries["Broadcast + Add Pred Pos+Scale at target_idx"]
        M_tgt -.-> TgtQueries
        
        AddPredPosCtx --> Concat["Concatenate along Sequence Axis\n[ctx (96), queries (96)] -> (B, 192, 192)"]
        TgtQueries --> Concat
        
        Concat --> PredTrans["2 x PreLNTransformerBlock(192, H=6)\n+ Final LayerNorm(192)\n(890,112 params)"]
        PredTrans --> SliceTgt["Slice Target Positions [:, 96:, :]"]
        SliceTgt --> OutProj["nnx.Linear(192, 256)\n(49,408 params)"]
        OutProj --> PredTargets["s_hat: (B, 96, 256)"]
    end

    subgraph TargetPath["Target Path (EMA Updated, No Gradients)"]
        X --> CNN_tgt["Siamese 3D ConvNeXt\n(Target Weights)"]
        CNN_tgt --> Trans_tgt["4 x PreLNTransformerBlock(256)\n(Full 192 Tokens Evaluated)"]
        Trans_tgt --> StopG["jax.lax.stop_gradient"]
        StopG --> GatherTgt["jnp.take_along_axis(target_idx)"]
        M_tgt -.-> GatherTgt
        GatherTgt --> PerTokenNorm["Per-Token LayerNorm\n(t - mu) / sqrt(sigma^2 + eps)"]
        PerTokenNorm --> NormTargets["s_tilde: (B, 96, 256)"]
    end

    subgraph LossObjective["Optimization Objective"]
        PredTargets --> L2["L_JEPA = (1 / N_tgt) || s_hat - s_tilde ||^2"]
        NormTargets --> L2
    end
```

### 3D Masking Specification (`sample_3d_jepa_masks`)
- **Grid Structure**: $4 \times 4 \times 4$ token grid partitioned into $8$ disjoint $2 \times 2 \times 2$ blocks ($8$ tokens each).
- **Target vs. Visible**: $4$ blocks ($32$ tokens) selected as target, remaining $4$ blocks ($32$ tokens) visible per shell ($96$ vis / $96$ tgt total across 3 shells).
- **Cross-Scale Complementary Masking**: With probability $p=0.3$, target blocks in LOD-0 are kept strictly visible in LOD-1 and LOD-2, forcing the context encoder to leverage multi-scale context to resolve fine spatial detail.

---

## 3. Stage 1B: Dual-LSTM Temporal Integrator & Dynamics

Stage 1B trains temporal persistence and physics grounding across $5\text{ Hz}$ trajectory sequences.

```mermaid
flowchart TD
    subgraph StepT["Single Temporal Integration Step t"]
        TokensT["Spatial Tokens z_t: (B, 192, 256)"] --> Pool["Pool 2x2x2 Blocks -> 24 Slots (B, 24, 256)"]
        StateT["State s_t: (B, 12)"] --> DeltaP["Extract Displacement dp = s_t[:, 3:6]"]
        
        subgraph EgoMotion["Ego-Motion Coordinate Compensation"]
            PrevH_sp["Prev h_sp: (B, 3, 2, 2, 2, 192)"]
            DeltaP --> Shift["map_coordinates(vol, coords + dp / (8 * scale))\nfor scale in (1.0, 2.0, 4.0)"]
            PrevH_sp --> Shift
            Shift --> ShiftedH_sp["Shifted Memory Slots: (B, 24, 192)"]
        end

        subgraph SpatialRecurrence["Spatial LSTM"]
            Pool --> LSTM_sp["Shared LSTMCellNNX(256, 192)\nApplied across all 24 slots in parallel"]
            ShiftedH_sp --> LSTM_sp
            LSTM_sp --> NewH_sp["New h_sp (B, 24, 192)"]
        end

        subgraph ProprioceptiveRecurrence["Proprioceptive State LSTM"]
            StateT --> LSTM_st["LSTMCellNNX(12, 64)"]
            PrevH_st["Prev h_st: (B, 64)"] --> LSTM_st
            LSTM_st --> NewH_st["New h_st (B, 64)"]
            NewH_st --> Proj_st["nnx.Linear(64, 192) -> q_t: (B, 1, 192)"]
        end

        subgraph Fusion["Cross-Attention Memory Query"]
            Proj_st --> CrossMHA["Pre-LN Cross Attention (H=6, D=192)\nQuery: q_t (B, 1, 192)\nKeys/Values: NewH_sp (B, 24, 192)"]
            NewH_sp --> CrossMHA
            CrossMHA --> MLP_fused["MLP (192 -> 384 -> 192) with Residual"]
            MLP_fused --> FusedContext["c_t: (B, 192)"]
        end

        subgraph ActionPredictor["Action-Conditioned Forward Predictor"]
            FusedContext --> CatAct["Concatenate [c_t, action_embed] -> (B, 256)"]
            ActionT["action_t: (B,) int32"] --> ActEmb["nnx.Embed(12, 64)"]
            ActEmb --> CatAct
            CatAct --> PredMLP["2-Layer MLP: Linear(256, 192) + GELU + Linear(192, 256)"]
            PredMLP --> PredNextZ["z_hat_{t+1}: (B, 256)"]
        end
    end

    subgraph RolloutAutoregression["Multi-Step Autoregressive Rollout (k=4 steps)"]
        PredNextZ --> RolloutStep1["Step t+1: Feed z_hat_{t+1} as input token slots"]
        RolloutStep1 --> RolloutStep2["Step t+2: Feed z_hat_{t+2}"]
        RolloutStep2 --> RolloutStepK["Step t+K: Feed z_hat_{t+K}"]
        RolloutStepK --> LossDyn["Rollout Loss: sum_{k=1}^K || z_hat_{t+k} - z_{t+k} ||^2"]
    end

    subgraph InverseDynamics["Auxiliary Inverse Dynamics Head (IDM)"]
        Z_t["z_t = mean(tokens_t, axis=spatial) (B, 256)"] --> Pair["Concat [z_t, z_{t+1}] -> (B, 512)"]
        Z_tplus1["z_{t+1} = mean(tokens_{t+1}, axis=spatial) (B, 256)"] --> Pair
        Pair --> IDM_MLP["MLP: Linear(512, 64) + GELU + Linear(64, 12)"]
        IDM_MLP --> ActLogits["Predicted Action Logits (B, 12)"]
        ActLogits --> LossIDM["Cross-Entropy(logits, action_t)"]
    end
```

---

## 4. Pre-Training Baselines Architecture

To rigorously ablate multi-scale concentric representation learning and self-supervised prediction targets, two baselines are implemented:

```mermaid
flowchart TD
    subgraph BaselineA["Baseline A: Single-LOD JEPA (Scale Ablation)"]
        X_single["Input Voxel Shells: (B, 3, 16, 16, 16, 16)"]
        SliceLOD["Slice Single Shell: x[:, lod_index : lod_index+1]\n(LOD-0 or LOD-2: 64 tokens)"]
        X_single --> SliceLOD
        SliceLOD --> JEPA_1Shell["SpatialJEPA(num_shells=1)\nContext Backbone: 3,647,136 params\nPredictor: 1,001,536 params\nTotal Trainable: 4,648,672 params"]
        JEPA_1Shell --> Out_single["(pred_targets, norm_ema_targets, context_tokens)"]
    end

    subgraph BaselineB["Baseline B: Generative 3D-MAE (Reconstruction Ablation)"]
        X_multi["Input Voxel Shells: (B, 3, 16, 16, 16, 16)"]
        X_multi --> Enc_mae["MultiLODBackbone(num_shells=3)\n(3,647,648 params, Visible Tokens Only)"]
        Enc_mae --> CtxTokens_mae["Visible Context Tokens: (B, 96, 256)"]
        
        CtxTokens_mae --> DecProj["nnx.Linear(256, 192)"]
        DecProj --> DecSeq["Concat [ctx_proj (96), mask_queries (96)] -> (B, 192, 192)"]
        DecSeq --> DecTrans["2 x PreLNTransformerBlock(192, H=6, mlp=4)\n+ LayerNorm(192) (890,112 params)"]
        DecTrans --> DecHead["nnx.Linear(192, 1024)\n(197,632 params)"]
        DecHead --> PredPatches["Predicted Voxel Patches: (B, 96, 1024)"]
        
        X_multi --> Patchify["patchify(x): (B, 192, 1024)\n4x4x4x16 Voxel Sub-volumes"]
        Patchify --> TgtPatches["Gather target_idx -> Ground Truth Patches (B, 96, 1024)"]
        
        PredPatches --> MAELoss["MSE Loss strictly on target patches"]
        TgtPatches --> MAELoss
    end
```

---

## 5. Event-Driven Shell Gating at Inference

During inference (and continuous deployment), identical or static shells bypass CNN evaluation via temporal difference thresholding $\tau_{\text{gate}}$.

```mermaid
flowchart LR
    ShellInput["x_t: (1, 3, 16, 16, 16, 16)"] --> DeltaComp["Delta_l = mean(|x_{t, l} - prev_shells_l|)"]
    
    DeltaComp --> Cond0{"Delta_0 > tau_gate?"}
    DeltaComp --> Cond1{"Delta_1 > tau_gate?"}
    DeltaComp --> Cond2{"Delta_2 > tau_gate?"}
    
    Cond0 -- Yes --> CNN0["CNN(shell_0) -> tokens_0"]
    Cond0 -- No --> Cache0["Reuse prev_cnn_tokens[0]"]
    
    Cond1 -- Yes --> CNN1["CNN(shell_1) -> tokens_1"]
    Cond1 -- No --> Cache1["Reuse prev_cnn_tokens[1]"]
    
    Cond2 -- Yes --> CNN2["CNN(shell_2) -> tokens_2"]
    Cond2 -- No --> Cache2["Reuse prev_cnn_tokens[2]"]
    
    CNN0 & Cache0 & CNN1 & Cache1 & CNN2 & Cache2 --> ConcatTok["tokens = [t0, t1, t2] (1, 192, 256)"]
    
    ConcatTok --> TransCond{"Any Delta_l > tau_gate?"}
    TransCond -- Yes --> RunTrans["Backbone.encode_tokens(tokens) -> z_t"]
    TransCond -- No --> SkipTrans["Return cached prev_z\n(Transformer Skipped!)"]
    
    RunTrans --> UpdateCache["Update Host Cache:\nprev_shells, prev_cnn_tokens, prev_z"]
```

---

## 6. Comprehensive Layer & Parameter Specification

| Module / Component | Output Tensor Shape | Trainable Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Input Voxel Shells** | `(B, 3, 16, 16, 16, 16)` | $0$ | 3 concentric shells, 16 channels, float32 $[0.0, 1.0]$ |
| **CoordConv3D** | `(B, 3, 16, 16, 16, 19)` | $0$ | Normalized meshgrid $(u, v, w) \in [-1, 1]$ concatenated |
| **SiameseVoxelCNN** | `(B, 192, 256)` | $470,944$ | ConvNeXt 3D + Squeeze-and-Excitation + $1 \times 1 \times 1$ Proj |
| **Spatial Pos Embed** | `(1, 64, 256)` | $16,384$ | Learnable 3D spatial coordinate embedding |
| **Scale Pos Embed** | `(1, 3, 1, 256)` | $768$ | Learnable LOD shell scale embedding $(1\times, 2\times, 4\times)$ |
| **Backbone Transformer** | `(B, N_vis, 256)` | $3,159,552$ | 4 Pre-LN Blocks ($D=256, H=8, \text{mlp}=4$) + LayerNorm |
| **Core Backbone Subtotal** | — | **3,647,648** | Target ~3,647,264 (diff: $+384$ / $+0.01\%$) |
| **JEPA Predictor Proj In** | `(B, N_vis, 192)` | $49,344$ | Linear projection $256 \to 192$ |
| **JEPA Predictor Embeds** | `(1, 192, 192)` | $13,056$ | Mask token $(192)$ + Spatial $(12,288)$ + Scale $(576)$ |
| **JEPA Predictor Blocks** | `(B, 192, 192)` | $890,112$ | 2 Pre-LN Blocks ($D=192, H=6, \text{mlp}=4$) + LayerNorm |
| **JEPA Predictor Proj Out**| `(B, N_tgt, 256)` | $49,408$ | Linear projection $192 \to 256$ |
| **JEPA Predictor Subtotal**| — | **1,001,920** | Target ~1,001,920 (exact $0.000\%$ diff) |
| **Stage 1A Total (Encoder+Pred)** | — | **4,649,568** | Target ~4,649,184 (diff: $+384$ / $+0.008\%$) |
| **Spatial LSTM (24 slots)**| `(B, 24, 192)` | $344,832$ | Shared `LSTMCellNNX(256, 192)` across 24 slots |
| **State LSTM & Proj** | `(B, 1, 192)` | $32,192$ | `LSTMCellNNX(12, 64)` ($19,712$) + `Linear(64, 192)` ($12,480$) |
| **Query Cross-Attention** | `(B, 192)` | $297,408$ | Pre-LN Cross-MHA ($148,224$) + MLP ($148,032$) + LNs ($1,152$) |
| **Action Conditioning & MLP** | `(B, 256)` | $99,520$ | `Embed(12, 64)` ($768$) + 2L Future MLP ($98,752$) |
| **Auxiliary IDM Head** | `(B, 12)` | $33,612$ | `Linear(512, 64)` ($32,832$) + `Linear(64, 12)` ($780$) |
| **Stage 1B Integrator Total** | — | **807,564** | Target ~724,864 (including auxiliary IDM head) |
| **Baseline A (Single-LOD)**| `(B, 32, 256)` | **4,648,672** | Target ~4,648,672 (exact $0.000\%$ diff) |
| **Baseline B Decoder** | `(B, 96, 1024)` | **1,150,144** | 2L Transformer + `Linear(192, 1024)` patch head |
| **Baseline B (3D-MAE) Total**| — | **4,797,792** | Target ~5.126M (diff: $-6.4\%$, within budget) |

---

## 7. Loss Formulations & Training Objectives

### Stage 1A: Spatial JEPA Loss
$$\mathcal{L}_{\text{Spatial}} = \frac{1}{B \cdot N_{\text{tgt}}} \sum_{b=1}^B \sum_{i \in \text{target\_idx}} \|\hat{s}_{b, i} - \tilde{s}_{b, i}\|_2^2$$
where $\tilde{s}_{b, i} = \frac{s^{\text{target}}_{b, i} - \mu(s^{\text{target}}_{b, i})}{\sqrt{\sigma^2(s^{\text{target}}_{b, i}) + 10^{-6}}}$ is the per-token standardized EMA target representation, and $\hat{s}_{b, i}$ is the predictor output.  
**Target Encoder Momentum Schedule**:
$$\theta_{\text{target}} \leftarrow m \cdot \theta_{\text{target}} + (1 - m) \cdot \theta_{\text{context}}, \quad m \in [0.99, 0.999]$$

### Stage 1B: Temporal Dynamics & Auxiliary IDM Loss
$$\mathcal{L}_{\text{Temporal}} = \frac{1}{B \cdot T_{\text{valid}} \cdot K} \sum_{b=1}^B \sum_{t=1}^{T_{\text{valid}}} \sum_{k=1}^K \|\hat{z}_{b, t+k} - z_{b, t+k}\|_2^2 + \lambda_{\text{IDM}} \mathcal{L}_{\text{IDM}}$$
$$\mathcal{L}_{\text{IDM}} = \frac{1}{B (T - 1)} \sum_{b=1}^B \sum_{t=1}^{T-1} \text{CrossEntropy}\left( \text{IDM}(z_{b, t}, z_{b, t+1}), a_{b, t} \right)$$
where $z_t = \frac{1}{192} \sum_{i=1}^{192} \text{tokens}_{t, i} \in \mathbb{R}^{256}$ is the globally pooled ground-truth latent state at time $t$.

### Baseline B: 3D-MAE Voxel Patch Reconstruction Loss
$$\mathcal{L}_{\text{MAE}} = \frac{1}{B \cdot N_{\text{tgt}}} \sum_{b=1}^B \sum_{i \in \text{target\_idx}} \|\hat{p}_{b, i} - p_{b, i}\|_2^2$$
where $p_{b, i} \in \mathbb{R}^{1024}$ is the flattened $4 \times 4 \times 4 \times 16$ ground-truth voxel sub-volume patch.
