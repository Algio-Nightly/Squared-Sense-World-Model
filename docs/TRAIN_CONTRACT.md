# Squared Sense — Training & Diagnostics Contract (JAX + Flax NNX + Optax)

## 1. Checkpointing (Zero-Dependency, Cross-Platform)
- Save/load pure-dict parameter states (`nnx.to_pure_dict(nnx.state(model, nnx.Param))`)
  converted to NumPy arrays via `pickle` (`checkpoints/<run_name>/best.pkl`, `last.pkl`).
- Also write `checkpoints/<run_name>/metrics.jsonl` (one JSON object per epoch) for paper plots.

## 2. Stage 1A Spatial JEPA Defaults (`train_spatial_jepa.py`)
- Loss: Smooth-L1 (or L1) on masked tokens vs. LayerNorm-normalized stop-grad EMA targets
        + 0.1 * context-token prediction anchor (or variance/cov on context)
        + alpha * VICReg_var (alpha=0.1) + beta * VICReg_cov (beta=0.04)
- Optimizer: AdamW (lr=3e-4, weight_decay=0.05, grad clip norm=1.0)
- Schedule: Cosine decay with 10-epoch linear warmup (80 epochs total, batch_size=32 or 64)
- EMA momentum: Cosine schedule from 0.996 -> 1.0 over total training steps.
- Diagnostics: Compute uncentered RankMe on validation embeddings every 5 epochs.

## 3. Stage 1B Temporal JEPA Defaults (`train_temporal_jepa.py`)
- Inputs: 16-step trajectory clips:
  `shells (B, 16, 3, 16, 16, 16, 16)`, `state (B, 16, 12)`, `action (B, 16)`
- Phase 1 (15 epochs): Freeze Stage 1A `SpatialJEPA`; train ONLY `DualLSTMTemporalIntegrator` (lr=3e-4).
- Phase 2 (25 epochs): Unfreeze ONLY the top 2 Transformer blocks of `context_encoder` at 0.1x lr (3e-5)
                       while keeping `DualLSTMTemporalIntegrator` at 3e-4 + 0.1x Stage 1A spatial anchor loss.
- Objective: K=4 autoregressive rollout discounted L1 loss (gamma=0.9) vs. stop-grad EMA pooled targets
             + omega * IDM cross-entropy (omega=0.2) + VICReg (alpha=0.1, beta=0.04).

## 4. Baselines (`train_baselines.py`)
- `--model b1_lod0`: `SingleLODJEPA(lod_index=0)` (same Stage 1A objective & schedule)
- `--model b2_lod2`: `SingleLODJEPA(lod_index=2)` (same Stage 1A objective & schedule)
- `--model b3_mae` : `Generative3DMAE()` (masked voxel MSE loss, AdamW lr=3e-4, 80 epochs)