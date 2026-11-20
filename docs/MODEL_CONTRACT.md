# Squared Sense — Model Architecture Contract (JAX + Flax NNX, IMMUTABLE)

## Framework & Conventions
- Framework : `jax`, `jax.numpy as jnp`, `flax.nnx as nnx`, `optax`
- Layout    : STRICTLY CHANNEL-LAST `(..., X, Y, Z, C)` everywhere.
- PRNG      : Pass `rngs: nnx.Rngs` into all `nnx.Module.__init__` constructors.

## Input / Output Tensor Shapes
- Voxel shells input : `x` shape `(B, 3, 16, 16, 16, 16)` float32 in `[0.0, 1.0]`
                       (Batch, LOD=3, X=16, Y=16, Z=16, Channels=16)
- Single shell input : `shell` shape `(N, 16, 16, 16, 16)` float32 (N, X, Y, Z, C)
- Proprioception     : `state` shape `(B, 12)` float32
- Discrete action    : `action` shape `(B,)` int32 (`num_actions = 12` default)
- Spatial tokens     : `tokens` shape `(B, 192, 256)` (3 LODs * 4*4*4 = 192 tokens, D=256)
- Coarse memory slots: `slots` shape `(B, 24, 256)` (3 LODs * 2*2*2 = 24 slots, D=256)

## Module Parameter Budgets
1. `models/cnn3d.py`         : Siamese 3D ConvNeXt + SE + CoordConv  -> ~470,560 params (0.471M)
2. `models/transformer.py`   : Pos+Scale Emb (17,152) + 4L Enc (3,159,552) + 2L Pred (1,001,920)
   - Core Backbone Subtotal  : ~3,647,264 params (3.647M)
   - Stage 1A Spatial Total  : ~4,649,184 params (4.649M)
3. `models/gating.py`        : Event-Driven Shell Gate (0 learnable params; stateful host cache)
4. `models/temporal_lstm.py` : Dual-LSTM + Query Cross-Attention     -> ~724,864 params (0.725M)
   - Full Online Training    : ~5,374,048 params (5.374M)
   - Full Inference Only     : ~4,372,128 params (4.372M)
5. `models/baselines.py`     :
   - Baseline A (Single-LOD) : ~4,648,672 params (1 shell, 64 tokens)
   - Baseline B (3D-MAE)     : ~5.126M params (3.647M shared encoder + 1.478M voxel decoder)

