"""Stage 1A Spatial JEPA Training Pipeline.

Trains MultiLODBackbone context encoder and JEPAPredictor via 3D spatial masking
and stop-gradient EMA targets with VICReg regularization.

Strictly follows docs/TRAIN_CONTRACT.md and docs/MODEL_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.dataset import concat_episode_shells, load_episode_splits
from models.transformer import SpatialJEPA, sample_3d_jepa_masks
from train.checkpoint import load_checkpoint, save_checkpoint
from train.diagnostics import append_metrics_jsonl, compute_rankme
from train.losses import cosine_ema_momentum, spatial_jepa_loss


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for Stage 1A Spatial JEPA training."""
    parser = argparse.ArgumentParser(
        description="Stage 1A Spatial JEPA Training for Squared Sense World Model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing dataset (.npz files).",
    )
    parser.add_argument(
        "--region_dir",
        type=str,
        default=None,
        help="Path to directory containing baked Minecraft regions (.mmap). If provided and --data_dir is omitted, episodes are loaded or generated directly from baked terrain.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/m2_spatial_jepa",
        help="Directory to save checkpoints and metrics.jsonl.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs (default: 80).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW (default: 3e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW (default: 0.05).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of linear warmup epochs (default: 10).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Weight for VICReg variance regularizer (default: 0.1).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="Weight for VICReg covariance regularizer (default: 0.04).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PRNG initialization (default: 42).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run fast smoke test (2 epochs on 16 synthetic random frames, batch_size=4).",
    )

    parsed = parser.parse_args(args)
    if parsed.smoke_test:
        parsed.epochs = 2
        parsed.batch_size = 4
        parsed.warmup_epochs = 1

    return parsed


@nnx.jit
def train_step(
    model: SpatialJEPA,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    rng_key: jax.Array,
    ema_momentum: float,
    alpha: float,
    beta: float,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single JIT-compiled optimization step for Stage 1A Spatial JEPA.

    Args:
        model: SpatialJEPA module.
        optimizer: nnx.Optimizer managing context_encoder and predictor parameters.
        x: (B, 3, 16, 16, 16, 16) voxel input batch.
        rng_key: JAX PRNG key for 3D mask sampling.
        ema_momentum: Scalar momentum for updating target encoder.
        alpha: VICReg variance loss weight.
        beta: VICReg covariance loss weight.

    Returns:
        (loss, metrics_dict) tuple of JAX scalars.
    """
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=model.num_shells, cross_scale_prob=0.3)

    def loss_fn(m: SpatialJEPA):
        pred_targets, norm_ema, context_tokens = m(x, vis_idx, tgt_idx)
        loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Strictly update ONLY context_encoder and predictor parameters
    optimizer.update(
        (model.context_encoder, model.predictor),
        (grads.context_encoder, grads.predictor),
    )

    # In-place EMA update of target encoder
    model.update_ema(ema_momentum)

    return loss, metrics


@nnx.jit
def val_step(
    model: SpatialJEPA,
    x: jax.Array,
    rng_key: jax.Array,
    alpha: float,
    beta: float,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Single JIT-compiled evaluation step for validation loss and metrics.

    Args:
        model: SpatialJEPA module.
        x: (B, 3, 16, 16, 16, 16) voxel input batch.
        rng_key: JAX PRNG key for 3D mask sampling.
        alpha: VICReg variance loss weight.
        beta: VICReg covariance loss weight.

    Returns:
        (loss, metrics_dict) tuple of JAX scalars.
    """
    b = x.shape[0]
    vis_idx, tgt_idx = sample_3d_jepa_masks(rng_key, b, num_shells=model.num_shells, cross_scale_prob=0.3)
    pred_targets, norm_ema, context_tokens = model(x, vis_idx, tgt_idx)
    loss, metrics = spatial_jepa_loss(pred_targets, norm_ema, context_tokens, alpha=alpha, beta=beta)

    # Measure embedding standard deviation across batch/token dimensions
    emb_std = jnp.mean(jnp.std(context_tokens, axis=0))
    metrics["emb_std"] = emb_std
    return loss, metrics


@nnx.jit
def extract_unmasked_embeddings(model: SpatialJEPA, x: jax.Array) -> jax.Array:
    """Extract unmasked context representations pooled across spatial tokens.

    Args:
        model: SpatialJEPA module.
        x: (B, 3, 16, 16, 16, 16) voxel input batch.

    Returns:
        (B, 256) mean-pooled embedding vectors.
    """
    tokens = model.context_encoder(x, visible_idx=None)  # (B, 192, 256)
    return jnp.mean(tokens, axis=1)  # (B, 256)


def load_dataset(
    data_dir: Optional[str],
    smoke_test: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or synthetically generate train and validation voxel data.

    Returns:
        Tuple of (train_data, val_data) arrays of shape (N, 3, 16, 16, 16, 16).
    """
    if smoke_test or data_dir is None:
        rng = np.random.default_rng(seed)
        train_x = rng.uniform(0.0, 1.0, size=(16, 3, 16, 16, 16, 16)).astype(np.float32)
        val_x = rng.uniform(0.0, 1.0, size=(4, 3, 16, 16, 16, 16)).astype(np.float32)
        return train_x, val_x

    splits = load_episode_splits(data_dir, seed=seed)
    train_x = concat_episode_shells(splits["train"])
    val_x = concat_episode_shells(splits["val"])
    return train_x, val_x


def train_spatial_jepa(args: argparse.Namespace) -> Tuple[SpatialJEPA, float]:
    """Run full training loop for Stage 1A Spatial JEPA."""
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    best_ckpt_path = os.path.join(args.save_dir, "best.pkl")
    last_ckpt_path = os.path.join(args.save_dir, "last.pkl")

    # 1. Initialize PRNG and Model
    root_key = jax.random.key(args.seed)
    model_key, train_key = jax.random.split(root_key)
    model = SpatialJEPA(num_shells=3, rngs=nnx.Rngs(int(jax.random.bits(model_key))))

    # 2. Load Data
    effective_data_dir = args.data_dir or args.region_dir
    train_x, val_x = load_dataset(effective_data_dir, args.smoke_test, args.seed)
    num_train = len(train_x)
    num_val = len(val_x)
    steps_per_epoch = max(1, math.ceil(num_train / args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    # 3. Configure Optimizer and Schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    # Target ONLY context_encoder and predictor
    optimizer = nnx.Optimizer((model.context_encoder, model.predictor), tx, wrt=nnx.Param)

    best_val_loss = float("inf")
    global_step = 0

    print(
        f"Starting Stage 1A Spatial JEPA Training:\n"
        f"  Epochs: {args.epochs} | Steps/Epoch: {steps_per_epoch} | Total Steps: {total_steps}\n"
        f"  Batch Size: {args.batch_size} | Peak LR: {args.lr} | Warmup Epochs: {args.warmup_epochs}\n"
        f"  Train Samples: {num_train} | Val Samples: {num_val}\n"
        f"  Save Dir: {args.save_dir}\n"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Shuffle training indices per epoch
        epoch_rng = np.random.default_rng(args.seed + epoch)
        train_indices = epoch_rng.permutation(num_train)

        train_loss_accum = 0.0
        train_pred_loss_accum = 0.0
        train_var_loss_accum = 0.0
        train_cov_loss_accum = 0.0

        for step in range(steps_per_epoch):
            batch_idx = train_indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_x = jnp.asarray(train_x[batch_idx])

            ema_m = cosine_ema_momentum(global_step, total_steps, start=0.996, end=1.0)
            step_key = jax.random.fold_in(train_key, global_step)

            loss, metrics = train_step(
                model=model,
                optimizer=optimizer,
                x=batch_x,
                rng_key=step_key,
                ema_momentum=ema_m,
                alpha=args.alpha,
                beta=args.beta,
            )

            train_loss_accum += float(loss)
            train_pred_loss_accum += float(metrics["pred_loss"])
            train_var_loss_accum += float(metrics["var_loss"])
            train_cov_loss_accum += float(metrics["cov_loss"])
            global_step += 1

        avg_train_loss = train_loss_accum / steps_per_epoch
        avg_train_pred_loss = train_pred_loss_accum / steps_per_epoch

        # Validation Loop
        val_steps = max(1, math.ceil(num_val / args.batch_size))
        val_loss_accum = 0.0
        val_emb_std_accum = 0.0

        for v_step in range(val_steps):
            v_idx = np.arange(v_step * args.batch_size, min((v_step + 1) * args.batch_size, num_val))
            v_batch = jnp.asarray(val_x[v_idx])
            v_key = jax.random.fold_in(train_key, 100_000 + epoch * 100 + v_step)

            v_loss, v_metrics = val_step(
                model=model,
                x=v_batch,
                rng_key=v_key,
                alpha=args.alpha,
                beta=args.beta,
            )
            val_loss_accum += float(v_loss)
            val_emb_std_accum += float(v_metrics["emb_std"])

        avg_val_loss = val_loss_accum / val_steps
        avg_emb_std = val_emb_std_accum / val_steps

        # Compute uncentered RankMe every 5 epochs and on the final epoch
        rankme_val: Optional[float] = None
        if (epoch % 5 == 0) or (epoch == args.epochs):
            val_pooled_list = []
            for v_step in range(val_steps):
                v_idx = np.arange(v_step * args.batch_size, min((v_step + 1) * args.batch_size, num_val))
                v_batch = jnp.asarray(val_x[v_idx])
                pooled = extract_unmasked_embeddings(model, v_batch)
                val_pooled_list.append(np.asarray(pooled))
            all_val_pooled = np.concatenate(val_pooled_list, axis=0)
            rankme_val = float(compute_rankme(all_val_pooled))

        # Checkpoint: save last.pkl
        save_checkpoint(
            last_ckpt_path,
            model,
            metadata={
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": avg_val_loss,
                "rankme": rankme_val,
            },
        )

        # Checkpoint: save best.pkl if lowest val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                best_ckpt_path,
                model,
                metadata={
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": best_val_loss,
                    "rankme": rankme_val,
                },
            )

        # Metrics logging
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": avg_train_loss,
            "train_pred_loss": avg_train_pred_loss,
            "val_loss": avg_val_loss,
            "emb_std": avg_emb_std,
            "ema_momentum": ema_m,
            "lr": float(schedule(global_step)),
            "epoch_sec": time.time() - epoch_start_time,
        }
        if rankme_val is not None:
            record["rankme"] = rankme_val

        append_metrics_jsonl(metrics_path, record)

        # Compact console progress line
        rankme_str = f"{rankme_val:6.2f}" if rankme_val is not None else "   N/A"
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Step {global_step:04d} | "
            f"Train Loss: {avg_train_loss:6.4f} | "
            f"Val Loss: {avg_val_loss:6.4f} | "
            f"Emb Std: {avg_emb_std:5.3f} | "
            f"RankMe: {rankme_str} | "
            f"EMA: {ema_m:6.4f}"
        )

    print(f"\nTraining completed! Best Val Loss: {best_val_loss:6.4f}")
    return model, best_val_loss


def main():
    """CLI entrypoint."""
    args = parse_args()
    train_spatial_jepa(args)


if __name__ == "__main__":
    main()
