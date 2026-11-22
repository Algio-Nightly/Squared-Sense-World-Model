"""GPU Environment & Dynamic VRAM Auto-Configuration for JAX.

Safeguards training execution on GPUs with constrained VRAM (e.g., 4 GB / 6 GB) by:
1. Setting non-greedy memory allocation (XLA_PYTHON_CLIENT_PREALLOCATE=false).
2. Setting fractional VRAM caps (XLA_PYTHON_CLIENT_MEM_FRACTION=0.55).
3. Querying available VRAM at startup.
4. Auto-configuring micro-batch sizes for gradient accumulation to prevent OOM.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple


def setup_gpu_environment(mem_fraction: float = 0.55) -> None:
    """Configure JAX XLA memory management environment variables prior to XLA runtime initialization."""
    # Prevent JAX from preallocating 75-90% of entire VRAM upfront
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Cap maximum JAX allocation to prevent starving host OS / display manager
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    # Bypass Triton GEMM layout assertion edge cases and slow multi-minute autotuning on mobile GPUs
    current_xla_flags = os.environ.get("XLA_FLAGS", "")
    flags = [
        "--xla_gpu_enable_triton_gemm=false",
        "--xla_gpu_autotune_level=0",
    ]
    needed = [f for f in flags if f not in current_xla_flags]
    if needed:
        os.environ["XLA_FLAGS"] = f"{current_xla_flags} {' '.join(needed)}".strip()


def query_gpu_vram() -> Optional[Dict[str, int]]:
    """Query NVIDIA GPU memory stats via nvidia-smi.

    Returns:
        Dict with 'total_mb', 'used_mb', 'free_mb', or None if nvidia-smi is unavailable.
    """
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None

    try:
        cmd = [
            nvsmi,
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return None
        parts = [int(p.strip()) for p in lines[0].split(",")]
        return {
            "total_mb": parts[0],
            "used_mb": parts[1],
            "free_mb": parts[2],
        }
    except Exception:
        return None


def auto_configure_batching(
    batch_size: int,
    micro_batch_size: Optional[int] = None,
    backend: Optional[str] = None,
) -> Tuple[int, Optional[int]]:
    """Auto-detect VRAM constraints and configure safe micro-batch gradient accumulation.

    Args:
        batch_size: Target effective batch size.
        micro_batch_size: User-specified micro-batch size (if any).
        backend: JAX default backend ('gpu', 'cpu', etc.).

    Returns:
        Tuple of (effective_batch_size, resolved_micro_batch_size).
    """
    if micro_batch_size is not None and micro_batch_size > 0:
        return batch_size, micro_batch_size

    # Only auto-tune for GPU execution
    if backend != "gpu":
        return batch_size, None

    vram = query_gpu_vram()
    if not vram:
        # Fallback heuristic if nvidia-smi is unavailable on GPU
        resolved_micro = 8 if batch_size >= 16 and (batch_size % 8 == 0) else None
        return batch_size, resolved_micro

    total_mb = vram["total_mb"]
    free_mb = vram["free_mb"]

    # If VRAM is constrained (<= 4096 MB total or < 3000 MB free)
    if total_mb <= 4600 or free_mb < 3000:
        if batch_size % 8 == 0:
            resolved_micro = 8
        elif batch_size % 4 == 0:
            resolved_micro = 4
        else:
            resolved_micro = max(1, batch_size // 4)

        print(
            f"[VRAM Auto-Config] Detected GPU with {total_mb} MiB total ({free_mb} MiB free).\n"
            f"[VRAM Auto-Config] Constrained VRAM detected: auto-configuring micro_batch_size={resolved_micro} "
            f"with {batch_size // resolved_micro} accumulation steps (effective batch size: {batch_size})."
        )
        return batch_size, resolved_micro

    # If VRAM is moderately constrained (<= 6144 MB total)
    if total_mb <= 6500:
        if batch_size > 16 and batch_size % 16 == 0:
            resolved_micro = 16
            print(
                f"[VRAM Auto-Config] Detected 6GB GPU: micro_batch_size set to 16 "
                f"(effective batch size: {batch_size})."
            )
            return batch_size, resolved_micro

    return batch_size, None

