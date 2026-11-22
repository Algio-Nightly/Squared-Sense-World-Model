#!/usr/bin/env bash
# ==============================================================================
# GPU Runner for Squared Sense World Model (WSL2 / Linux CUDA)
# ==============================================================================
set -euo pipefail

# If invoked from Git Bash / MSYS / Cygwin on Windows, seamlessly bridge to WSL2 Debian
case "$(uname -s)" in
    CYGWIN*|MINGW*|MSYS*)
        exec wsl.exe -d Debian bash ./run_gpu.sh "$@"
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv-wsl"

if [ ! -d "$VENV_DIR" ]; then
    echo "[run_gpu] Error: .venv-wsl virtual environment not found at $VENV_DIR"
    echo "[run_gpu] Run setup first: uv venv .venv-wsl --python 3.13"
    exit 1
fi

# Ensure non-greedy allocation and fraction limits for 4GB VRAM GPU
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "=================================================================="
echo "Squared Sense World Model — GPU Execution (WSL2)"
echo "Repo Root : $SCRIPT_DIR"
echo "Python    : $VENV_DIR/bin/python"
echo "XLA Memory: PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE, MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "=================================================================="

exec "$VENV_DIR/bin/python" "$@"

