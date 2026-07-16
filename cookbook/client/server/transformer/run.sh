#!/bin/bash
# Twinkle Server launcher for Moore Threads MUSA.
# Original CUDA version backed up as: run.sh.cuda.bak
#
# Ray custom resource name MUST be "MUSA" (matches Platform class __name__).
# Do NOT use --num-gpus (that is CUDA scheduling). Use --resources='{"MUSA":N}'.
#
# PID pressure note: this container has a low cgroup pids.max (~9830).
# Limit BLAS/OMP threads and Ray --num-cpus to avoid pthread_create failures.

set -e

export RAY_ROTATION_MAX_BYTES=1024
export RAY_ROTATION_BACKUP_COUNT=1
# Ask Ray not to overwrite visible-device env (best-effort; MUSA key may be ignored on older Ray)
export RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# --- Plan 2: cap native thread pools (avoid OpenBLAS/OMP exploding PIDs) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Reduce Ray idle worker prestart pressure when supported
export RAY_worker_maximum_startup_concurrency=8

# Keep Ray CPU claim small so it does not prestart hundreds of workers (host shows 255 CPUs).
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Plan 1: stop any previous Ray cluster cleanly ---
echo "Stopping any existing Ray cluster..."
ray stop --force 2>/dev/null || true
# Give cgroup a moment to reclaim PIDs after ray stop
sleep 2

# --- Single-card minimal bring-up (recommended first) ---
# Head node: 1 MUSA card for model worker
echo "Starting Ray head (MUSA=1, num-cpus=${RAY_NUM_CPUS})..."
MUSA_VISIBLE_DEVICES=0 ray start --head --port=6379 --num-gpus=0 \
  --num-cpus="${RAY_NUM_CPUS}" \
  --resources='{"MUSA":1}' \
  --disable-usage-stats --include-dashboard=false

# CPU-only worker for gateway / processor (no accelerator)
echo "Starting Ray CPU worker..."
MUSA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0 \
  --num-cpus="${RAY_NUM_CPUS}"

# --- Multi-card example (commented; enable after single-card works) ---
# Align ranks/nproc_per_node in server_config.yaml with MUSA count below.
# MUSA_VISIBLE_DEVICES=0,1,2,3 ray start --head --port=6379 --num-gpus=0 \
#   --num-cpus="${RAY_NUM_CPUS}" \
#   --resources='{"MUSA":4}' \
#   --disable-usage-stats --include-dashboard=false
# MUSA_VISIBLE_DEVICES=4,5,6,7 ray start --address=127.0.0.1:6379 --num-gpus=0 \
#   --num-cpus="${RAY_NUM_CPUS}" \
#   --resources='{"MUSA":4}'
# MUSA_VISIBLE_DEVICES="" ray start --address=127.0.0.1:6379 --num-gpus=0 \
#   --num-cpus="${RAY_NUM_CPUS}"

echo "Starting Twinkle server (config: server_config.yaml)..."
python server.py
