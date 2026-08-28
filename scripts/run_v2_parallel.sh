#!/usr/bin/env bash

set -u

cd ~/lobster-trading

mkdir -p runs/parallel_logs


echo "============================================================"
echo "LOBSTER PARALLEL EXPERIMENT"
echo "Start: $(date -Iseconds)"
echo "============================================================"


# ============================================================
# CPU: Ridge
# CPU cores 0-3
# ============================================================

echo "Starting Ridge on CPU..."

OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 \
taskset -c 0-3 \
python -u src/train_cross_section_v2.py \
  --model ridge \
  --device cpu \
  --top-k 3,5,10 \
  --run-name v2_ridge \
  > runs/parallel_logs/ridge.log 2>&1 &

RIDGE_PID=$!


# ============================================================
# GPU 0: XGBoost
# CPU helper cores 4-9
# ============================================================

echo "Starting XGBoost on GPU0..."

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=6 \
taskset -c 4-9 \
python -u src/train_cross_section_v2.py \
  --model xgb \
  --device cuda \
  --top-k 3,5,10 \
  --run-name v2_xgb_gpu0 \
  > runs/parallel_logs/xgb_gpu0.log 2>&1 &

XGB_PID=$!


# ============================================================
# GPU 1: PyTorch MLP FP16
# CPU helper cores 10-15
# ============================================================

echo "Starting MLP on GPU1..."

CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=6 \
taskset -c 10-15 \
python -u src/train_cross_section_v2.py \
  --model mlp \
  --device cuda \
  --epochs 120 \
  --batch-size 8192 \
  --top-k 3,5,10 \
  --run-name v2_mlp_gpu1 \
  > runs/parallel_logs/mlp_gpu1.log 2>&1 &

MLP_PID=$!


echo
echo "Ridge PID : $RIDGE_PID"
echo "XGB PID   : $XGB_PID"
echo "MLP PID   : $MLP_PID"

echo
echo "Use:"
echo "watch -n 1 nvidia-smi"
echo
echo "Logs:"
echo "tail -f runs/parallel_logs/*.log"

echo
echo "Waiting for all jobs..."


wait $RIDGE_PID
RIDGE_STATUS=$?

wait $XGB_PID
XGB_STATUS=$?

wait $MLP_PID
MLP_STATUS=$?


echo
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Ridge exit: $RIDGE_STATUS"
echo "XGB exit  : $XGB_STATUS"
echo "MLP exit  : $MLP_STATUS"
echo "End       : $(date -Iseconds)"
