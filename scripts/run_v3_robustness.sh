#!/usr/bin/env bash

set -u

cd ~/lobster-trading

SEEDS=(1 7 42 123 2026 3407 8765 9999)

mkdir -p runs/v3_launcher_logs


run_one() {
    GPU=$1
    CPUSET=$2
    MODEL=$3
    SEED=$4

    NAME="v3_${MODEL}_seed${SEED}"

    echo "START $NAME on GPU $GPU / CPUs $CPUSET"

    CUDA_VISIBLE_DEVICES=$GPU \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    taskset -c "$CPUSET" \
    python -u src/train_cross_section_v3.py \
      --model "$MODEL" \
      --seed "$SEED" \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --dropout 0.10 \
      --hidden 1024,512,256 \
      --clip-q 0.01 \
      --epochs 200 \
      --patience 8 \
      --eval-every 5 \
      --batch-size 16384 \
      --rank-day-batch 128 \
      --top-k 3,5,10 \
      --run-name "$NAME" \
      > "runs/v3_launcher_logs/${NAME}.log" 2>&1
}


for ((i=0; i<${#SEEDS[@]}; i+=2)); do

    S0=${SEEDS[$i]}
    S1=${SEEDS[$((i+1))]}

    echo
    echo "============================================================"
    echo "WAVE $((i/2+1))"
    echo "Seeds: $S0 $S1"
    echo "============================================================"

    run_one 0 0-3  mlp_reg  "$S0" &
    P1=$!

    run_one 0 4-7  mlp_rank "$S0" &
    P2=$!

    run_one 1 8-11 mlp_reg  "$S1" &
    P3=$!

    run_one 1 12-15 mlp_rank "$S1" &
    P4=$!

    wait $P1
    E1=$?

    wait $P2
    E2=$?

    wait $P3
    E3=$?

    wait $P4
    E4=$?

    echo "Wave exits: $E1 $E2 $E3 $E4"

done


echo
echo "============================================================"
echo "ALL V3 RUNS COMPLETE"
echo "============================================================"
