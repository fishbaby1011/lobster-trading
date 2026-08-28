#!/usr/bin/env bash

set -u

cd ~/lobster-trading

PY="$HOME/venvs/lobster/bin/python"

HORIZONS=(1 3 5 10 20)
SEEDS=(1 7 42 123 2026 3407 8765 9999)

# 現在：
# GPU_LIST="0 1"
#
# 未來 8×V100：
# GPU_LIST="0 1 2 3 4 5 6 7"

read -ra GPUS <<< "${GPU_LIST:-0 1}"

JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

LOG_DIR="runs/v4_launcher_logs"

mkdir -p "$LOG_DIR"

JOB_FILE="$LOG_DIR/jobs.txt"

: > "$JOB_FILE"


for H in "${HORIZONS[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo "$H $S" >> "$JOB_FILE"
    done
done


TOTAL_JOBS=$(wc -l < "$JOB_FILE")

TOTAL_WORKERS=$(
    (
        ${#GPUS[@]}
        * JOBS_PER_GPU
    )
)


echo "============================================================"
echo "LOBSTER V4 MULTI-HORIZON"
echo "============================================================"

echo "GPUs           : ${GPUS[*]}"
echo "Jobs/GPU       : $JOBS_PER_GPU"
echo "Workers        : $TOTAL_WORKERS"
echo "Experiments    : $TOTAL_JOBS"
echo "Start          : $(date -Iseconds)"
echo


run_worker() {

    WORKER_ID=$1
    GPU=$2

    echo \
      "Worker $WORKER_ID -> GPU $GPU"


    INDEX=0

    while read -r H S; do

        if (
            (
                INDEX
                % TOTAL_WORKERS
            )
            != WORKER_ID
        ); then
            INDEX=$((INDEX + 1))
            continue
        fi


        NAME="v4_h${H}_mlp_rank_seed${S}"

        LOG="$LOG_DIR/${NAME}.log"


        echo \
          "$(date -Iseconds) START $NAME GPU=$GPU"


        CUDA_VISIBLE_DEVICES="$GPU" \
        OMP_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PY" \
          src/train_cross_section_v4.py \
          --model mlp_rank \
          --horizon "$H" \
          --seed "$S" \
          --lr 0.001 \
          --weight-decay 0.0001 \
          --dropout 0.10 \
          --hidden 1024,512,256 \
          --clip-q 0.01 \
          --epochs 200 \
          --patience 8 \
          --eval-every 5 \
          --rank-day-batch 128 \
          --top-k 3,5,10,15,20 \
          --run-name "$NAME" \
          > "$LOG" 2>&1


        STATUS=$?


        echo \
          "$(date -Iseconds) END $NAME status=$STATUS"


        echo \
          "$NAME,$GPU,$STATUS" \
          >> "$LOG_DIR/status.csv"


        INDEX=$((INDEX + 1))

    done < "$JOB_FILE"
}


echo \
  "name,gpu,status" \
  > "$LOG_DIR/status.csv"


PIDS=()

WORKER_ID=0


for GPU in "${GPUS[@]}"; do

    for (
        SLOT=0;
        SLOT<JOBS_PER_GPU;
        SLOT++
    ); do

        run_worker \
          "$WORKER_ID" \
          "$GPU" &

        PIDS+=("$!")

        WORKER_ID=$(
            (
                WORKER_ID
                + 1
            )
        )

    done

done


FAILED=0


for PID in "${PIDS[@]}"; do

    wait "$PID"

    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        FAILED=1
    fi

done


echo
echo "============================================================"
echo "V4 COMPLETE"
echo "============================================================"

echo "End: $(date -Iseconds)"

echo
echo "Job status:"
cat "$LOG_DIR/status.csv"


if grep -q ',[1-9][0-9]*$' \
    "$LOG_DIR/status.csv"; then

    echo
    echo "WARNING: one or more jobs failed."

    exit 1

fi


echo
echo "All jobs completed successfully."
