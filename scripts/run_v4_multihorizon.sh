#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/lobster-trading" || exit 1

PY="$HOME/venvs/lobster/bin/python"

HORIZONS=(1 3 5 10 20)
SEEDS=(1 7 42 123 2026 3407 8765 9999)

read -r -a GPUS <<< "${GPU_LIST:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

LOG_DIR="runs/v4_launcher_logs"
mkdir -p "$LOG_DIR"

JOB_FILE="$LOG_DIR/jobs.txt"
STATUS_FILE="$LOG_DIR/status.csv"

: > "$JOB_FILE"
printf 'name,gpu,status\n' > "$STATUS_FILE"

for H in "${HORIZONS[@]}"; do
    for S in "${SEEDS[@]}"; do
        printf '%s %s\n' "$H" "$S" >> "$JOB_FILE"
    done
done

TOTAL_JOBS=$(wc -l < "$JOB_FILE")
TOTAL_WORKERS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

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
    local worker_id="$1"
    local gpu="$2"

    local index=0
    local h s name log status

    echo "Worker $worker_id -> GPU $gpu"

    while read -r h s; do

        if (( index % TOTAL_WORKERS != worker_id )); then
            index=$((index + 1))
            continue
        fi

        name="v4_h${h}_mlp_rank_seed${s}"
        log="$LOG_DIR/${name}.log"

        echo "$(date -Iseconds) START $name GPU=$gpu"

        CUDA_VISIBLE_DEVICES="$gpu" \
        OMP_NUM_THREADS=2 \
        MKL_NUM_THREADS=2 \
        OPENBLAS_NUM_THREADS=2 \
        "$PY" src/train_cross_section_v4.py \
            --model mlp_rank \
            --horizon "$h" \
            --seed "$s" \
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
            --run-name "$name" \
            > "$log" 2>&1

        status=$?

        echo "$(date -Iseconds) END $name status=$status"

        printf '%s,%s,%s\n' \
            "$name" \
            "$gpu" \
            "$status" \
            >> "$STATUS_FILE"

        index=$((index + 1))
    done < "$JOB_FILE"
}


PIDS=()
WORKER_ID=0

for GPU in "${GPUS[@]}"; do

    for ((SLOT=0; SLOT<JOBS_PER_GPU; SLOT++)); do

        run_worker \
            "$WORKER_ID" \
            "$GPU" &

        PIDS+=("$!")

        WORKER_ID=$((WORKER_ID + 1))
    done

done


FAILED=0

for PID in "${PIDS[@]}"; do

    if ! wait "$PID"; then
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

cat "$STATUS_FILE"


if awk -F, \
    'NR>1 && $3 != 0 {bad=1} END {exit !bad}' \
    "$STATUS_FILE"
then
    echo
    echo "WARNING: one or more jobs failed."
    exit 1
fi


if (( FAILED != 0 )); then
    echo
    echo "WARNING: one or more workers exited unexpectedly."
    exit 1
fi


echo
echo "All jobs completed successfully."
