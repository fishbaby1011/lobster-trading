#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/lobster-trading" || exit 1

PY="$HOME/venvs/lobster/bin/python"

HORIZONS=(10 20)
SEEDS=(1 7 42 123 2026 3407 8765 9999)

read -r -a GPUS <<< "${GPU_LIST:-0 1}"

JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

LOG_DIR="runs/v5_launcher_logs"

mkdir -p "$LOG_DIR"

JOB_FILE="$LOG_DIR/jobs.txt"
STATUS_FILE="$LOG_DIR/status.csv"

: > "$JOB_FILE"

printf 'name,gpu,status\n' \
  > "$STATUS_FILE"


for H in "${HORIZONS[@]}"; do

  for S in "${SEEDS[@]}"; do

    printf '%s %s\n' \
      "$H" \
      "$S" \
      >> "$JOB_FILE"

  done

done


TOTAL_JOBS=$(wc -l < "$JOB_FILE")

TOTAL_WORKERS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))


echo "============================================================"
echo "LOBSTER V5 REFIT + OOS"
echo "============================================================"

echo "GPUs        : ${GPUS[*]}"
echo "Jobs/GPU    : $JOBS_PER_GPU"
echo "Workers     : $TOTAL_WORKERS"
echo "Experiments : $TOTAL_JOBS"
echo "Start       : $(date -Iseconds)"
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


    name="v5_h${h}_seed${s}"

    log="$LOG_DIR/${name}.log"


    echo \
      "$(date -Iseconds) START $name GPU=$gpu"


    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    "$PY" \
      src/train_cross_section_v5.py \
      --horizon "$h" \
      --seed "$s" \
      --epochs 200 \
      --patience 8 \
      --eval-every 5 \
      --rank-day-batch 128 \
      --num-workers 0 \
      --run-name "$name" \
      > "$log" 2>&1


    status=$?


    echo \
      "$(date -Iseconds) END $name status=$status"


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


# ============================================================
# WAIT FOR ALL WORKERS
# ============================================================

FAILED=0

for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done

echo
echo "============================================================"
echo "V5 TRAINING COMPLETE"
echo "============================================================"
echo "End: $(date -Iseconds)"

cat "$STATUS_FILE"

DONE=$(awk 'NR>1 {n++} END {print n+0}' "$STATUS_FILE")
BAD=$(awk -F, 'NR>1 && $3 != 0 {n++} END {print n+0}' "$STATUS_FILE")

echo
echo "Completed: $DONE / $TOTAL_JOBS"
echo "Failed   : $BAD"

if (( DONE != TOTAL_JOBS )); then
    echo "ERROR: incomplete experiment set"
    exit 1
fi

if (( BAD != 0 || FAILED != 0 )); then
    echo "ERROR: one or more experiments failed"
    exit 1
fi

echo "All V5 experiments completed successfully."
