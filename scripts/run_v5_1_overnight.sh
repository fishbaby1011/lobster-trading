#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/lobster-trading" || exit 1

PY="$HOME/venvs/lobster/bin/python"

HORIZONS=(10 20)

FEATURE_SETS=(
  all
  no_absolute_returns
  no_relative
  no_market_context
  no_volatility
  no_trend
)

TRAIN_WINDOWS=(
  expanding
  3
  5
  8
)

SEEDS=(
  1
  7
  42
  123
  2026
  3407
  8765
  9999
  31415
  27182
  16180
  42424
  13579
  24680
  55555
  77777
)

read -r -a GPUS <<< "${GPU_LIST:-0 1}"

JOBS_PER_GPU="${JOBS_PER_GPU:-2}"

LOG_DIR="runs/v5_1_launcher_logs"

mkdir -p "$LOG_DIR"

JOBS_RAW="$LOG_DIR/jobs_raw.tsv"
JOBS="$LOG_DIR/jobs.tsv"
STATUS="$LOG_DIR/status.csv"
LOCK="$LOG_DIR/status.lock"

: > "$JOBS_RAW"

printf 'name,gpu,status,action,seconds\n' > "$STATUS"


# ============================================================
# BUILD EXPERIMENT MATRIX
# ============================================================

for H in "${HORIZONS[@]}"; do

  for FS in "${FEATURE_SETS[@]}"; do

    for TW in "${TRAIN_WINDOWS[@]}"; do

      for SEED in "${SEEDS[@]}"; do

        printf '%s\t%s\t%s\t%s\n' \
          "$H" \
          "$FS" \
          "$TW" \
          "$SEED" \
          >> "$JOBS_RAW"

      done

    done

  done

done


# ============================================================
# DETERMINISTIC SHUFFLE
#
# 讓不同種類的實驗平均散到兩張 GPU，
# 而不是一張卡長時間只跑同一種設定。
# ============================================================

"$PY" - "$JOBS_RAW" "$JOBS" <<'PY'
from pathlib import Path
import random
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

jobs = [
    line
    for line in src.read_text().splitlines()
    if line.strip()
]

random.Random(20260829).shuffle(jobs)

dst.write_text(
    "\n".join(jobs) + "\n"
)

print(
    f"Deterministic shuffled jobs: {len(jobs)}"
)
PY


TOTAL_JOBS=$(wc -l < "$JOBS")

TOTAL_WORKERS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))


echo "============================================================"
echo "LOBSTER V5.1 OVERNIGHT ROBUSTNESS GAUNTLET"
echo "============================================================"
echo "GPUs        : ${GPUS[*]}"
echo "Jobs/GPU    : $JOBS_PER_GPU"
echo "Workers     : $TOTAL_WORKERS"
echo "Experiments : $TOTAL_JOBS"
echo "Start       : $(date -Iseconds)"
echo


# ============================================================
# THREAD-SAFE STATUS WRITER
# ============================================================

record_status() {

  local name="$1"
  local gpu="$2"
  local status="$3"
  local action="$4"
  local seconds="$5"

  {
    flock -x 200

    printf '%s,%s,%s,%s,%s\n' \
      "$name" \
      "$gpu" \
      "$status" \
      "$action" \
      "$seconds" \
      >> "$STATUS"

  } 200>"$LOCK"

}


# ============================================================
# RESUME SUPPORT
#
# 如果某個 experiment 已經完成 9 個 OOS years，
# 就不要再重跑。
# ============================================================

already_complete() {

  local name="$1"
  local f

  for f in runs/"${name}"_*/summary.json; do

    [ -f "$f" ] || continue

    if "$PY" - "$f" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as fh:
    data = json.load(fh)

years = int(
    data.get(
        "years",
        0,
    )
)

raise SystemExit(
    0 if years == 9 else 1
)
PY
    then
      return 0
    fi

  done

  return 1
}


# ============================================================
# WORKER
# ============================================================

run_worker() {

  local worker_id="$1"
  local gpu="$2"

  local index=0

  local h
  local fs
  local tw
  local seed

  local name
  local log
  local status

  local started
  local ended
  local elapsed


  echo "Worker $worker_id -> GPU $gpu"


  while IFS=$'\t' read -r h fs tw seed; do

    if (( index % TOTAL_WORKERS != worker_id )); then

      index=$((index + 1))

      continue

    fi


    name="v5_1_h${h}_fs${fs}_tw${tw}_seed${seed}"

    log="$LOG_DIR/${name}.log"


    if already_complete "$name"; then

      echo "$(date -Iseconds) SKIP $name"

      record_status \
        "$name" \
        "$gpu" \
        0 \
        skip \
        0

      index=$((index + 1))

      continue

    fi


    echo "$(date -Iseconds) START $name GPU=$gpu"


    started=$(date +%s)


    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    "$PY" \
      src/train_cross_section_v5_1.py \
      --horizon "$h" \
      --seed "$seed" \
      --feature-set "$fs" \
      --train-window "$tw" \
      --epochs 200 \
      --patience 8 \
      --eval-every 5 \
      --rank-day-batch 128 \
      --num-workers 0 \
      --run-name "$name" \
      > "$log" 2>&1


    status=$?


    ended=$(date +%s)

    elapsed=$((ended - started))


    echo \
      "$(date -Iseconds) END $name status=$status seconds=$elapsed"


    record_status \
      "$name" \
      "$gpu" \
      "$status" \
      run \
      "$elapsed"


    index=$((index + 1))

  done < "$JOBS"

}


# ============================================================
# START WORKERS
# ============================================================

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
# WAIT
# ============================================================

WORKER_FAILED=0


for PID in "${PIDS[@]}"; do

  if ! wait "$PID"; then

    WORKER_FAILED=1

  fi

done


# ============================================================
# VALIDATE TRAINING
# ============================================================

DONE=$(
  awk '
    NR > 1 { n++ }
    END { print n + 0 }
  ' "$STATUS"
)


BAD=$(
  awk -F, '
    NR > 1 && $3 != 0 { n++ }
    END { print n + 0 }
  ' "$STATUS"
)


echo
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Completed : $DONE / $TOTAL_JOBS"
echo "Failed    : $BAD"
echo "Time      : $(date -Iseconds)"


if (( DONE != TOTAL_JOBS )); then

  echo "ERROR: incomplete experiment count"

  exit 1

fi


if (( BAD != 0 || WORKER_FAILED != 0 )); then

  echo "ERROR: one or more jobs failed"

  exit 1

fi


# ============================================================
# AGGREGATION
# ============================================================

echo
echo "============================================================"
echo "AGGREGATING ROBUSTNESS RESULTS"
echo "============================================================"


"$PY" \
  src/aggregate_v5_1.py \
  --expected "$TOTAL_JOBS" \
  2>&1 \
  | tee runs/v5_1_aggregate.log


AGG_STATUS=${PIPESTATUS[0]}


if (( AGG_STATUS != 0 )); then

  echo "ERROR: aggregate_v5_1.py failed"

  exit "$AGG_STATUS"

fi


# ============================================================
# PORTFOLIO / REAL EXECUTION ANALYSIS
# ============================================================

echo
echo "============================================================"
echo "RUNNING REAL-EXECUTION PORTFOLIO CHECK"
echo "============================================================"


"$PY" \
  src/analyze_v5_1_portfolio.py \
  --random-trials 10000 \
  --bootstrap-trials 20000 \
  2>&1 \
  | tee runs/v5_1_portfolio_analysis.log


PORT_STATUS=${PIPESTATUS[0]}


if (( PORT_STATUS != 0 )); then

  echo "ERROR: portfolio analysis failed"

  exit "$PORT_STATUS"

fi


echo
echo "============================================================"
echo "LOBSTER V5.1 OVERNIGHT COMPLETE"
echo "============================================================"
echo "End: $(date -Iseconds)"
echo
echo "Results:"
echo "  results/v5_1_overnight_summary.txt"
echo "  results/v5_1_portfolio_summary.txt"
echo "  results/v5_1_primary_random.csv"
echo "  results/v5_1_primary_ic_bootstrap.csv"
