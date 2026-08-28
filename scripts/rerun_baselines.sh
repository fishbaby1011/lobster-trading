#!/usr/bin/env bash

set -u

cd "$(git rev-parse --show-toplevel)"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_DIR="runs/baseline_compare_${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/run.log"

mkdir -p "$RUN_DIR"

# 從這裡開始，stdout + stderr 全部同時顯示並寫進 log
exec > >(tee "$LOG_FILE") 2>&1

echo "======================================================================"
echo "LOBSTER TRADING BASELINE RERUN"
echo "======================================================================"
echo
echo "Start time: $(date -Iseconds)"
echo "Run dir:    $RUN_DIR"
echo

echo "======================================================================"
echo "SYSTEM"
echo "======================================================================"

uname -a
echo

echo "--- Python ---"
python --version

echo
echo "--- Git ---"
git rev-parse --short HEAD 2>/dev/null || true
git status --short

echo
echo "--- NVIDIA ---"
nvidia-smi

echo
echo "--- Disk ---"
df -h /

echo
echo "--- Memory ---"
free -h


# ---------------------------------------------------------------------
# 保存 Python 套件版本
# ---------------------------------------------------------------------

pip freeze > "${RUN_DIR}/pip-freeze.txt"


# =====================================================================
# BASELINE V1
# =====================================================================

echo
echo
echo "######################################################################"
echo "# BASELINE V1"
echo "######################################################################"
echo

/usr/bin/time -v python src/baseline.py

V1_EXIT=$?

echo
echo "Baseline V1 exit code: $V1_EXIT"

if [ -f results/baseline_backtest.csv ]; then
    cp results/baseline_backtest.csv \
       "${RUN_DIR}/baseline_v1_backtest.csv"
fi

if [ -f data/2330_dataset.parquet ]; then
    cp data/2330_dataset.parquet \
       "${RUN_DIR}/baseline_v1_dataset.parquet"
fi


# =====================================================================
# BASELINE V2
# =====================================================================

echo
echo
echo "######################################################################"
echo "# BASELINE V2"
echo "######################################################################"
echo

/usr/bin/time -v python src/baseline_v2.py

V2_EXIT=$?

echo
echo "Baseline V2 exit code: $V2_EXIT"


# ---------------------------------------------------------------------
# 保存 V2 結果快照
# ---------------------------------------------------------------------

for FILE in \
    results/threshold_scan.csv \
    results/test_predictions_v2.csv \
    results/test_backtest_v2.csv \
    results/walk_forward.csv \
    results/equity_curve_v2.png
do
    if [ -f "$FILE" ]; then
        cp "$FILE" "$RUN_DIR/"
    fi
done


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

cat > "${RUN_DIR}/metadata.txt" <<META
run_id=baseline_compare_${TIMESTAMP}
timestamp=${TIMESTAMP}
git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
python=$(python --version 2>&1)
baseline_v1_exit=${V1_EXIT}
baseline_v2_exit=${V2_EXIT}
META


echo
echo
echo "======================================================================"
echo "RUN COMPLETE"
echo "======================================================================"
echo
echo "End time: $(date -Iseconds)"
echo
echo "Saved files:"
find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n' | sort

echo
echo "Full run directory:"
echo "$RUN_DIR"

echo
echo "Full log:"
echo "$LOG_FILE"
