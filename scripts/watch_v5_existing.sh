#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/lobster-trading" || exit 1

PY="$HOME/venvs/lobster/bin/python"
STATUS="runs/v5_launcher_logs/status.csv"
TOTAL=16

echo "============================================================"
echo "LOBSTER V5 WATCHER"
echo "============================================================"
echo "Start: $(date -Iseconds)"
echo

while true; do

    DONE=$(awk '
        NR > 1 { n++ }
        END { print n + 0 }
    ' "$STATUS" 2>/dev/null)

    BAD=$(awk -F, '
        NR > 1 && $3 != 0 { n++ }
        END { print n + 0 }
    ' "$STATUS" 2>/dev/null)

    ACTIVE=$(pgrep -fc '[t]rain_cross_section_v5.py' || true)

    echo "$(date -Iseconds) active=$ACTIVE completed=$DONE/$TOTAL failed=$BAD"

    if (( BAD > 0 )); then
        echo
        echo "ERROR: failed V5 experiment detected."
        cat "$STATUS"
        exit 1
    fi

    if (( DONE >= TOTAL )); then
        echo
        echo "All $TOTAL training experiments reported completion."
        break
    fi

    if (( ACTIVE == 0 )); then

        echo "No Python worker visible; waiting 20s to rule out job transition..."
        sleep 20

        ACTIVE2=$(pgrep -fc '[t]rain_cross_section_v5.py' || true)

        DONE2=$(awk '
            NR > 1 { n++ }
            END { print n + 0 }
        ' "$STATUS" 2>/dev/null)

        if (( ACTIVE2 == 0 && DONE2 < TOTAL )); then
            echo
            echo "ERROR: workers stopped before all experiments completed."
            echo "Completed: $DONE2/$TOTAL"
            echo
            cat "$STATUS"
            exit 2
        fi
    fi

    sleep 30
done


echo
echo "===== FINAL TRAINING STATUS ====="
cat "$STATUS"

BAD=$(awk -F, '
    NR > 1 && $3 != 0 { n++ }
    END { print n + 0 }
' "$STATUS")

if (( BAD != 0 )); then
    echo "ERROR: refusing analysis because training jobs failed."
    exit 3
fi


echo
echo "============================================================"
echo "STARTING V5 REALITY-CHECK ANALYSIS"
echo "============================================================"
echo "Time: $(date -Iseconds)"

"$PY" \
    src/analyze_v5.py \
    --random-trials 1000 \
    2>&1 | tee runs/v5_final_summary.log

ANALYZE_STATUS=${PIPESTATUS[0]}

if (( ANALYZE_STATUS != 0 )); then
    echo
    echo "ERROR: analyze_v5.py failed with status=$ANALYZE_STATUS"
    exit "$ANALYZE_STATUS"
fi


echo
echo "============================================================"
echo "LOBSTER V5 ALL DONE"
echo "============================================================"
echo "End: $(date -Iseconds)"
echo
echo "Results:"
echo "  results/v5_fold_metrics.csv"
echo "  results/v5_portfolio_leaderboard.csv"
echo "  results/v5_robust_10bps.csv"
echo "  results/v5_random_baseline.csv"
echo "  runs/v5_final_summary.log"
