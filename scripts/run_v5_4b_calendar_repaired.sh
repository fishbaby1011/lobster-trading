#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/lobster-trading" || exit 1

PY="$HOME/venvs/lobster/bin/python"

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

GPUS=(0 1)

JOBS_PER_GPU=2

WORKERS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

LOG_DIR="runs/v5_4b_launcher_logs"

mkdir -p "$LOG_DIR"


is_complete() {

  local seed="$1"

  "$PY" - "$seed" <<'PY'
from pathlib import Path
import json
import sys

seed = int(sys.argv[1])

pattern = (
    f"v5_4b_repaired_h20_seed{seed}_*"
)

for run in sorted(
    Path("runs").glob(pattern),
    reverse=True,
):
    f = run / "summary.json"

    if not f.exists():
        continue

    try:
        x = json.loads(
            f.read_text()
        )
    except Exception:
        continue

    if int(
        x.get("years", 0)
    ) == 9:
        raise SystemExit(0)

raise SystemExit(1)
PY
}


run_worker() {

  local worker="$1"
  local gpu="$2"

  for ((
    i=worker;
    i<${#SEEDS[@]};
    i+=WORKERS
  )); do

    seed="${SEEDS[$i]}"

    if is_complete "$seed"; then

      echo \
        "[SKIP] worker=$worker gpu=$gpu seed=$seed"

      continue
    fi

    name="v5_4b_repaired_h20_seed${seed}"

    log="${LOG_DIR}/seed${seed}.log"

    echo \
      "[START] worker=$worker gpu=$gpu seed=$seed"

    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    "$PY" \
      src/train_cross_section_v5_1.py \
      --horizon 20 \
      --seed "$seed" \
      --feature-set all \
      --train-window expanding \
      --epochs 200 \
      --patience 8 \
      --eval-every 5 \
      --rank-day-batch 128 \
      --num-workers 0 \
      --run-name "$name" \
      > "$log" 2>&1

    rc=$?

    echo \
      "[END] worker=$worker gpu=$gpu seed=$seed rc=$rc"

    if [ "$rc" -ne 0 ]; then
      return "$rc"
    fi

  done
}


PIDS=()

worker=0

for gpu in "${GPUS[@]}"; do

  for ((slot=0; slot<JOBS_PER_GPU; slot++)); do

    run_worker \
      "$worker" \
      "$gpu" &

    PIDS+=("$!")

    worker=$((worker + 1))

  done

done


BAD=0

for pid in "${PIDS[@]}"; do

  if ! wait "$pid"; then
    BAD=$((BAD + 1))
  fi

done


if [ "$BAD" -ne 0 ]; then

  echo "V5.4B FAILED workers=$BAD"

  exit 1

fi


echo
echo "============================================"
echo "V5.4B TRAINING COMPLETE"
echo "============================================"


"$PY" - <<'PY'
from pathlib import Path
import json

import pandas as pd


SEEDS = [
    1, 7, 42, 123,
    2026, 3407, 8765, 9999,
    31415, 27182, 16180, 42424,
    13579, 24680, 55555, 77777,
]


rows = []
folds = []


for seed in SEEDS:

    pattern = (
        f"v5_4b_repaired_h20_seed{seed}_*"
    )

    candidates = sorted(
        Path("runs").glob(pattern),
        reverse=True,
    )

    found = None

    for run in candidates:

        sf = run / "summary.json"
        ff = run / "fold_metrics.csv"

        if not (
            sf.exists()
            and ff.exists()
        ):
            continue

        s = json.loads(
            sf.read_text()
        )

        if int(
            s.get("years", 0)
        ) != 9:
            continue

        found = run
        break

    if found is None:

        raise RuntimeError(
            f"missing seed {seed}"
        )


    s = json.loads(
        (
            found
            / "summary.json"
        ).read_text()
    )

    rows.append({
        "seed":
            seed,

        "run":
            found.name,

        **s,
    })


    f = pd.read_csv(
        found
        / "fold_metrics.csv"
    )

    f["run"] = (
        found.name
    )

    folds.append(f)


summary = pd.DataFrame(rows)

folds = pd.concat(
    folds,
    ignore_index=True,
)


summary.to_csv(
    "results/v5_4b_repaired_seed_summary.csv",
    index=False,
)

folds.to_csv(
    "results/v5_4b_repaired_folds.csv",
    index=False,
)


old = pd.read_csv(
    "results/v5_1_robustness.csv"
)

baseline = old[
    (old["horizon"] == 20)
    &
    (old["feature_set"] == "all")
    &
    (
        old["train_window"]
        .astype(str)
        == "expanding"
    )
].iloc[0]


new_ic = (
    summary[
        "mean_test_ic"
    ]
)


yearly = (
    folds.groupby(
        "year",
        as_index=False,
    )
    .agg(
        seed_count=(
            "seed",
            "nunique",
        ),

        test_ic_mean=(
            "test_ic",
            "mean",
        ),

        test_ic_std=(
            "test_ic",
            "std",
        ),

        test_ic_min=(
            "test_ic",
            "min",
        ),

        test_ic_max=(
            "test_ic",
            "max",
        ),
    )
)


yearly.to_csv(
    "results/v5_4b_repaired_yearly.csv",
    index=False,
)


old_ic = float(
    baseline["ic_mean"]
)

new_mean = float(
    new_ic.mean()
)


out = []

out.append(
    "=" * 90
)

out.append(
    "LOBSTER V5.4B REPAIRED-DATA VALIDATION"
)

out.append(
    "=" * 90
)

out.append("")

out.append(
    "Frozen configuration:"
)

out.append(
    "H20 / all features / expanding / 16 seeds"
)

out.append("")

out.append(
    f"Old v5.1 IC mean : {old_ic:.6f}"
)

out.append(
    f"New repaired mean: {new_mean:.6f}"
)

out.append(
    f"Delta            : {new_mean - old_ic:+.6f}"
)

out.append("")

out.append(
    f"New IC std       : {new_ic.std(ddof=1):.6f}"
)

out.append(
    f"New IC min       : {new_ic.min():.6f}"
)

out.append(
    f"New IC max       : {new_ic.max():.6f}"
)

out.append(
    "Positive seeds   : "
    f"{(new_ic > 0).sum()} / {len(new_ic)}"
)

out.append("")

out.append(
    "YEARLY OOS IC"
)

out.append(
    yearly.to_string(
        index=False
    )
)


text = "\n".join(out)

Path(
    "results/v5_4b_repaired_compare.txt"
).write_text(
    text + "\n"
)

print(text)
PY


echo
echo "============================================"
echo "V5.4B COMPLETE"
echo "============================================"
