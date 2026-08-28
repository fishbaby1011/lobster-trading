from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


p = argparse.ArgumentParser()

p.add_argument(
    "--expected",
    type=int,
    default=768,
)

p.add_argument(
    "--allow-partial",
    action="store_true",
)

args = p.parse_args()


RUNS = Path("runs")
RESULTS = Path("results")

RESULTS.mkdir(
    exist_ok=True
)


latest = {}


for run in sorted(
    RUNS.glob("v5_1_h*_fs*_tw*_seed*_*")
):

    summary_file = (
        run / "summary.json"
    )

    fold_file = (
        run / "fold_metrics.csv"
    )

    config_file = (
        run / "config.json"
    )

    if not (
        summary_file.exists()
        and fold_file.exists()
        and config_file.exists()
    ):
        continue

    try:
        summary = json.loads(
            summary_file.read_text()
        )

        config = json.loads(
            config_file.read_text()
        )

    except Exception:
        continue

    # Overnight runs must contain all 9 OOS years.
    if int(
        summary.get(
            "years",
            0,
        )
    ) != 9:
        continue

    key = (
        int(config["horizon"]),
        str(config["feature_set"]),
        str(config["train_window"]),
        int(config["seed"]),
    )

    latest[key] = (
        run,
        summary,
        config,
    )


if not latest:
    raise RuntimeError(
        "No completed v5.1 overnight runs found"
    )


if (
    len(latest) != args.expected
    and not args.allow_partial
):
    raise RuntimeError(
        f"Expected {args.expected} runs, "
        f"found {len(latest)}"
    )


run_rows = []
fold_parts = []


for (
    h,
    feature_set,
    train_window,
    seed,
), (
    run,
    summary,
    config,
) in latest.items():

    run_rows.append({
        "horizon":
            h,

        "feature_set":
            feature_set,

        "train_window":
            train_window,

        "seed":
            seed,

        "mean_test_ic":
            summary["mean_test_ic"],

        "median_test_ic":
            summary["median_test_ic"],

        "std_test_ic":
            summary["std_test_ic"],

        "mean_test_spread":
            summary["mean_test_spread"],

        "mean_best_epoch":
            summary["mean_best_epoch"],

        "mean_valid_ic":
            summary["mean_valid_ic"],

        "run":
            run.name,

        "git_commit":
            config.get(
                "git_commit"
            ),
    })

    f = pd.read_csv(
        run / "fold_metrics.csv"
    )

    f["run"] = run.name

    fold_parts.append(
        f
    )


runs = pd.DataFrame(
    run_rows
)

folds = pd.concat(
    fold_parts,
    ignore_index=True,
)


runs.to_csv(
    RESULTS
    / "v5_1_all_runs.csv",
    index=False,
)

folds.to_csv(
    RESULTS
    / "v5_1_all_folds.csv",
    index=False,
)


def frac_positive(x):
    x = pd.Series(x).dropna()

    if len(x) == 0:
        return np.nan

    return float(
        (x > 0).mean()
    )


robust = (
    runs
    .groupby(
        [
            "horizon",
            "feature_set",
            "train_window",
        ],
        as_index=False,
    )
    .agg(
        seeds=(
            "seed",
            "nunique",
        ),

        ic_mean=(
            "mean_test_ic",
            "mean",
        ),

        ic_median=(
            "mean_test_ic",
            "median",
        ),

        ic_std=(
            "mean_test_ic",
            "std",
        ),

        ic_min=(
            "mean_test_ic",
            "min",
        ),

        ic_max=(
            "mean_test_ic",
            "max",
        ),

        positive_seed_fraction=(
            "mean_test_ic",
            frac_positive,
        ),

        spread_mean=(
            "mean_test_spread",
            "mean",
        ),

        valid_ic_mean=(
            "mean_valid_ic",
            "mean",
        ),

        best_epoch_mean=(
            "mean_best_epoch",
            "mean",
        ),
    )
)


robust["ic_minus_std"] = (
    robust["ic_mean"]
    - robust["ic_std"]
)


yearly = (
    folds
    .groupby(
        [
            "horizon",
            "feature_set",
            "train_window",
            "year",
        ],
        as_index=False,
    )
    .agg(
        seeds=(
            "seed",
            "nunique",
        ),

        ic_mean=(
            "test_ic",
            "mean",
        ),

        ic_median=(
            "test_ic",
            "median",
        ),

        ic_std=(
            "test_ic",
            "std",
        ),

        spread_mean=(
            "test_spread",
            "mean",
        ),
    )
)


year_pivot = (
    yearly
    .pivot_table(
        index=[
            "horizon",
            "feature_set",
            "train_window",
        ],
        columns="year",
        values="ic_mean",
    )
    .reset_index()
)


year_pivot.columns = [
    (
        f"ic_{int(c)}"
        if isinstance(
            c,
            (int, float, np.integer),
        )
        else c
    )
    for c in year_pivot.columns
]


robust = robust.merge(
    year_pivot,
    on=[
        "horizon",
        "feature_set",
        "train_window",
    ],
    how="left",
)


robust.to_csv(
    RESULTS
    / "v5_1_robustness.csv",
    index=False,
)

yearly.to_csv(
    RESULTS
    / "v5_1_yearly_ic.csv",
    index=False,
)


baseline = (
    robust[
        robust.feature_set
        == "all"
    ][
        [
            "horizon",
            "train_window",
            "ic_mean",
            "ic_std",
        ]
    ]
    .rename(
        columns={
            "ic_mean":
                "all_ic_mean",

            "ic_std":
                "all_ic_std",
        }
    )
)


ablation = robust.merge(
    baseline,
    on=[
        "horizon",
        "train_window",
    ],
    how="left",
)


ablation[
    "delta_ic_vs_all"
] = (
    ablation["ic_mean"]
    - ablation["all_ic_mean"]
)


ablation.to_csv(
    RESULTS
    / "v5_1_ablation.csv",
    index=False,
)


out = []

out.append(
    "=" * 110
)

out.append(
    "LOBSTER V5.1 OVERNIGHT ROBUSTNESS"
)

out.append(
    "=" * 110
)

out.append(
    f"Completed runs: {len(runs)}"
)

out.append("")


all_view = (
    robust[
        robust.feature_set
        == "all"
    ]
    .sort_values(
        [
            "horizon",
            "train_window",
        ]
    )
)


out.append(
    "ALL-FEATURE BASELINE ACROSS TRAIN WINDOWS"
)

out.append(
    all_view[
        [
            "horizon",
            "train_window",
            "seeds",
            "ic_mean",
            "ic_median",
            "ic_std",
            "ic_min",
            "positive_seed_fraction",
            "spread_mean",
            "ic_2024",
            "ic_2025",
            "ic_2026",
        ]
    ].to_string(
        index=False
    )
)

out.append("")


out.append(
    "FEATURE ABLATION DELTA VS ALL"
)

out.append(
    ablation[
        [
            "horizon",
            "train_window",
            "feature_set",
            "ic_mean",
            "delta_ic_vs_all",
            "ic_std",
            "positive_seed_fraction",
        ]
    ]
    .sort_values(
        [
            "horizon",
            "train_window",
            "delta_ic_vs_all",
        ]
    )
    .to_string(
        index=False
    )
)

out.append("")


out.append(
    "TOP CONFIGS BY MEAN IC "
    "(DIAGNOSTIC ONLY — DO NOT SELECT LIVE MODEL FROM THIS TABLE)"
)

out.append(
    robust[
        [
            "horizon",
            "feature_set",
            "train_window",
            "ic_mean",
            "ic_std",
            "ic_min",
            "ic_minus_std",
            "spread_mean",
            "ic_2024",
        ]
    ]
    .sort_values(
        "ic_mean",
        ascending=False,
    )
    .head(20)
    .to_string(
        index=False
    )
)


text = "\n".join(
    out
)


(
    RESULTS
    / "v5_1_overnight_summary.txt"
).write_text(
    text
    + "\n"
)


print(text)
