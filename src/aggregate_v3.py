from pathlib import Path
import json

import numpy as np
import pandas as pd


runs = sorted(
    Path("runs").glob(
        "v3_*_seed*_*"
    )
)


summary_frames = []
prediction_runs = []


for run in runs:

    summary_file = (
        run / "summary.csv"
    )

    config_file = (
        run / "config.json"
    )

    pred_file = (
        run / "predictions.parquet"
    )


    if not (
        summary_file.exists()
        and config_file.exists()
    ):
        continue


    with config_file.open() as f:
        config = json.load(f)


    summary = pd.read_csv(
        summary_file
    )


    summary["run"] = run.name
    summary["clip_q"] = config["clip_q"]
    summary["hidden"] = str(
        config["hidden"]
    )
    summary["lr"] = config["lr"]

    summary_frames.append(
        summary
    )


    if pred_file.exists():
        prediction_runs.append(
            (
                run,
                config,
                pred_file,
            )
        )


all_summary = pd.concat(
    summary_frames,
    ignore_index=True,
)


print()
print("=" * 120)
print("ALL RUNS")
print("=" * 120)

print(
    all_summary[
        [
            "model",
            "seed",
            "top_k",
            "mean_ic",
            "positive_alpha_years",
            "mean_alpha_vs_0050",
            "median_alpha_vs_0050",
            "mean_active_sharpe",
            "mean_turnover",
        ]
    ]
    .sort_values(
        [
            "model",
            "top_k",
            "seed",
        ]
    )
    .to_string(
        index=False
    )
)


grouped = (
    all_summary
    .groupby(
        [
            "model",
            "top_k",
        ]
    )
    .agg(
        seeds=("seed", "nunique"),

        ic_mean=("mean_ic", "mean"),
        ic_median=("mean_ic", "median"),
        ic_std=("mean_ic", "std"),
        ic_min=("mean_ic", "min"),
        ic_max=("mean_ic", "max"),

        alpha_mean=(
            "mean_alpha_vs_0050",
            "mean",
        ),

        alpha_median=(
            "mean_alpha_vs_0050",
            "median",
        ),

        alpha_std=(
            "mean_alpha_vs_0050",
            "std",
        ),

        sharpe_mean=(
            "mean_active_sharpe",
            "mean",
        ),

        positive_years_mean=(
            "positive_alpha_years",
            "mean",
        ),

        turnover_mean=(
            "mean_turnover",
            "mean",
        ),
    )
    .reset_index()
)


print()
print("=" * 120)
print("ROBUSTNESS LEADERBOARD")
print("=" * 120)

print(
    grouped
    .sort_values(
        [
            "ic_mean",
            "alpha_median",
        ],
        ascending=False,
    )
    .to_string(
        index=False
    )
)


Path("results").mkdir(
    exist_ok=True
)


all_summary.to_csv(
    "results/v3_all_runs.csv",
    index=False,
)


grouped.to_csv(
    "results/v3_robustness_leaderboard.csv",
    index=False,
)


# ============================================================
# TOP-3 SEED STABILITY
# ============================================================

records = []


for run, config, pred_file in (
    prediction_runs
):

    pred = pd.read_parquet(
        pred_file
    )

    seed = config["seed"]
    model = config["model"]

    for date, day in pred.groupby(
        pred.index
    ):

        top = tuple(
            sorted(
                day.nlargest(
                    3,
                    "pred_alpha"
                )[
                    "ticker"
                ].tolist()
            )
        )

        records.append({
            "model": model,
            "seed": seed,
            "date": date,
            "top3": top,
        })


selection = pd.DataFrame(
    records
)


stability_rows = []


for model, group in (
    selection.groupby("model")
):

    pivot = {}

    for seed, seed_df in (
        group.groupby("seed")
    ):

        pivot[seed] = {
            row.date:
                set(row.top3)

            for row
            in seed_df.itertuples()
        }


    seeds = sorted(
        pivot.keys()
    )


    scores = []


    for i in range(
        len(seeds)
    ):
        for j in range(
            i + 1,
            len(seeds)
        ):

            a = pivot[
                seeds[i]
            ]

            b = pivot[
                seeds[j]
            ]

            dates = (
                set(a.keys())
                & set(b.keys())
            )


            for date in dates:

                union = (
                    a[date]
                    | b[date]
                )

                inter = (
                    a[date]
                    & b[date]
                )

                if union:
                    scores.append(
                        len(inter)
                        / len(union)
                    )


    stability_rows.append({
        "model": model,
        "seed_pairs":
            len(seeds)
            * (len(seeds) - 1)
            // 2,

        "mean_top3_jaccard":
            np.mean(scores)
            if scores
            else np.nan,

        "median_top3_jaccard":
            np.median(scores)
            if scores
            else np.nan,
    })


stability = pd.DataFrame(
    stability_rows
)


print()
print("=" * 120)
print("TOP-3 SEED STABILITY")
print("=" * 120)

print(
    stability.to_string(
        index=False
    )
)


stability.to_csv(
    "results/v3_top3_seed_stability.csv",
    index=False,
)


print()
print("Saved:")
print("results/v3_all_runs.csv")
print("results/v3_robustness_leaderboard.csv")
print("results/v3_top3_seed_stability.csv")
