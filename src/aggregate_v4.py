from pathlib import Path
import json

import pandas as pd


rows = []


for run in sorted(
    Path("runs").glob(
        "v4_h*_mlp_rank_seed*_*"
    )
):

    config_file = (
        run / "config.json"
    )

    summary_file = (
        run / "summary.csv"
    )


    if not (
        config_file.exists()
        and summary_file.exists()
    ):
        continue


    with config_file.open() as f:
        config = json.load(f)


    df = pd.read_csv(
        summary_file
    )


    df["horizon"] = (
        config["horizon"]
    )

    df["run"] = run.name

    rows.append(df)


if not rows:
    raise RuntimeError(
        "No v4 results found"
    )


all_runs = pd.concat(
    rows,
    ignore_index=True,
)


all_runs.to_csv(
    "results/v4_all_runs.csv",
    index=False,
)


leaderboard = (
    all_runs
    .groupby(
        [
            "horizon",
            "top_k",
        ]
    )
    .agg(
        seeds=(
            "seed",
            "nunique",
        ),

        ic_mean=(
            "mean_ic",
            "mean",
        ),

        ic_median=(
            "mean_ic",
            "median",
        ),

        ic_std=(
            "mean_ic",
            "std",
        ),

        ic_min=(
            "mean_ic",
            "min",
        ),

        ic_max=(
            "mean_ic",
            "max",
        ),

        alpha_mean=(
            "mean_alpha_vs_0050",
            "mean",
        ),

        alpha_median=(
            "mean_alpha_vs_0050",
            "median",
        ),

        active_sharpe_mean=(
            "mean_active_sharpe",
            "mean",
        ),

        turnover_mean=(
            "mean_turnover",
            "mean",
        ),

        positive_years_mean=(
            "positive_alpha_years",
            "mean",
        ),
    )
    .reset_index()
)


leaderboard.to_csv(
    "results/v4_multihorizon_leaderboard.csv",
    index=False,
)


# 每個 run 的 IC 在不同 top-k 是相同的，
# 所以每個 run 只留一行分析 signal decay
signal_base = (
    all_runs
    .sort_values(
        "top_k"
    )
    .drop_duplicates(
        subset=["run"]
    )
)


decay = (
    signal_base
    .groupby("horizon")
    .agg(
        seeds=(
            "seed",
            "nunique",
        ),

        ic_mean=(
            "mean_ic",
            "mean",
        ),

        ic_median=(
            "mean_ic",
            "median",
        ),

        ic_std=(
            "mean_ic",
            "std",
        ),

        ic_min=(
            "mean_ic",
            "min",
        ),

        ic_max=(
            "mean_ic",
            "max",
        ),

        spread_mean=(
            "mean_spread",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "horizon"
    )
)


decay.to_csv(
    "results/v4_signal_decay.csv",
    index=False,
)


print()
print("=" * 110)
print("SIGNAL DECAY")
print("=" * 110)

print(
    decay.to_string(
        index=False
    )
)


print()
print("=" * 110)
print("PORTFOLIO LEADERBOARD")
print("=" * 110)

print(
    leaderboard
    .sort_values(
        [
            "ic_mean",
            "active_sharpe_mean",
        ],
        ascending=False,
    )
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(
    "results/v4_signal_decay.csv"
)
print(
    "results/v4_multihorizon_leaderboard.csv"
)
print(
    "results/v4_all_runs.csv"
)
