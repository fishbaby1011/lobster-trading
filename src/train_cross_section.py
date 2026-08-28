from pathlib import Path
from datetime import datetime
import argparse
import json
import logging
import math
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=[
        "ridge",
        "xgb",
    ],
    default="xgb",
)

parser.add_argument(
    "--device",
    choices=[
        "cpu",
        "cuda",
    ],
    default="cpu",
)

parser.add_argument(
    "--top-k",
    type=int,
    default=5,
)

parser.add_argument(
    "--start-year",
    type=int,
    default=2018,
)

parser.add_argument(
    "--end-year",
    type=int,
    default=2026,
)

parser.add_argument(
    "--run-name",
    default=None,
)

args = parser.parse_args()


# ============================================================
# CONSTANTS
# ============================================================

BUY_FEE = 0.001425
SELL_FEE = 0.001425
STOCK_SELL_TAX = 0.003

ROUND_TRIP_COST = (
    BUY_FEE
    + SELL_FEE
    + STOCK_SELL_TAX
)


# ============================================================
# LOAD METADATA
# ============================================================

with open(
    "data/universe_metadata.json"
) as f:
    metadata = json.load(f)


FEATURES = (
    metadata["features"]
)

HORIZON = (
    metadata["horizon"]
)


# ============================================================
# RUN DIR
# ============================================================

timestamp = (
    datetime.now()
    .strftime(
        "%Y%m%d_%H%M%S"
    )
)


name = (
    args.run_name
    or f"{args.model}_{args.device}"
)


RUN_ID = (
    f"{name}_{timestamp}"
)


RUN_DIR = (
    Path("runs")
    / RUN_ID
)


RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "cross_section"
)

logger.setLevel(
    logging.INFO
)


formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)


console = (
    logging.StreamHandler()
)

console.setFormatter(
    formatter
)


file_handler = (
    logging.FileHandler(
        RUN_DIR
        / "run.log"
    )
)

file_handler.setFormatter(
    formatter
)


logger.addHandler(
    console
)

logger.addHandler(
    file_handler
)


# ============================================================
# SYSTEM INFO
# ============================================================

def shell(cmd):
    try:
        return (
            subprocess.check_output(
                cmd,
                shell=True,
                text=True,
                stderr=subprocess.STDOUT,
            )
        )
    except Exception as exc:
        return str(exc)


config = {
    "run_id":
        RUN_ID,

    "model":
        args.model,

    "device":
        args.device,

    "top_k":
        args.top_k,

    "start_year":
        args.start_year,

    "end_year":
        args.end_year,

    "features":
        FEATURES,

    "horizon":
        HORIZON,

    "git_commit":
        shell(
            "git rev-parse HEAD"
        ).strip(),

    "python":
        shell(
            "python --version"
        ).strip(),

    "nvidia_smi":
        shell(
            "nvidia-smi -L"
        ).strip(),
}


with open(
    RUN_DIR / "config.json",
    "w"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    "data/universe_dataset.parquet"
)


df.index = (
    pd.to_datetime(
        df.index
    )
)


df["label_end_date"] = (
    pd.to_datetime(
        df["label_end_date"]
    )
)


logger.info(
    f"Dataset rows: {len(df):,}"
)

logger.info(
    f"Stocks: {df['ticker'].nunique()}"
)

logger.info(
    f"Dates: "
    f"{df.index.min()} "
    f"-> "
    f"{df.index.max()}"
)


# ============================================================
# MODEL
# ============================================================

def make_model():

    if args.model == "ridge":

        return Pipeline([
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                Ridge(
                    alpha=10.0
                )
            ),
        ])


    if args.model == "xgb":

        device = (
            "cuda"
            if args.device == "cuda"
            else "cpu"
        )


        return XGBRegressor(
            n_estimators=700,

            max_depth=4,

            learning_rate=0.03,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="reg:squarederror",

            tree_method="hist",

            device=device,

            random_state=42,

            n_jobs=8,
        )


    raise ValueError(
        args.model
    )


# ============================================================
# METRICS
# ============================================================

def max_drawdown(
    equity
):
    peak = (
        equity.cummax()
    )


    dd = (
        equity / peak
        - 1
    )


    return (
        dd.min()
    )


def daily_ic(
    predictions
):
    values = []


    for date, group in (
        predictions
        .groupby(
            predictions.index
        )
    ):

        if len(group) < 5:
            continue


        ic = (
            group[
                "pred_alpha"
            ]
            .corr(
                group[
                    "future_alpha"
                ],
                method="spearman"
            )
        )


        if not np.isnan(ic):
            values.append(
                {
                    "date":
                        date,

                    "ic":
                        ic,
                }
            )


    return pd.DataFrame(
        values
    )


# ============================================================
# BACKTEST
# ============================================================

def backtest(
    predictions,
    top_k
):
    dates = sorted(
        predictions.index.unique()
    )


    # 5-day non-overlapping portfolio
    rebalance_dates = (
        dates[::HORIZON]
    )


    rows = []


    for date in rebalance_dates:

        day = (
            predictions.loc[
                predictions.index == date
            ]
            .copy()
        )


        day = (
            day.dropna(
                subset=[
                    "pred_alpha",
                    "stock_future_ret",
                    "market_future_ret",
                ]
            )
        )


        if len(day) < top_k:
            continue


        top = (
            day.nlargest(
                top_k,
                "pred_alpha"
            )
        )


        gross_ret = (
            top[
                "stock_future_ret"
            ]
            .mean()
        )


        net_ret = (
            (
                1
                + gross_ret
            )
            *
            (
                1
                - ROUND_TRIP_COST
            )
            - 1
        )


        benchmark_ret = (
            day[
                "market_future_ret"
            ]
            .iloc[0]
        )


        realized_alpha = (
            net_ret
            - benchmark_ret
        )


        rows.append({
            "date":
                date,

            "portfolio_ret":
                net_ret,

            "benchmark_ret":
                benchmark_ret,

            "realized_alpha":
                realized_alpha,

            "top_k":
                top_k,

            "stocks":
                ",".join(
                    top["ticker"]
                    .tolist()
                ),
        })


    result = (
        pd.DataFrame(rows)
        .set_index("date")
    )


    return result


# ============================================================
# WALK FORWARD
# ============================================================

year_metrics = []
all_predictions = []
all_rebalances = []
feature_rows = []


for year in range(
    args.start_year,
    args.end_year + 1
):

    test_start = pd.Timestamp(
        f"{year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{year + 1}-01-01"
    )


    # --------------------------------------------------------
    # PURGED TRAIN
    #
    # 不只 date < test_start。
    #
    # label_end_date 也一定要 < test_start，
    # 防止 label 跨進測試年度。
    # --------------------------------------------------------

    train = df[
        (
            df.index
            < test_start
        )
        &
        (
            df["label_end_date"]
            < test_start
        )
    ].copy()


    test = df[
        (
            df.index
            >= test_start
        )
        &
        (
            df.index
            < test_end
        )
    ].copy()


    if (
        len(train) < 5000
        or len(test) < 500
    ):
        logger.warning(
            f"{year}: insufficient data"
        )

        continue


    logger.info(
        "=" * 70
    )

    logger.info(
        f"YEAR {year}"
    )

    logger.info(
        f"Train: {len(train):,}"
    )

    logger.info(
        f"Test: {len(test):,}"
    )


    model = (
        make_model()
    )


    model.fit(
        train[FEATURES],
        train["future_alpha"],
    )


    pred = (
        model.predict(
            test[FEATURES]
        )
    )


    predictions = (
        test[
            [
                "ticker",
                "stock_future_ret",
                "market_future_ret",
                "future_alpha",
            ]
        ]
        .copy()
    )


    predictions[
        "pred_alpha"
    ] = pred


    predictions[
        "test_year"
    ] = year


    # --------------------------------------------------------
    # IC
    # --------------------------------------------------------

    ic_df = (
        daily_ic(
            predictions
        )
    )


    mean_ic = (
        ic_df["ic"].mean()
    )


    median_ic = (
        ic_df["ic"].median()
    )


    positive_ic = (
        (
            ic_df["ic"] > 0
        )
        .mean()
    )


    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    rebalance = (
        backtest(
            predictions,
            args.top_k
        )
    )


    rebalance[
        "test_year"
    ] = year


    strategy_return = (
        (
            1
            + rebalance[
                "portfolio_ret"
            ]
        )
        .prod()
        - 1
    )


    benchmark_return = (
        (
            1
            + rebalance[
                "benchmark_ret"
            ]
        )
        .prod()
        - 1
    )


    alpha = (
        strategy_return
        - benchmark_return
    )


    equity = (
        (
            1
            + rebalance[
                "portfolio_ret"
            ]
        )
        .cumprod()
    )


    dd = (
        max_drawdown(
            equity
        )
    )


    active = (
        rebalance[
            "portfolio_ret"
        ]
        - rebalance[
            "benchmark_ret"
        ]
    )


    if (
        len(active) > 1
        and active.std() > 0
    ):

        active_sharpe = (
            active.mean()
            / active.std()
            * math.sqrt(
                252 / HORIZON
            )
        )

    else:
        active_sharpe = np.nan


    positive_alpha_periods = (
        (
            rebalance[
                "realized_alpha"
            ]
            > 0
        )
        .mean()
    )


    row = {
        "year":
            year,

        "train_rows":
            len(train),

        "test_rows":
            len(test),

        "mean_ic":
            mean_ic,

        "median_ic":
            median_ic,

        "positive_ic_days":
            positive_ic,

        "strategy_return":
            strategy_return,

        "benchmark_return":
            benchmark_return,

        "alpha":
            alpha,

        "max_drawdown":
            dd,

        "active_sharpe":
            active_sharpe,

        "positive_alpha_periods":
            positive_alpha_periods,

        "rebalances":
            len(rebalance),
    }


    year_metrics.append(
        row
    )


    all_predictions.append(
        predictions
    )


    all_rebalances.append(
        rebalance
    )


    logger.info(
        f"Mean IC: {mean_ic:.4f}"
    )

    logger.info(
        f"IC > 0 days: "
        f"{positive_ic:.2%}"
    )

    logger.info(
        f"Strategy: "
        f"{strategy_return:.2%}"
    )

    logger.info(
        f"0050: "
        f"{benchmark_return:.2%}"
    )

    logger.info(
        f"Alpha: "
        f"{alpha:.2%}"
    )

    logger.info(
        f"Active Sharpe: "
        f"{active_sharpe:.4f}"
    )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    if args.model == "xgb":

        importance = (
            model.feature_importances_
        )

    else:

        importance = (
            np.abs(
                model.named_steps[
                    "model"
                ]
                .coef_
            )
        )


    for feature, value in zip(
        FEATURES,
        importance
    ):

        feature_rows.append({
            "year":
                year,

            "feature":
                feature,

            "importance":
                float(value),
        })


# ============================================================
# SAVE
# ============================================================

metrics_df = (
    pd.DataFrame(
        year_metrics
    )
)


predictions_df = (
    pd.concat(
        all_predictions
    )
)


rebalances_df = (
    pd.concat(
        all_rebalances
    )
)


feature_df = (
    pd.DataFrame(
        feature_rows
    )
)


metrics_df.to_csv(
    RUN_DIR
    / "yearly_metrics.csv",
    index=False,
)


predictions_df.to_parquet(
    RUN_DIR
    / "predictions.parquet"
)


rebalances_df.to_csv(
    RUN_DIR
    / "rebalances.csv"
)


feature_df.to_csv(
    RUN_DIR
    / "feature_importance_by_year.csv",
    index=False,
)


feature_summary = (
    feature_df
    .groupby("feature")[
        "importance"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
    .reset_index()
)


feature_summary.to_csv(
    RUN_DIR
    / "feature_importance.csv",
    index=False,
)


# ============================================================
# ALL-YEAR EQUITY
# ============================================================

rebalances_df = (
    rebalances_df
    .sort_index()
)


rebalances_df[
    "strategy_equity"
] = (
    (
        1
        + rebalances_df[
            "portfolio_ret"
        ]
    )
    .cumprod()
)


rebalances_df[
    "benchmark_equity"
] = (
    (
        1
        + rebalances_df[
            "benchmark_ret"
        ]
    )
    .cumprod()
)


rebalances_df.to_csv(
    RUN_DIR
    / "rebalances.csv"
)


plt.figure(
    figsize=(12, 6)
)


plt.plot(
    rebalances_df.index,
    rebalances_df[
        "strategy_equity"
    ],
    label="Lobster Top-K",
)


plt.plot(
    rebalances_df.index,
    rebalances_df[
        "benchmark_equity"
    ],
    label="0050",
)


plt.xlabel(
    "Date"
)

plt.ylabel(
    "Equity"
)

plt.title(
    f"Cross-sectional "
    f"{args.model.upper()} "
    f"Top-{args.top_k}"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    RUN_DIR
    / "equity_curve.png",
    dpi=160,
)


plt.close()


# ============================================================
# SUMMARY
# ============================================================

logger.info(
    "=" * 70
)

logger.info(
    "FINAL SUMMARY"
)

logger.info(
    "=" * 70
)


logger.info(
    f"Mean yearly IC: "
    f"{metrics_df['mean_ic'].mean():.4f}"
)


positive_alpha_years = (
    (
        metrics_df[
            "alpha"
        ] > 0
    )
    .sum()
)


logger.info(
    f"Positive alpha years: "
    f"{positive_alpha_years}"
    f"/{len(metrics_df)}"
)


total_return = (
    rebalances_df[
        "strategy_equity"
    ]
    .iloc[-1]
    - 1
)


benchmark_total = (
    rebalances_df[
        "benchmark_equity"
    ]
    .iloc[-1]
    - 1
)


logger.info(
    f"Total Strategy Return: "
    f"{total_return:.2%}"
)


logger.info(
    f"Total Benchmark Return: "
    f"{benchmark_total:.2%}"
)


logger.info(
    f"Run saved to: "
    f"{RUN_DIR}"
)
