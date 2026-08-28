from pathlib import Path
from datetime import datetime
import argparse
import json
import logging
import math
import os
import random
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
    choices=["ridge", "xgb", "mlp"],
    required=True,
)

parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cpu",
)

parser.add_argument(
    "--top-k",
    default="3,5,10",
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
    "--epochs",
    type=int,
    default=100,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=8192,
)

parser.add_argument(
    "--run-name",
    default=None,
)

args = parser.parse_args()

TOP_K_LIST = [
    int(x)
    for x in args.top_k.split(",")
]


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BUY_FEE = 0.001425
SELL_FEE = 0.001425
SELL_TAX = 0.003

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# META
# ============================================================

with open(
    "data/universe_metadata.json"
) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
HORIZON = metadata["horizon"]


# ============================================================
# RUN DIR
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

name = (
    args.run_name
    or f"{args.model}_{args.device}"
)

RUN_DIR = (
    Path("runs")
    / f"{name}_{timestamp}"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOG
# ============================================================

logger = logging.getLogger(
    "lobster-v2"
)

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console = logging.StreamHandler()
console.setFormatter(formatter)

file_handler = logging.FileHandler(
    RUN_DIR / "run.log"
)
file_handler.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file_handler)


# ============================================================
# SYSTEM INFO
# ============================================================

def shell(cmd):
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return str(exc)


config = {
    "model": args.model,
    "device": args.device,
    "top_k": TOP_K_LIST,
    "start_year": args.start_year,
    "end_year": args.end_year,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "horizon": HORIZON,
    "features": FEATURES,
    "git_commit": shell("git rev-parse HEAD"),
    "python": shell("python --version"),
    "nvidia": shell("nvidia-smi -L"),
}

with open(
    RUN_DIR / "config.json",
    "w"
) as f:
    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# DATA
# ============================================================

df = pd.read_parquet(
    "data/universe_dataset.parquet"
)

df.index = pd.to_datetime(df.index)

df["label_end_date"] = pd.to_datetime(
    df["label_end_date"]
)

logger.info(
    f"Rows: {len(df):,}"
)

logger.info(
    f"Stocks: {df['ticker'].nunique()}"
)

logger.info(
    f"Dates: {df.index.min()} -> {df.index.max()}"
)


# ============================================================
# UTILS
# ============================================================

def max_drawdown(equity):
    peak = equity.cummax()

    return (
        equity / peak - 1
    ).min()


def sharpe(ret):
    if (
        len(ret) < 2
        or ret.std(ddof=1) == 0
    ):
        return np.nan

    return (
        ret.mean()
        / ret.std(ddof=1)
        * math.sqrt(
            252 / HORIZON
        )
    )


# ============================================================
# MODEL
# ============================================================

def make_sklearn_model():

    if args.model == "ridge":

        return Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(
                    alpha=10.0
                ),
            ),
        ])

    if args.model == "xgb":

        return XGBRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.025,

            subsample=0.8,
            colsample_bytree=0.8,

            min_child_weight=5,

            reg_alpha=0.05,
            reg_lambda=1.0,

            objective="reg:squarederror",

            tree_method="hist",

            device=(
                "cuda"
                if args.device == "cuda"
                else "cpu"
            ),

            random_state=SEED,

            n_jobs=6,
        )

    raise ValueError(args.model)


# ============================================================
# MLP
# ============================================================

def train_mlp(
    X_train,
    y_train,
    X_test,
):

    import torch
    import torch.nn as nn
    from torch.utils.data import (
        TensorDataset,
        DataLoader,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device("cuda")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test
    ).astype(np.float32)

    y_train = np.asarray(
        y_train,
        dtype=np.float32,
    )

    # --------------------------------------------
    # Target normalization
    # --------------------------------------------

    y_mean = float(
        y_train.mean()
    )

    y_std = float(
        y_train.std()
    )

    if y_std < 1e-8:
        y_std = 1.0

    y_train_norm = (
        (y_train - y_mean)
        / y_std
    ).astype(np.float32)


    class MLP(nn.Module):

        def __init__(self, n_features):
            super().__init__()

            self.net = nn.Sequential(

                nn.Linear(
                    n_features,
                    1024,
                ),

                nn.GELU(),

                nn.LayerNorm(
                    1024
                ),

                nn.Dropout(
                    0.10
                ),


                nn.Linear(
                    1024,
                    512,
                ),

                nn.GELU(),

                nn.LayerNorm(
                    512
                ),

                nn.Dropout(
                    0.10
                ),


                nn.Linear(
                    512,
                    256,
                ),

                nn.GELU(),

                nn.LayerNorm(
                    256
                ),

                nn.Dropout(
                    0.05
                ),


                nn.Linear(
                    256,
                    1,
                ),
            )

        def forward(self, x):
            return (
                self.net(x)
                .squeeze(-1)
            )


    torch.manual_seed(SEED)

    model = MLP(
        X_train.shape[1]
    ).to(device)


    dataset = TensorDataset(
        torch.from_numpy(
            X_train
        ),
        torch.from_numpy(
            y_train_norm
        ),
    )


    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
    )


    loss_fn = (
        nn.SmoothL1Loss()
    )


    amp_enabled = True

    scaler_amp = (
        torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )
    )


    model.train()

    for epoch in range(
        1,
        args.epochs + 1
    ):

        running_loss = 0.0
        samples = 0

        for xb, yb in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):

                pred = model(xb)

                loss = loss_fn(
                    pred,
                    yb,
                )


            scaler_amp.scale(
                loss
            ).backward()


            scaler_amp.step(
                optimizer
            )


            scaler_amp.update()


            running_loss += (
                loss.item()
                * len(xb)
            )

            samples += len(xb)


        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == args.epochs
        ):

            logger.info(
                f"MLP epoch "
                f"{epoch}/{args.epochs} "
                f"loss="
                f"{running_loss/samples:.6f}"
            )


    # --------------------------------------------
    # Predict
    # --------------------------------------------

    model.eval()

    preds = []

    X_tensor = torch.from_numpy(
        X_test
    )

    pred_loader = DataLoader(
        X_tensor,
        batch_size=16384,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


    with torch.no_grad():

        for xb in pred_loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                pred = model(xb)

            preds.append(
                pred.float()
                .cpu()
                .numpy()
            )


    pred_norm = np.concatenate(
        preds
    )


    pred_real = (
        pred_norm * y_std
        + y_mean
    )


    del model

    torch.cuda.empty_cache()

    return pred_real


# ============================================================
# DAILY IC / SPREAD
# ============================================================

def calculate_daily_stats(
    predictions
):

    rows = []

    for date, day in predictions.groupby(
        predictions.index
    ):

        if len(day) < 10:
            continue

        ic = (
            day["pred_alpha"]
            .corr(
                day["future_alpha"],
                method="spearman",
            )
        )

        n = max(
            int(len(day) * 0.20),
            1,
        )

        sorted_day = day.sort_values(
            "pred_alpha"
        )

        bottom = (
            sorted_day
            .head(n)[
                "future_alpha"
            ]
            .mean()
        )

        top = (
            sorted_day
            .tail(n)[
                "future_alpha"
            ]
            .mean()
        )

        rows.append({
            "date": date,
            "ic": ic,
            "top_alpha": top,
            "bottom_alpha": bottom,
            "top_bottom_spread": (
                top - bottom
            ),
        })

    return pd.DataFrame(
        rows
    ).set_index("date")


# ============================================================
# TURNOVER-AWARE BACKTEST
# ============================================================

def run_portfolio(
    predictions,
    top_k,
):

    dates = sorted(
        predictions.index.unique()
    )

    rebalance_dates = (
        dates[::HORIZON]
    )

    rows = []

    old_weights = {}

    for date in rebalance_dates:

        day = predictions.loc[
            predictions.index == date
        ].copy()

        day = day.dropna(
            subset=[
                "pred_alpha",
                "stock_future_ret",
                "market_future_ret",
            ]
        )

        if len(day) < top_k:
            continue


        # --------------------------------------------
        # New target portfolio
        # --------------------------------------------

        selected = (
            day.nlargest(
                top_k,
                "pred_alpha",
            )
        )

        target_weights = {
            ticker: 1.0 / top_k
            for ticker
            in selected["ticker"]
        }


        # --------------------------------------------
        # Turnover
        # --------------------------------------------

        universe = (
            set(old_weights)
            | set(target_weights)
        )

        buy_turnover = 0.0
        sell_turnover = 0.0


        for ticker in universe:

            old = old_weights.get(
                ticker,
                0.0,
            )

            new = target_weights.get(
                ticker,
                0.0,
            )

            change = new - old

            if change > 0:
                buy_turnover += change
            else:
                sell_turnover += (
                    -change
                )


        transaction_cost = (
            buy_turnover * BUY_FEE
            +
            sell_turnover
            * (
                SELL_FEE
                + SELL_TAX
            )
        )


        # --------------------------------------------
        # Portfolio return
        # --------------------------------------------

        return_map = dict(
            zip(
                day["ticker"],
                day["stock_future_ret"],
            )
        )


        gross_ret = sum(
            weight
            * return_map[ticker]

            for ticker, weight
            in target_weights.items()
        )


        net_ret = (
            (1 - transaction_cost)
            * (1 + gross_ret)
            - 1
        )


        market_ret = float(
            day[
                "market_future_ret"
            ].iloc[0]
        )


        # --------------------------------------------
        # Equal-weight universe
        # gross benchmark
        # --------------------------------------------

        equal_weight_ret = (
            day[
                "stock_future_ret"
            ].mean()
        )


        # --------------------------------------------
        # Drift weights to end of holding period
        #
        # 下一次 rebalance 用真正漂移後權重
        # --------------------------------------------

        grown = {}

        for ticker, weight in (
            target_weights.items()
        ):

            grown[ticker] = (
                weight
                * (
                    1
                    + return_map[ticker]
                )
            )


        total_grown = sum(
            grown.values()
        )


        if total_grown > 0:

            old_weights = {
                ticker:
                    value
                    / total_grown

                for ticker, value
                in grown.items()
            }

        else:
            old_weights = (
                target_weights.copy()
            )


        rows.append({
            "date":
                date,

            "top_k":
                top_k,

            "gross_ret":
                gross_ret,

            "net_ret":
                net_ret,

            "0050_ret":
                market_ret,

            "equal_weight_ret":
                equal_weight_ret,

            "buy_turnover":
                buy_turnover,

            "sell_turnover":
                sell_turnover,

            "turnover":
                (
                    buy_turnover
                    + sell_turnover
                ) / 2,

            "transaction_cost":
                transaction_cost,

            "stocks":
                ",".join(
                    selected[
                        "ticker"
                    ].tolist()
                ),
        })


    result = (
        pd.DataFrame(rows)
        .set_index("date")
    )


    result[
        "gross_equity"
    ] = (
        1
        + result[
            "gross_ret"
        ]
    ).cumprod()


    result[
        "net_equity"
    ] = (
        1
        + result[
            "net_ret"
        ]
    ).cumprod()


    result[
        "0050_equity"
    ] = (
        1
        + result[
            "0050_ret"
        ]
    ).cumprod()


    result[
        "equal_weight_equity"
    ] = (
        1
        + result[
            "equal_weight_ret"
        ]
    ).cumprod()


    return result


# ============================================================
# WALK-FORWARD
# ============================================================

year_metrics = []

all_predictions = []

all_portfolios = []

all_daily_stats = []


for year in range(
    args.start_year,
    args.end_year + 1
):

    test_start = pd.Timestamp(
        f"{year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{year+1}-01-01"
    )


    # --------------------------------------------
    # PURGED TRAIN
    # --------------------------------------------

    train = df[
        (
            df.index < test_start
        )
        &
        (
            df["label_end_date"]
            < test_start
        )
    ].copy()


    test = df[
        (
            df.index >= test_start
        )
        &
        (
            df.index < test_end
        )
    ].copy()


    if (
        len(train) < 5000
        or len(test) < 500
    ):
        continue


    logger.info("=" * 70)
    logger.info(f"YEAR {year}")
    logger.info(
        f"Train: {len(train):,}"
    )
    logger.info(
        f"Test : {len(test):,}"
    )


    # ========================================================
    # TRAIN
    # ========================================================

    if args.model == "mlp":

        pred = train_mlp(

            train[FEATURES].values,

            train[
                "future_alpha"
            ].values,

            test[FEATURES].values,
        )

    else:

        model = (
            make_sklearn_model()
        )

        model.fit(
            train[FEATURES],
            train[
                "future_alpha"
            ],
        )

        pred = model.predict(
            test[FEATURES]
        )


    predictions = test[
        [
            "ticker",
            "stock_future_ret",
            "market_future_ret",
            "future_alpha",
        ]
    ].copy()


    predictions[
        "pred_alpha"
    ] = pred


    predictions[
        "year"
    ] = year


    all_predictions.append(
        predictions
    )


    # ========================================================
    # DAILY RANK STATS
    # ========================================================

    daily_stats = (
        calculate_daily_stats(
            predictions
        )
    )


    daily_stats[
        "year"
    ] = year


    all_daily_stats.append(
        daily_stats
    )


    mean_ic = (
        daily_stats["ic"]
        .mean()
    )

    median_ic = (
        daily_stats["ic"]
        .median()
    )

    positive_ic = (
        (
            daily_stats["ic"]
            > 0
        )
        .mean()
    )

    mean_spread = (
        daily_stats[
            "top_bottom_spread"
        ]
        .mean()
    )


    logger.info(
        f"Mean IC: {mean_ic:.4f}"
    )

    logger.info(
        f"Median IC: {median_ic:.4f}"
    )

    logger.info(
        f"IC > 0 days: "
        f"{positive_ic:.2%}"
    )

    logger.info(
        f"Mean Top-Bottom Spread: "
        f"{mean_spread:.4%}"
    )


    # ========================================================
    # MULTIPLE TOP-K
    # ========================================================

    for top_k in TOP_K_LIST:

        portfolio = run_portfolio(
            predictions,
            top_k,
        )

        portfolio[
            "year"
        ] = year

        all_portfolios.append(
            portfolio
        )


        gross_total = (
            portfolio[
                "gross_equity"
            ].iloc[-1]
            - 1
        )


        net_total = (
            portfolio[
                "net_equity"
            ].iloc[-1]
            - 1
        )


        market_total = (
            portfolio[
                "0050_equity"
            ].iloc[-1]
            - 1
        )


        ew_total = (
            portfolio[
                "equal_weight_equity"
            ].iloc[-1]
            - 1
        )


        active_ret = (
            portfolio[
                "net_ret"
            ]
            - portfolio[
                "0050_ret"
            ]
        )


        row = {

            "year":
                year,

            "model":
                args.model,

            "top_k":
                top_k,

            "mean_ic":
                mean_ic,

            "median_ic":
                median_ic,

            "positive_ic_days":
                positive_ic,

            "mean_top_bottom_spread":
                mean_spread,

            "gross_return":
                gross_total,

            "net_return":
                net_total,

            "0050_return":
                market_total,

            "equal_weight_return":
                ew_total,

            "alpha_vs_0050":
                net_total
                - market_total,

            "alpha_vs_equal_weight":
                net_total
                - ew_total,

            "cost_drag":
                gross_total
                - net_total,

            "mean_turnover":
                portfolio[
                    "turnover"
                ].mean(),

            "total_transaction_cost":
                portfolio[
                    "transaction_cost"
                ].sum(),

            "max_drawdown":
                max_drawdown(
                    portfolio[
                        "net_equity"
                    ]
                ),

            "sharpe":
                sharpe(
                    portfolio[
                        "net_ret"
                    ]
                ),

            "active_sharpe":
                sharpe(
                    active_ret
                ),

            "periods":
                len(portfolio),
        }


        year_metrics.append(
            row
        )


        logger.info(
            f"Top-{top_k} | "
            f"Net={net_total:.2%} | "
            f"0050={market_total:.2%} | "
            f"EW={ew_total:.2%} | "
            f"Alpha0050="
            f"{row['alpha_vs_0050']:.2%} | "
            f"Turnover="
            f"{row['mean_turnover']:.2%}"
        )


# ============================================================
# SAVE
# ============================================================

metrics_df = pd.DataFrame(
    year_metrics
)

predictions_df = pd.concat(
    all_predictions
).sort_index()

portfolios_df = pd.concat(
    all_portfolios
).sort_index()

daily_stats_df = pd.concat(
    all_daily_stats
).sort_index()


metrics_df.to_csv(
    RUN_DIR
    / "yearly_metrics.csv",
    index=False,
)


predictions_df.to_parquet(
    RUN_DIR
    / "predictions.parquet"
)


portfolios_df.to_csv(
    RUN_DIR
    / "portfolios.csv"
)


daily_stats_df.to_csv(
    RUN_DIR
    / "daily_rank_stats.csv"
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for top_k in TOP_K_LIST:

    subset = metrics_df[
        metrics_df["top_k"]
        == top_k
    ]

    positive_alpha_years = (
        (
            subset[
                "alpha_vs_0050"
            ]
            > 0
        ).sum()
    )

    summary_rows.append({

        "model":
            args.model,

        "top_k":
            top_k,

        "mean_ic":
            subset[
                "mean_ic"
            ].mean(),

        "mean_top_bottom_spread":
            subset[
                "mean_top_bottom_spread"
            ].mean(),

        "positive_alpha_years":
            positive_alpha_years,

        "years":
            len(subset),

        "mean_alpha_vs_0050":
            subset[
                "alpha_vs_0050"
            ].mean(),

        "mean_alpha_vs_equal_weight":
            subset[
                "alpha_vs_equal_weight"
            ].mean(),

        "mean_turnover":
            subset[
                "mean_turnover"
            ].mean(),

        "mean_active_sharpe":
            subset[
                "active_sharpe"
            ].mean(),
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    RUN_DIR
    / "summary.csv",
    index=False,
)


logger.info("=" * 70)
logger.info("FINAL SUMMARY")
logger.info("=" * 70)

logger.info(
    "\n"
    + summary_df.to_string(
        index=False
    )
)


# ============================================================
# EQUITY PLOT FOR TOP-5
# ============================================================

plot_df = portfolios_df[
    portfolios_df[
        "top_k"
    ] == 5
].sort_index()


if len(plot_df) > 0:

    # 重新從整段期間 compound
    strategy_equity = (
        1
        + plot_df[
            "net_ret"
        ]
    ).cumprod()

    benchmark_equity = (
        1
        + plot_df[
            "0050_ret"
        ]
    ).cumprod()

    ew_equity = (
        1
        + plot_df[
            "equal_weight_ret"
        ]
    ).cumprod()


    plt.figure(
        figsize=(13, 7)
    )


    plt.plot(
        plot_df.index,
        strategy_equity,
        label=f"{args.model} Top-5",
    )


    plt.plot(
        plot_df.index,
        benchmark_equity,
        label="0050",
    )


    plt.plot(
        plot_df.index,
        ew_equity,
        label="Universe Equal Weight",
    )


    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(
        f"Lobster Cross-sectional v2 - "
        f"{args.model.upper()}"
    )

    plt.legend()
    plt.tight_layout()


    plt.savefig(
        RUN_DIR
        / "equity_curve.png",
        dpi=160,
    )


    plt.close()


logger.info(
    f"Saved: {RUN_DIR}"
)
