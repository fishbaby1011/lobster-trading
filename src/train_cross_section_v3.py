from pathlib import Path
from datetime import datetime
import argparse
import copy
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

from sklearn.preprocessing import StandardScaler


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["mlp_reg", "mlp_rank"],
    required=True,
)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.10)

parser.add_argument(
    "--hidden",
    default="1024,512,256",
)

parser.add_argument(
    "--clip-q",
    type=float,
    default=0.01,
    help="Train-target clipping quantile. 0 disables clipping.",
)

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=8)
parser.add_argument("--eval-every", type=int, default=5)

parser.add_argument(
    "--batch-size",
    type=int,
    default=16384,
)

parser.add_argument(
    "--rank-day-batch",
    type=int,
    default=128,
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
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
    "--run-name",
    default=None,
)

args = parser.parse_args()

TOP_K = [
    int(x)
    for x in args.top_k.split(",")
]

HIDDEN = [
    int(x)
    for x in args.hidden.split(",")
]


# ============================================================
# CONSTANTS
# ============================================================

BUY_FEE = 0.001425
SELL_FEE = 0.001425
SELL_TAX = 0.003

SEED = args.seed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# PATHS / META
# ============================================================

with open(
    "data/universe_metadata.json"
) as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
HORIZON = metadata["horizon"]


timestamp = (
    datetime.now()
    .strftime("%Y%m%d_%H%M%S")
)

base_name = (
    args.run_name
    or f"v3_{args.model}_seed{SEED}"
)

RUN_DIR = (
    Path("runs")
    / f"{base_name}_{timestamp}"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    f"lobster-v3-{SEED}"
)

logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console = logging.StreamHandler()
console.setFormatter(formatter)

fh = logging.FileHandler(
    RUN_DIR / "run.log"
)

fh.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(fh)


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
    "seed": SEED,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "dropout": args.dropout,
    "hidden": HIDDEN,
    "clip_q": args.clip_q,
    "epochs": args.epochs,
    "patience": args.patience,
    "eval_every": args.eval_every,
    "batch_size": args.batch_size,
    "rank_day_batch": args.rank_day_batch,
    "top_k": TOP_K,
    "features": FEATURES,
    "horizon": HORIZON,
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
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    "data/universe_dataset.parquet"
)

df.index = pd.to_datetime(
    df.index
)

df["label_end_date"] = pd.to_datetime(
    df["label_end_date"]
)

logger.info(
    f"Rows={len(df):,}, "
    f"stocks={df['ticker'].nunique()}, "
    f"dates={df.index.min()}->{df.index.max()}"
)


# ============================================================
# METRICS
# ============================================================

def mean_daily_ic(
    frame,
    predictions,
):
    tmp = frame[
        ["ticker", "future_alpha"]
    ].copy()

    tmp["pred"] = predictions

    values = []

    for _, day in tmp.groupby(
        tmp.index
    ):
        if len(day) < 10:
            continue

        ic = day["pred"].corr(
            day["future_alpha"],
            method="spearman",
        )

        if not np.isnan(ic):
            values.append(ic)

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def max_drawdown(equity):
    peak = equity.cummax()

    return (
        equity / peak - 1
    ).min()


def annualized_sharpe(ret):
    if (
        len(ret) < 2
        or ret.std(ddof=1) == 0
    ):
        return np.nan

    return (
        ret.mean()
        / ret.std(ddof=1)
        * math.sqrt(252 / HORIZON)
    )


# ============================================================
# MODEL
# ============================================================

def make_model(
    n_features,
):

    import torch.nn as nn

    layers = []

    last = n_features

    for i, width in enumerate(
        HIDDEN
    ):
        layers.extend([
            nn.Linear(last, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(
                args.dropout
                if i < len(HIDDEN) - 1
                else args.dropout / 2
            ),
        ])

        last = width

    layers.append(
        nn.Linear(last, 1)
    )

    return nn.Sequential(
        *layers
    )


# ============================================================
# TARGET PREP
# ============================================================

def prepare_target(
    y_train,
    y_other=None,
):
    y_train = np.asarray(
        y_train,
        dtype=np.float32,
    )

    if args.clip_q > 0:
        lo = float(
            np.quantile(
                y_train,
                args.clip_q,
            )
        )

        hi = float(
            np.quantile(
                y_train,
                1 - args.clip_q,
            )
        )

        y_train_clip = np.clip(
            y_train,
            lo,
            hi,
        )

    else:
        lo = -np.inf
        hi = np.inf
        y_train_clip = y_train.copy()

    mean = float(
        y_train_clip.mean()
    )

    std = float(
        y_train_clip.std()
    )

    if std < 1e-8:
        std = 1.0

    y_train_norm = (
        (y_train_clip - mean)
        / std
    ).astype(np.float32)

    if y_other is None:
        return (
            y_train_norm,
            lo,
            hi,
            mean,
            std,
        )

    y_other = np.asarray(
        y_other,
        dtype=np.float32,
    )

    y_other_clip = np.clip(
        y_other,
        lo,
        hi,
    )

    y_other_norm = (
        (y_other_clip - mean)
        / std
    ).astype(np.float32)

    return (
        y_train_norm,
        y_other_norm,
        lo,
        hi,
        mean,
        std,
    )


# ============================================================
# REGRESSION TRAINING
# ============================================================

def train_regression(
    train,
    valid,
    test,
):

    import torch
    import torch.nn as nn

    from torch.utils.data import (
        TensorDataset,
        DataLoader,
    )

    device = torch.device("cuda")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURES]
    ).astype(np.float32)

    X_valid = scaler.transform(
        valid[FEATURES]
    ).astype(np.float32)

    X_test = scaler.transform(
        test[FEATURES]
    ).astype(np.float32)


    (
        y_train,
        y_valid,
        lo,
        hi,
        y_mean,
        y_std,
    ) = prepare_target(
        train["future_alpha"].values,
        valid["future_alpha"].values,
    )


    model = make_model(
        len(FEATURES)
    ).to(device)


    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )


    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    loss_fn = nn.SmoothL1Loss()

    amp_scaler = (
        torch.cuda.amp.GradScaler()
    )


    X_valid_t = torch.from_numpy(
        X_valid
    )

    best_ic = -np.inf
    best_epoch = 0
    best_state = None
    bad_evals = 0


    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        total_loss = 0.0
        total_n = 0

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
            ):
                pred = (
                    model(xb)
                    .squeeze(-1)
                )

                loss = loss_fn(
                    pred,
                    yb,
                )

            amp_scaler.scale(
                loss
            ).backward()

            amp_scaler.step(
                optimizer
            )

            amp_scaler.update()

            total_loss += (
                loss.item()
                * len(xb)
            )

            total_n += len(xb)


        if (
            epoch % args.eval_every == 0
            or epoch == 1
        ):
            model.eval()

            preds = []

            with torch.no_grad():
                for start in range(
                    0,
                    len(X_valid_t),
                    32768,
                ):
                    xb = (
                        X_valid_t[
                            start:start + 32768
                        ]
                        .to(
                            device,
                            non_blocking=True,
                        )
                    )

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        p = (
                            model(xb)
                            .squeeze(-1)
                        )

                    preds.append(
                        p.float()
                        .cpu()
                        .numpy()
                    )

            valid_pred_norm = (
                np.concatenate(preds)
            )

            valid_pred = (
                valid_pred_norm
                * y_std
                + y_mean
            )

            valid_ic = mean_daily_ic(
                valid,
                valid_pred,
            )

            logger.info(
                f"epoch={epoch:03d} "
                f"loss={total_loss/total_n:.6f} "
                f"validIC={valid_ic:.5f}"
            )

            score = (
                valid_ic
                if not np.isnan(valid_ic)
                else -np.inf
            )

            if score > best_ic + 1e-5:
                best_ic = score
                best_epoch = epoch

                best_state = {
                    k: v.detach()
                        .cpu()
                        .clone()
                    for k, v
                    in model.state_dict().items()
                }

                bad_evals = 0

            else:
                bad_evals += 1

            if bad_evals >= args.patience:
                logger.info(
                    f"early stop @ {epoch}, "
                    f"best={best_epoch}, "
                    f"best_valid_ic={best_ic:.5f}"
                )

                break


    if best_state is not None:
        model.load_state_dict(
            best_state
        )


    model.eval()

    X_test_t = torch.from_numpy(
        X_test
    )

    preds = []

    with torch.no_grad():
        for start in range(
            0,
            len(X_test_t),
            32768,
        ):
            xb = (
                X_test_t[
                    start:start + 32768
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                p = (
                    model(xb)
                    .squeeze(-1)
                )

            preds.append(
                p.float()
                .cpu()
                .numpy()
            )


    pred_norm = np.concatenate(
        preds
    )

    pred = (
        pred_norm
        * y_std
        + y_mean
    )


    del model

    torch.cuda.empty_cache()

    return (
        pred,
        best_epoch,
        best_ic,
        lo,
        hi,
    )


# ============================================================
# DAILY PADDED TENSORS FOR RANKING
# ============================================================

def build_daily_tensor(
    frame,
    scaler,
    y_lo,
    y_hi,
):

    dates = sorted(
        frame.index.unique()
    )

    groups = [
        frame.loc[
            frame.index == date
        ]
        for date in dates
    ]

    max_n = max(
        len(g)
        for g in groups
    )

    n_days = len(groups)
    n_features = len(FEATURES)

    X = np.zeros(
        (
            n_days,
            max_n,
            n_features,
        ),
        dtype=np.float32,
    )

    y = np.zeros(
        (
            n_days,
            max_n,
        ),
        dtype=np.float32,
    )

    mask = np.zeros(
        (
            n_days,
            max_n,
        ),
        dtype=bool,
    )


    for i, g in enumerate(groups):

        x = scaler.transform(
            g[FEATURES]
        ).astype(np.float32)

        yy = np.clip(
            g["future_alpha"]
            .values
            .astype(np.float32),
            y_lo,
            y_hi,
        )

        n = len(g)

        X[i, :n] = x
        y[i, :n] = yy
        mask[i, :n] = True


    return (
        dates,
        X,
        y,
        mask,
    )


# ============================================================
# RANK LOSS
# ============================================================

def pairwise_rank_loss(
    predictions,
    targets,
    mask,
):

    import torch
    import torch.nn.functional as F

    pdiff = (
        predictions.unsqueeze(2)
        - predictions.unsqueeze(1)
    )

    ydiff = (
        targets.unsqueeze(2)
        - targets.unsqueeze(1)
    )

    sign = torch.sign(
        ydiff
    )


    pair_mask = (
        mask.unsqueeze(2)
        &
        mask.unsqueeze(1)
        &
        (sign != 0)
    )


    n = predictions.shape[1]

    upper = torch.triu(
        torch.ones(
            (n, n),
            dtype=torch.bool,
            device=predictions.device,
        ),
        diagonal=1,
    )


    pair_mask = (
        pair_mask
        &
        upper.unsqueeze(0)
    )


    values = (
        -sign
        * pdiff
    )[pair_mask]


    if values.numel() == 0:
        return (
            predictions.sum()
            * 0.0
        )


    return F.softplus(
        values
    ).mean()


# ============================================================
# RANK TRAINING
# ============================================================

def train_ranking(
    train,
    valid,
    test,
):

    import torch

    from torch.utils.data import (
        TensorDataset,
        DataLoader,
    )


    device = torch.device("cuda")

    scaler = StandardScaler()

    scaler.fit(
        train[FEATURES]
    )


    raw_y = (
        train["future_alpha"]
        .values
        .astype(np.float32)
    )


    if args.clip_q > 0:
        lo = float(
            np.quantile(
                raw_y,
                args.clip_q,
            )
        )

        hi = float(
            np.quantile(
                raw_y,
                1 - args.clip_q,
            )
        )
    else:
        lo = -np.inf
        hi = np.inf


    (
        _,
        X_train,
        y_train,
        mask_train,
    ) = build_daily_tensor(
        train,
        scaler,
        lo,
        hi,
    )


    model = make_model(
        len(FEATURES)
    ).to(device)


    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(mask_train),
    )


    loader = DataLoader(
        dataset,
        batch_size=args.rank_day_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    amp_scaler = (
        torch.cuda.amp.GradScaler()
    )


    best_ic = -np.inf
    best_epoch = 0
    best_state = None
    bad_evals = 0


    X_valid = scaler.transform(
        valid[FEATURES]
    ).astype(np.float32)


    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        running = 0.0
        batches = 0

        for xb, yb, mb in loader:

            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            mb = mb.to(
                device,
                non_blocking=True,
            )


            B, N, F = xb.shape

            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                pred = (
                    model(
                        xb.reshape(
                            B * N,
                            F,
                        )
                    )
                    .reshape(B, N)
                )


                loss = pairwise_rank_loss(
                    pred,
                    yb,
                    mb,
                )


            amp_scaler.scale(
                loss
            ).backward()


            amp_scaler.step(
                optimizer
            )


            amp_scaler.update()


            running += loss.item()
            batches += 1


        if (
            epoch % args.eval_every == 0
            or epoch == 1
        ):

            valid_pred = predict_model(
                model,
                X_valid,
                device,
            )


            valid_ic = mean_daily_ic(
                valid,
                valid_pred,
            )


            logger.info(
                f"epoch={epoch:03d} "
                f"rankloss={running/max(batches,1):.6f} "
                f"validIC={valid_ic:.5f}"
            )


            score = (
                valid_ic
                if not np.isnan(valid_ic)
                else -np.inf
            )


            if score > best_ic + 1e-5:

                best_ic = score
                best_epoch = epoch

                best_state = {
                    k: v.detach()
                        .cpu()
                        .clone()
                    for k, v
                    in model.state_dict().items()
                }

                bad_evals = 0

            else:
                bad_evals += 1


            if bad_evals >= args.patience:

                logger.info(
                    f"early stop @ {epoch}, "
                    f"best={best_epoch}, "
                    f"best_valid_ic={best_ic:.5f}"
                )

                break


    if best_state is not None:
        model.load_state_dict(
            best_state
        )


    X_test = scaler.transform(
        test[FEATURES]
    ).astype(np.float32)


    pred = predict_model(
        model,
        X_test,
        device,
    )


    del model

    torch.cuda.empty_cache()


    return (
        pred,
        best_epoch,
        best_ic,
        lo,
        hi,
    )


def predict_model(
    model,
    X,
    device,
):

    import torch

    model.eval()

    tensor = torch.from_numpy(
        X
    )

    out = []

    with torch.no_grad():

        for start in range(
            0,
            len(tensor),
            32768,
        ):

            xb = (
                tensor[
                    start:start + 32768
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                p = (
                    model(xb)
                    .squeeze(-1)
                )

            out.append(
                p.float()
                .cpu()
                .numpy()
            )

    return np.concatenate(out)


# ============================================================
# DAILY STATS
# ============================================================

def daily_stats(
    predictions,
):

    rows = []

    for date, day in predictions.groupby(
        predictions.index
    ):

        if len(day) < 10:
            continue

        ic = day[
            "pred_alpha"
        ].corr(
            day["future_alpha"],
            method="spearman",
        )

        n = max(
            int(len(day) * 0.20),
            1,
        )

        ordered = day.sort_values(
            "pred_alpha"
        )

        bottom = (
            ordered.head(n)[
                "future_alpha"
            ].mean()
        )

        top = (
            ordered.tail(n)[
                "future_alpha"
            ].mean()
        )

        rows.append({
            "date": date,
            "ic": ic,
            "top_alpha": top,
            "bottom_alpha": bottom,
            "spread": top - bottom,
        })


    return (
        pd.DataFrame(rows)
        .set_index("date")
    )


# ============================================================
# TURNOVER BACKTEST
# ============================================================

def backtest(
    predictions,
    top_k,
):

    dates = sorted(
        predictions.index.unique()
    )

    rebalance_dates = (
        dates[::HORIZON]
    )

    previous_weights = {}

    rows = []


    for date in rebalance_dates:

        day = predictions.loc[
            predictions.index == date
        ].copy()

        if len(day) < top_k:
            continue


        selected = day.nlargest(
            top_k,
            "pred_alpha",
        )


        target_weights = {
            t: 1.0 / top_k
            for t
            in selected["ticker"]
        }


        universe = (
            set(previous_weights)
            | set(target_weights)
        )


        buy_turnover = 0.0
        sell_turnover = 0.0


        for ticker in universe:

            old = previous_weights.get(
                ticker,
                0.0,
            )

            new = target_weights.get(
                ticker,
                0.0,
            )

            delta = new - old

            if delta > 0:
                buy_turnover += delta
            else:
                sell_turnover += -delta


        cost = (
            buy_turnover
            * BUY_FEE
            +
            sell_turnover
            * (
                SELL_FEE
                + SELL_TAX
            )
        )


        ret_map = dict(
            zip(
                day["ticker"],
                day["stock_future_ret"],
            )
        )


        gross = sum(
            weight * ret_map[ticker]
            for ticker, weight
            in target_weights.items()
        )


        net = (
            (1 - cost)
            * (1 + gross)
            - 1
        )


        market = float(
            day["market_future_ret"]
            .iloc[0]
        )


        ew = float(
            day["stock_future_ret"]
            .mean()
        )


        grown = {
            ticker:
            weight
            * (
                1
                + ret_map[ticker]
            )

            for ticker, weight
            in target_weights.items()
        }


        total = sum(
            grown.values()
        )


        if total > 0:
            previous_weights = {
                k: v / total
                for k, v
                in grown.items()
            }
        else:
            previous_weights = (
                target_weights.copy()
            )


        rows.append({
            "date": date,
            "gross_ret": gross,
            "net_ret": net,
            "0050_ret": market,
            "equal_weight_ret": ew,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "turnover": (
                buy_turnover
                + sell_turnover
            ) / 2,
            "transaction_cost": cost,
            "stocks": ",".join(
                selected["ticker"]
                .tolist()
            ),
        })


    return (
        pd.DataFrame(rows)
        .set_index("date")
    )


# ============================================================
# WALK FORWARD
# ============================================================

metrics = []
all_predictions = []
all_portfolios = []
all_daily = []


for year in range(
    args.start_year,
    args.end_year + 1
):

    valid_year = year - 1

    valid_start = pd.Timestamp(
        f"{valid_year}-01-01"
    )

    test_start = pd.Timestamp(
        f"{year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{year+1}-01-01"
    )


    train = df[
        (df.index < valid_start)
        &
        (
            df["label_end_date"]
            < valid_start
        )
    ].copy()


    valid = df[
        (df.index >= valid_start)
        &
        (df.index < test_start)
        &
        (
            df["label_end_date"]
            < test_start
        )
    ].copy()


    test = df[
        (df.index >= test_start)
        &
        (df.index < test_end)
    ].copy()


    if (
        len(train) < 5000
        or len(valid) < 500
        or len(test) < 500
    ):
        continue


    logger.info("=" * 80)

    logger.info(
        f"YEAR {year} | "
        f"train={len(train):,} "
        f"valid={len(valid):,} "
        f"test={len(test):,}"
    )


    if args.model == "mlp_reg":

        (
            pred,
            best_epoch,
            best_valid_ic,
            clip_lo,
            clip_hi,
        ) = train_regression(
            train,
            valid,
            test,
        )

    else:

        (
            pred,
            best_epoch,
            best_valid_ic,
            clip_lo,
            clip_hi,
        ) = train_ranking(
            train,
            valid,
            test,
        )


    predictions = test[
        [
            "ticker",
            "stock_future_ret",
            "market_future_ret",
            "future_alpha",
        ]
    ].copy()


    predictions["pred_alpha"] = pred
    predictions["year"] = year
    predictions["seed"] = SEED


    all_predictions.append(
        predictions
    )


    stats = daily_stats(
        predictions
    )

    stats["year"] = year
    stats["seed"] = SEED

    all_daily.append(stats)


    mean_ic = float(
        stats["ic"].mean()
    )

    median_ic = float(
        stats["ic"].median()
    )

    positive_ic_days = float(
        (stats["ic"] > 0).mean()
    )

    spread = float(
        stats["spread"].mean()
    )


    logger.info(
        f"TEST IC={mean_ic:.5f} | "
        f"median={median_ic:.5f} | "
        f"IC>0={positive_ic_days:.2%} | "
        f"spread={spread:.4%}"
    )


    for k in TOP_K:

        portfolio = backtest(
            predictions,
            k,
        )

        portfolio["year"] = year
        portfolio["seed"] = SEED
        portfolio["top_k"] = k

        all_portfolios.append(
            portfolio
        )


        gross_return = (
            (1 + portfolio["gross_ret"])
            .prod()
            - 1
        )

        net_return = (
            (1 + portfolio["net_ret"])
            .prod()
            - 1
        )

        market_return = (
            (1 + portfolio["0050_ret"])
            .prod()
            - 1
        )

        ew_return = (
            (
                1
                + portfolio[
                    "equal_weight_ret"
                ]
            )
            .prod()
            - 1
        )


        equity = (
            1
            + portfolio["net_ret"]
        ).cumprod()


        active = (
            portfolio["net_ret"]
            - portfolio["0050_ret"]
        )


        row = {
            "year": year,
            "model": args.model,
            "seed": SEED,
            "top_k": k,

            "best_epoch": best_epoch,
            "best_valid_ic": best_valid_ic,

            "clip_lo": clip_lo,
            "clip_hi": clip_hi,

            "mean_ic": mean_ic,
            "median_ic": median_ic,
            "positive_ic_days":
                positive_ic_days,

            "mean_top_bottom_spread":
                spread,

            "gross_return":
                gross_return,

            "net_return":
                net_return,

            "0050_return":
                market_return,

            "equal_weight_return":
                ew_return,

            "alpha_vs_0050":
                net_return
                - market_return,

            "alpha_vs_equal_weight":
                net_return
                - ew_return,

            "mean_turnover":
                portfolio[
                    "turnover"
                ].mean(),

            "total_cost":
                portfolio[
                    "transaction_cost"
                ].sum(),

            "max_drawdown":
                max_drawdown(
                    equity
                ),

            "sharpe":
                annualized_sharpe(
                    portfolio["net_ret"]
                ),

            "active_sharpe":
                annualized_sharpe(
                    active
                ),

            "periods":
                len(portfolio),
        }


        metrics.append(row)


        logger.info(
            f"Top-{k} | "
            f"net={net_return:.2%} | "
            f"0050={market_return:.2%} | "
            f"EW={ew_return:.2%} | "
            f"alpha={row['alpha_vs_0050']:.2%} | "
            f"turn={row['mean_turnover']:.2%}"
        )


# ============================================================
# SAVE
# ============================================================

metrics_df = pd.DataFrame(
    metrics
)

predictions_df = pd.concat(
    all_predictions
).sort_index()

portfolio_df = pd.concat(
    all_portfolios
).sort_index()

daily_df = pd.concat(
    all_daily
).sort_index()


metrics_df.to_csv(
    RUN_DIR / "yearly_metrics.csv",
    index=False,
)

predictions_df.to_parquet(
    RUN_DIR / "predictions.parquet"
)

portfolio_df.to_csv(
    RUN_DIR / "portfolios.csv"
)

daily_df.to_csv(
    RUN_DIR / "daily_rank_stats.csv"
)


summary_rows = []

for k in TOP_K:

    s = metrics_df[
        metrics_df["top_k"] == k
    ]

    summary_rows.append({
        "model": args.model,
        "seed": SEED,
        "top_k": k,

        "mean_ic":
            s["mean_ic"].mean(),

        "median_year_ic":
            s["mean_ic"].median(),

        "mean_spread":
            s[
                "mean_top_bottom_spread"
            ].mean(),

        "positive_alpha_years":
            (
                s["alpha_vs_0050"] > 0
            ).sum(),

        "years": len(s),

        "mean_alpha_vs_0050":
            s["alpha_vs_0050"].mean(),

        "mean_alpha_vs_equal_weight":
            s[
                "alpha_vs_equal_weight"
            ].mean(),

        "median_alpha_vs_0050":
            s[
                "alpha_vs_0050"
            ].median(),

        "mean_turnover":
            s["mean_turnover"].mean(),

        "mean_active_sharpe":
            s["active_sharpe"].mean(),

        "mean_best_epoch":
            s["best_epoch"].mean(),

        "mean_valid_ic":
            s["best_valid_ic"].mean(),
    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    RUN_DIR / "summary.csv",
    index=False,
)


logger.info("=" * 80)
logger.info("FINAL SUMMARY")
logger.info("=" * 80)

logger.info(
    "\n"
    + summary.to_string(
        index=False
    )
)

logger.info(
    f"SAVED: {RUN_DIR}"
)
