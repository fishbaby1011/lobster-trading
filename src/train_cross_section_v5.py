from pathlib import Path
from datetime import datetime
import argparse, json, logging, random, subprocess

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


p = argparse.ArgumentParser()

p.add_argument(
    "--horizon",
    type=int,
    choices=[10, 20],
    required=True,
)

p.add_argument(
    "--seed",
    type=int,
    default=42,
)

p.add_argument(
    "--lr",
    type=float,
    default=1e-3,
)

p.add_argument(
    "--weight-decay",
    type=float,
    default=1e-4,
)

p.add_argument(
    "--dropout",
    type=float,
    default=0.10,
)

p.add_argument(
    "--hidden",
    default="1024,512,256",
)

p.add_argument(
    "--clip-q",
    type=float,
    default=0.01,
)

p.add_argument(
    "--epochs",
    type=int,
    default=200,
)

p.add_argument(
    "--patience",
    type=int,
    default=8,
)

p.add_argument(
    "--eval-every",
    type=int,
    default=5,
)

p.add_argument(
    "--rank-day-batch",
    type=int,
    default=128,
)

p.add_argument(
    "--num-workers",
    type=int,
    default=0,
)

p.add_argument(
    "--start-year",
    type=int,
    default=2018,
)

p.add_argument(
    "--end-year",
    type=int,
    default=2026,
)

p.add_argument(
    "--run-name",
)

args = p.parse_args()


H = args.horizon
SEED = args.seed

HIDDEN = [
    int(x)
    for x in args.hidden.split(",")
]


def shell(cmd):
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    except Exception as e:
        return str(e)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


with open(
    "data/universe_multihorizon_metadata.json"
) as f:
    meta = json.load(f)


FEATURES = meta["features"]


ts = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

name = (
    args.run_name
    or f"v5_h{H}_seed{SEED}"
)

RUN = (
    Path("runs")
    / f"{name}_{ts}"
)

RUN.mkdir(
    parents=True,
    exist_ok=True,
)


log = logging.getLogger(
    f"v5-h{H}-s{SEED}"
)

log.setLevel(logging.INFO)

log.handlers.clear()


fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)


for handler in (
    logging.StreamHandler(),
    logging.FileHandler(
        RUN / "run.log"
    ),
):

    handler.setFormatter(fmt)

    log.addHandler(
        handler
    )


config = {
    "version":
        "v5",

    "horizon":
        H,

    "seed":
        SEED,

    "lr":
        args.lr,

    "weight_decay":
        args.weight_decay,

    "dropout":
        args.dropout,

    "hidden":
        HIDDEN,

    "clip_q":
        args.clip_q,

    "epochs":
        args.epochs,

    "patience":
        args.patience,

    "eval_every":
        args.eval_every,

    "rank_day_batch":
        args.rank_day_batch,

    "start_year":
        args.start_year,

    "end_year":
        args.end_year,

    "features":
        FEATURES,

    "git_commit":
        shell(
            "git rev-parse HEAD"
        ),

    "nvidia":
        shell(
            "nvidia-smi -L"
        ),

    "methodology":
        (
            "select epoch on Y-1, "
            "then refit on all purged "
            "pre-Y data for that epoch "
            "count, then predict Y"
        ),
}


(
    RUN
    / "config.json"
).write_text(
    json.dumps(
        config,
        indent=2,
        ensure_ascii=False,
    )
)


df = pd.read_parquet(
    "data/universe_multihorizon.parquet"
).sort_index()


df.index = pd.to_datetime(
    df.index
)


df["stock_future_ret"] = (
    df[
        f"stock_future_ret_h{H}"
    ]
)

df["market_future_ret"] = (
    df[
        f"market_future_ret_h{H}"
    ]
)

df["future_alpha"] = (
    df[
        f"future_alpha_h{H}"
    ]
)

df["label_end_date"] = (
    pd.to_datetime(
        df[
            f"label_end_date_h{H}"
        ]
    )
)


df = df.dropna(
    subset=
    FEATURES
    + [
        "stock_future_ret",
        "market_future_ret",
        "future_alpha",
        "label_end_date",
    ]
)


log.info(
    f"rows={len(df):,} "
    f"stocks={df.ticker.nunique()} "
    f"dates={df.index.min()}"
    f"->{df.index.max()} "
    f"H={H}"
)


def make_model(
    n_features,
):

    import torch.nn as nn

    layers = []

    last = n_features


    for i, width in enumerate(
        HIDDEN
    ):

        layers += [
            nn.Linear(
                last,
                width,
            ),

            nn.GELU(),

            nn.LayerNorm(
                width
            ),

            nn.Dropout(
                args.dropout
                if i
                < len(HIDDEN) - 1
                else
                args.dropout / 2
            ),
        ]

        last = width


    layers.append(
        nn.Linear(
            last,
            1,
        )
    )


    return nn.Sequential(
        *layers
    )


def clip_bounds(
    frame,
):

    if args.clip_q <= 0:
        return (
            -np.inf,
            np.inf,
        )


    y = (
        frame[
            "future_alpha"
        ]
        .to_numpy(
            np.float32
        )
    )


    return (
        float(
            np.quantile(
                y,
                args.clip_q,
            )
        ),

        float(
            np.quantile(
                y,
                1 - args.clip_q,
            )
        ),
    )


def daily_tensor(
    frame,
    scaler,
    lo,
    hi,
):

    groups = list(
        frame.groupby(
            frame.index,
            sort=True,
        )
    )


    max_n = max(
        len(g)
        for _, g
        in groups
    )


    X = np.zeros(
        (
            len(groups),
            max_n,
            len(FEATURES),
        ),
        np.float32,
    )


    y = np.zeros(
        (
            len(groups),
            max_n,
        ),
        np.float32,
    )


    mask = np.zeros(
        (
            len(groups),
            max_n,
        ),
        bool,
    )


    for i, (_, g) in enumerate(
        groups
    ):

        n = len(g)


        X[i, :n] = (
            scaler
            .transform(
                g[FEATURES]
            )
            .astype(
                np.float32
            )
        )


        y[i, :n] = np.clip(
            g[
                "future_alpha"
            ].to_numpy(
                np.float32
            ),
            lo,
            hi,
        )


        mask[i, :n] = True


    return (
        X,
        y,
        mask,
    )


def rank_loss(
    pred,
    target,
    mask,
):

    import torch

    import torch.nn.functional as F


    pdiff = (
        pred.unsqueeze(2)
        - pred.unsqueeze(1)
    )


    sign = torch.sign(
        target.unsqueeze(2)
        - target.unsqueeze(1)
    )


    valid = (
        mask.unsqueeze(2)
        &
        mask.unsqueeze(1)
        &
        (sign != 0)
    )


    upper = torch.triu(
        torch.ones(
            (
                pred.shape[1],
                pred.shape[1],
            ),
            dtype=torch.bool,
            device=pred.device,
        ),
        diagonal=1,
    )


    values = (
        -sign
        * pdiff
    )[
        valid
        &
        upper.unsqueeze(0)
    ]


    if values.numel() == 0:
        return (
            pred.sum()
            * 0.0
        )


    return (
        F.softplus(
            values
        )
        .mean()
    )


def loader_for(
    X,
    y,
    mask,
    seed,
):

    import torch

    from torch.utils.data import (
        DataLoader,
        TensorDataset,
    )


    gen = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )


    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        ),

        batch_size=
            args.rank_day_batch,

        shuffle=True,

        generator=gen,

        num_workers=
            args.num_workers,

        pin_memory=True,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
):

    import torch


    model.train()


    total = 0.0
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
                .reshape(
                    B,
                    N,
                )
            )


            loss = rank_loss(
                pred,
                yb,
                mb,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        total += float(
            loss.item()
        )

        batches += 1


    return (
        total
        / max(
            batches,
            1,
        )
    )


def predict(
    model,
    X,
    device,
):

    import torch


    model.eval()


    X = torch.from_numpy(
        X
    )


    out = []


    with torch.no_grad():

        for i in range(
            0,
            len(X),
            32768,
        ):

            xb = (
                X[
                    i:i+32768
                ]
                .to(
                    device
                )
            )


            out.append(
                model(xb)
                .squeeze(-1)
                .float()
                .cpu()
                .numpy()
            )


    return np.concatenate(
        out
    )


def mean_daily_ic(
    frame,
    pred,
):

    t = frame[
        [
            "ticker",
            "future_alpha",
        ]
    ].copy()


    t["pred"] = pred


    vals = []


    for _, g in t.groupby(
        t.index
    ):

        if len(g) < 10:
            continue


        v = (
            g["pred"]
            .corr(
                g["future_alpha"],
                method="spearman",
            )
        )


        if not np.isnan(v):
            vals.append(v)


    if not vals:
        return np.nan


    return float(
        np.mean(vals)
    )


def mean_daily_spread(
    frame,
    pred,
):

    t = frame[
        ["future_alpha"]
    ].copy()


    t["pred"] = pred


    vals = []


    for _, g in t.groupby(
        t.index
    ):

        if len(g) < 10:
            continue


        n = max(
            int(
                len(g)
                * .2
            ),
            1,
        )


        g = g.sort_values(
            "pred"
        )


        vals.append(
            g.tail(n)[
                "future_alpha"
            ].mean()
            -
            g.head(n)[
                "future_alpha"
            ].mean()
        )


    if not vals:
        return np.nan


    return float(
        np.mean(vals)
    )


def choose_epoch(
    train,
    valid,
    fold_seed,
):

    import torch


    seed_all(
        fold_seed
    )


    device = torch.device(
        "cuda"
    )


    ss = (
        StandardScaler()
        .fit(
            train[FEATURES]
        )
    )


    lo, hi = clip_bounds(
        train
    )


    X, y, m = daily_tensor(
        train,
        ss,
        lo,
        hi,
    )


    loader = loader_for(
        X,
        y,
        m,
        fold_seed,
    )


    model = (
        make_model(
            len(FEATURES)
        )
        .to(device)
    )


    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )


    amp = (
        torch.amp
        .GradScaler(
            "cuda"
        )
    )


    Xv = (
        ss.transform(
            valid[FEATURES]
        )
        .astype(
            np.float32
        )
    )


    best_ic = -np.inf
    best_epoch = 1
    bad = 0


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        loss = train_epoch(
            model,
            loader,
            opt,
            amp,
            device,
        )


        if (
            epoch == 1
            or
            epoch
            % args.eval_every
            == 0
        ):

            vic = mean_daily_ic(
                valid,
                predict(
                    model,
                    Xv,
                    device,
                ),
            )


            log.info(
                f"select "
                f"epoch={epoch:03d} "
                f"loss={loss:.6f} "
                f"validIC={vic:.5f}"
            )


            score = (
                vic
                if not np.isnan(vic)
                else
                -np.inf
            )


            if (
                score
                >
                best_ic
                + 1e-5
            ):

                best_ic = score
                best_epoch = epoch
                bad = 0

            else:
                bad += 1


            if (
                bad
                >= args.patience
            ):

                log.info(
                    f"select "
                    f"early-stop={epoch} "
                    f"best_epoch={best_epoch} "
                    f"best_valid_ic="
                    f"{best_ic:.5f}"
                )

                break


    del model

    torch.cuda.empty_cache()


    return (
        int(best_epoch),
        float(best_ic),
    )


def refit_predict(
    refit,
    test,
    epochs,
    fold_seed,
):

    import torch


    seed_all(
        fold_seed
    )


    device = torch.device(
        "cuda"
    )


    ss = (
        StandardScaler()
        .fit(
            refit[FEATURES]
        )
    )


    lo, hi = clip_bounds(
        refit
    )


    X, y, m = daily_tensor(
        refit,
        ss,
        lo,
        hi,
    )


    loader = loader_for(
        X,
        y,
        m,
        fold_seed,
    )


    model = (
        make_model(
            len(FEATURES)
        )
        .to(device)
    )


    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )


    amp = (
        torch.amp
        .GradScaler(
            "cuda"
        )
    )


    final_loss = np.nan


    for _ in range(
        epochs
    ):

        final_loss = train_epoch(
            model,
            loader,
            opt,
            amp,
            device,
        )


    Xt = (
        ss.transform(
            test[FEATURES]
        )
        .astype(
            np.float32
        )
    )


    pred = predict(
        model,
        Xt,
        device,
    )


    del model

    torch.cuda.empty_cache()


    return (
        pred,
        lo,
        hi,
        float(final_loss),
    )


folds = []
predictions = []


for year in range(
    args.start_year,
    args.end_year + 1,
):

    valid_start = pd.Timestamp(
        f"{year-1}-01-01"
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
            df.label_end_date
            < valid_start
        )
    ].copy()


    valid = df[
        (df.index >= valid_start)
        &
        (df.index < test_start)
        &
        (
            df.label_end_date
            < test_start
        )
    ].copy()


    refit = df[
        (df.index < test_start)
        &
        (
            df.label_end_date
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
        or
        len(valid) < 500
        or
        len(refit) < 500
        or
        len(test) < 500
    ):

        log.warning(
            f"skip {year}: "
            f"train={len(train)} "
            f"valid={len(valid)} "
            f"refit={len(refit)} "
            f"test={len(test)}"
        )

        continue


    log.info(
        "=" * 90
    )


    log.info(
        f"YEAR {year} "
        f"train={len(train):,} "
        f"valid={len(valid):,} "
        f"refit={len(refit):,} "
        f"test={len(test):,}"
    )


    fold_seed = (
        SEED
        * 100000
        + year
    )


    (
        best_epoch,
        best_valid_ic,
    ) = choose_epoch(
        train,
        valid,
        fold_seed,
    )


    log.info(
        f"REFIT "
        f"year={year} "
        f"epochs={best_epoch} "
        f"rows={len(refit):,}"
    )


    (
        pred,
        lo,
        hi,
        final_loss,
    ) = refit_predict(
        refit,
        test,
        best_epoch,
        fold_seed,
    )


    test_ic = mean_daily_ic(
        test,
        pred,
    )


    test_spread = (
        mean_daily_spread(
            test,
            pred,
        )
    )


    log.info(
        f"TEST "
        f"year={year} "
        f"IC={test_ic:.5f} "
        f"spread="
        f"{test_spread:.4%}"
    )


    out = test[
        [
            "ticker",
            "future_alpha",
            "stock_future_ret",
            "market_future_ret",
        ]
    ].copy()


    out["pred_alpha"] = pred
    out["year"] = year
    out["seed"] = SEED
    out["horizon"] = H


    predictions.append(
        out
    )


    folds.append({
        "year":
            year,

        "horizon":
            H,

        "seed":
            SEED,

        "train_rows":
            len(train),

        "valid_rows":
            len(valid),

        "refit_rows":
            len(refit),

        "test_rows":
            len(test),

        "best_epoch":
            best_epoch,

        "best_valid_ic":
            best_valid_ic,

        "test_ic":
            test_ic,

        "test_spread":
            test_spread,

        "refit_final_loss":
            final_loss,

        "clip_lo":
            lo,

        "clip_hi":
            hi,
    })


if not predictions:
    raise RuntimeError(
        "No OOS predictions produced"
    )


pred = (
    pd.concat(
        predictions
    )
    .sort_index()
)


fold_df = pd.DataFrame(
    folds
)


pred.to_parquet(
    RUN
    / "predictions.parquet"
)


fold_df.to_csv(
    RUN
    / "fold_metrics.csv",
    index=False,
)


summary = {
    "horizon":
        H,

    "seed":
        SEED,

    "years":
        len(fold_df),

    "mean_test_ic":
        float(
            fold_df.test_ic.mean()
        ),

    "median_test_ic":
        float(
            fold_df.test_ic.median()
        ),

    "std_test_ic":
        float(
            fold_df.test_ic.std(
                ddof=1
            )
        ),

    "mean_test_spread":
        float(
            fold_df.test_spread.mean()
        ),

    "mean_best_epoch":
        float(
            fold_df.best_epoch.mean()
        ),

    "mean_valid_ic":
        float(
            fold_df.best_valid_ic.mean()
        ),
}


(
    RUN
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


log.info(
    "=" * 90
)


log.info(
    "FINAL SUMMARY\n"
    +
    json.dumps(
        summary,
        indent=2,
    )
)


log.info(
    f"SAVED: {RUN}"
)
