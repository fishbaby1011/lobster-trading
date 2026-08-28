from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd


p = argparse.ArgumentParser()

p.add_argument(
    "--seeds",
    default=
        "1,7,42,123,2026,3407,8765,9999",
)

p.add_argument(
    "--random-trials",
    type=int,
    default=1000,
)

# 延續 v4 的研究假設。
# 不是宣稱這是券商當下精確費率。
p.add_argument(
    "--buy-fee",
    type=float,
    default=0.001425,
)

p.add_argument(
    "--sell-fee",
    type=float,
    default=0.001425,
)

p.add_argument(
    "--sell-tax",
    type=float,
    default=0.003,
)

p.add_argument(
    "--slippage-bps",
    default="0,5,10,20",
)

args = p.parse_args()


SEEDS = [
    int(x)
    for x
    in args.seeds.split(",")
]


SLIPS = [
    float(x)
    for x
    in args.slippage_bps.split(",")
]


HORIZONS = [
    10,
    20,
]


ENTRY_EXIT = [
    (3, 3),
    (3, 5),
    (3, 10),

    (5, 5),
    (5, 10),
    (5, 15),

    (10, 10),
    (10, 15),
    (10, 20),

    (15, 15),
    (15, 20),
    (15, 25),
]


RESULTS = Path(
    "results"
)

RESULTS.mkdir(
    exist_ok=True
)


def latest_run(
    h,
    s,
):

    runs = sorted(
        Path("runs").glob(
            f"v5_h{h}_seed{s}_*"
        )
    )


    if not runs:
        raise FileNotFoundError(
            f"missing v5 "
            f"h={h} seed={s}"
        )


    return runs[-1]


pred_parts = []
fold_parts = []
manifest = []


for h in HORIZONS:

    for s in SEEDS:

        run = latest_run(
            h,
            s,
        )


        for filename in [
            "predictions.parquet",
            "fold_metrics.csv",
            "config.json",
        ]:

            if not (
                run
                / filename
            ).exists():

                raise RuntimeError(
                    f"incomplete run: "
                    f"{run}"
                )


        cfg = json.loads(
            (
                run
                / "config.json"
            ).read_text()
        )


        x = (
            pd.read_parquet(
                run
                / "predictions.parquet"
            )
            .reset_index()
        )


        x = x.rename(
            columns={
                x.columns[0]:
                    "date"
            }
        )


        x["date"] = (
            pd.to_datetime(
                x["date"]
            )
        )


        pred_parts.append(
            x
        )


        m = pd.read_csv(
            run
            / "fold_metrics.csv"
        )


        m["run"] = run.name


        fold_parts.append(
            m
        )


        manifest.append({
            "horizon":
                h,

            "seed":
                s,

            "run":
                run.name,

            "git_commit":
                cfg.get(
                    "git_commit"
                ),
        })


pd.DataFrame(
    manifest
).to_csv(
    RESULTS
    / "v5_run_manifest.csv",
    index=False,
)


folds = pd.concat(
    fold_parts,
    ignore_index=True,
)


folds.to_csv(
    RESULTS
    / "v5_fold_metrics.csv",
    index=False,
)


pred = pd.concat(
    pred_parts,
    ignore_index=True,
)


# ------------------------------------------------------------
# 每個 seed / date 先轉 cross-sectional percentile rank
# 再跨 8 seeds 平均。
# ------------------------------------------------------------

pred["seed_rank"] = (
    pred
    .groupby(
        [
            "horizon",
            "seed",
            "date",
        ]
    )[
        "pred_alpha"
    ]
    .rank(
        method="average",
        pct=True,
    )
)


ens = (
    pred
    .groupby(
        [
            "horizon",
            "date",
            "ticker",
        ],
        as_index=False,
    )
    .agg(
        score=(
            "seed_rank",
            "mean",
        ),

        n_seeds=(
            "seed",
            "nunique",
        ),
    )
)


if (
    ens.n_seeds.min()
    != len(SEEDS)
):
    raise RuntimeError(
        "not all ensemble rows "
        "contain all seeds"
    )


h10d = ens.loc[
    ens.horizon == 10,
    "date",
]


h20d = ens.loc[
    ens.horizon == 20,
    "date",
]


common_start = max(
    h10d.min(),
    h20d.min(),
)


common_end = min(
    h10d.max(),
    h20d.max(),
)


ens = ens[
    (ens.date >= common_start)
    &
    (ens.date <= common_end)
].copy()


raw = (
    pd.read_parquet(
        "data/"
        "universe_multihorizon.parquet"
    )
    .reset_index()
)


raw = raw.rename(
    columns={
        raw.columns[0]:
            "date"
    }
)


raw["date"] = pd.to_datetime(
    raw["date"]
)


def attach_returns(
    frame,
    h,
):

    r = raw[
        [
            "date",
            "ticker",
            f"stock_future_ret_h{h}",
            f"market_future_ret_h{h}",
        ]
    ].rename(
        columns={
            f"stock_future_ret_h{h}":
                "stock_ret",

            f"market_future_ret_h{h}":
                "market_ret",
        }
    )


    out = frame.merge(
        r,
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )


    return (
        out
        .dropna(
            subset=[
                "score",
                "stock_ret",
                "market_ret",
            ]
        )
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


h10 = ens[
    ens.horizon == 10
][
    [
        "date",
        "ticker",
        "score",
    ]
].copy()


h20 = ens[
    ens.horizon == 20
][
    [
        "date",
        "ticker",
        "score",
    ]
].copy()


signals = {
    "h10": {
        "h":
            10,

        "frame":
            attach_returns(
                h10,
                10,
            ),
    },

    "h20": {
        "h":
            20,

        "frame":
            attach_returns(
                h20,
                20,
            ),
    },
}


mix = h10.merge(
    h20,
    on=[
        "date",
        "ticker",
    ],
    suffixes=(
        "_h10",
        "_h20",
    ),
    validate=
        "one_to_one",
)


for w10 in [
    0.75,
    0.50,
    0.25,
]:

    w20 = (
        1
        - w10
    )


    z = mix[
        [
            "date",
            "ticker",
        ]
    ].copy()


    z["score"] = (
        w10
        * mix.score_h10
        +
        w20
        * mix.score_h20
    )


    signal_name = (
        f"h10_"
        f"{int(w10*100):02d}"
        f"_h20_"
        f"{int(w20*100):02d}"
    )


    signals[
        signal_name
    ] = {
        "h":
            10,

        "frame":
            attach_returns(
                z,
                10,
            ),
    }


for spec in signals.values():

    spec["days"] = {
        date:
            g

        for date, g
        in spec[
            "frame"
        ].groupby(
            "date",
            sort=True,
        )
    }


    spec[
        "rebalance_dates"
    ] = (
        sorted(
            spec[
                "days"
            ]
        )[
            ::spec["h"]
        ]
    )


def choose_target(
    day,
    prev_names,
    entry_k,
    exit_k,
    rng=None,
):

    if rng is None:

        ordered = (
            day
            .sort_values(
                "score",
                ascending=False,
            )
            .ticker
            .tolist()
        )

    else:

        ordered = (
            day
            .ticker
            .iloc[
                rng.permutation(
                    len(day)
                )
            ]
            .tolist()
        )


    rank = {
        ticker:
            i + 1

        for i, ticker
        in enumerate(
            ordered
        )
    }


    kept = sorted(
        [
            ticker

            for ticker
            in prev_names

            if (
                ticker
                in rank
                and
                rank[ticker]
                <= exit_k
            )
        ],

        key=lambda ticker:
            rank[ticker],
    )[
        :entry_k
    ]


    selected = list(
        kept
    )


    for ticker in ordered:

        if (
            ticker
            not in selected
        ):
            selected.append(
                ticker
            )


        if (
            len(selected)
            >= entry_k
        ):
            break


    if (
        len(selected)
        < entry_k
    ):
        return None


    return {
        ticker:
            1 / entry_k

        for ticker
        in selected
    }


def simulate(
    spec,
    entry_k,
    exit_k,
    slip_bps,
    random_seed=None,
    equal_weight=False,
):

    prev = {}
    rows = []


    rng = (
        np.random.default_rng(
            random_seed
        )

        if random_seed
        is not None

        else None
    )


    slip = (
        slip_bps
        / 10000
    )


    for date in spec[
        "rebalance_dates"
    ]:

        day = spec[
            "days"
        ][date]


        if equal_weight:

            tickers = (
                day.ticker
                .tolist()
            )


            target = {
                ticker:
                    1
                    / len(tickers)

                for ticker
                in tickers
            }

        else:

            target = (
                choose_target(
                    day,
                    set(prev),
                    entry_k,
                    exit_k,
                    rng,
                )
            )


        if not target:
            continue


        universe = (
            set(prev)
            |
            set(target)
        )


        buy = 0.0
        sell = 0.0


        for ticker in universe:

            delta = (
                target.get(
                    ticker,
                    0.0,
                )
                -
                prev.get(
                    ticker,
                    0.0,
                )
            )


            if delta > 0:
                buy += delta

            else:
                sell += -delta


        cost = (
            buy
            * (
                args.buy_fee
                + slip
            )
            +
            sell
            * (
                args.sell_fee
                + args.sell_tax
                + slip
            )
        )


        ret = dict(
            zip(
                day.ticker,
                day.stock_ret,
            )
        )


        gross = sum(
            weight
            * ret[ticker]

            for ticker, weight
            in target.items()
        )


        net = (
            (1 - cost)
            * (1 + gross)
            - 1
        )


        market = float(
            day.market_ret.iloc[0]
        )


        grown = {
            ticker:
                weight
                * (
                    1
                    + ret[ticker]
                )

            for ticker, weight
            in target.items()
        }


        total = sum(
            grown.values()
        )


        if total > 0:

            prev = {
                ticker:
                    value
                    / total

                for ticker, value
                in grown.items()
            }

        else:

            prev = target.copy()


        rows.append({
            "date":
                date,

            "gross_ret":
                gross,

            "net_ret":
                net,

            "market_ret":
                market,

            "turnover":
                (
                    buy
                    + sell
                )
                / 2,

            "transaction_cost":
                cost,

            "stocks":
                ",".join(
                    sorted(
                        target
                    )
                ),
        })


    return pd.DataFrame(
        rows
    )


def compound(
    x,
):

    return float(
        np.prod(
            1
            + np.asarray(x)
        )
        - 1
    )


def sharpe(
    x,
    h,
):

    x = (
        pd.Series(x)
        .dropna()
    )


    if (
        len(x) < 2
        or
        x.std(
            ddof=1
        ) == 0
    ):
        return np.nan


    return float(
        x.mean()
        / x.std(
            ddof=1
        )
        * math.sqrt(
            252
            / h
        )
    )


def drawdown(
    x,
):

    eq = (
        1
        + pd.Series(x)
    ).cumprod()


    return float(
        (
            eq
            / eq.cummax()
            - 1
        ).min()
    )


def cagr(
    total,
    start,
    end,
):

    years = (
        (
            pd.Timestamp(end)
            -
            pd.Timestamp(start)
        ).days
        / 365.25
    )


    if (
        years <= 0
        or
        total <= -1
    ):
        return np.nan


    return float(
        (
            1
            + total
        )
        ** (
            1
            / years
        )
        - 1
    )


def evaluate(
    strategy,
    ew,
    h,
):

    x = strategy.merge(
        ew[
            [
                "date",
                "net_ret",
            ]
        ].rename(
            columns={
                "net_ret":
                    "ew_net_ret"
            }
        ),
        on="date",
        validate="one_to_one",
    )


    st = compound(
        x.net_ret
    )


    gross = compound(
        x.gross_ret
    )


    bmk = compound(
        x.market_ret
    )


    ewret = compound(
        x.ew_net_ret
    )


    active = (
        x.net_ret
        - x.market_ret
    )


    yearly = x.copy()


    yearly["year"] = (
        pd.to_datetime(
            yearly.date
        )
        .dt.year
    )


    annual = []


    for year, g in yearly.groupby(
        "year"
    ):

        sr = compound(
            g.net_ret
        )


        br = compound(
            g.market_ret
        )


        annual.append(
            (
                int(year),
                sr - br,
            )
        )


    (
        worst_year,
        worst_alpha,
    ) = min(
        annual,
        key=lambda z:
            z[1],
    )


    return {
        "periods":
            len(x),

        "start_date":
            x.date.min(),

        "end_date":
            x.date.max(),

        "strategy_total_return":
            st,

        "strategy_cagr":
            cagr(
                st,
                x.date.min(),
                x.date.max(),
            ),

        "gross_total_return":
            gross,

        "cost_drag_total_return":
            gross - st,

        "0050_total_return":
            bmk,

        "0050_cagr":
            cagr(
                bmk,
                x.date.min(),
                x.date.max(),
            ),

        "ew_net_total_return":
            ewret,

        "ew_net_cagr":
            cagr(
                ewret,
                x.date.min(),
                x.date.max(),
            ),

        "alpha_simple_vs_0050":
            st - bmk,

        "relative_return_vs_0050":
            (
                (1 + st)
                / (1 + bmk)
                - 1
            ),

        "alpha_simple_vs_ew_net":
            st - ewret,

        "strategy_sharpe":
            sharpe(
                x.net_ret,
                h,
            ),

        "active_sharpe":
            sharpe(
                active,
                h,
            ),

        "period_max_drawdown":
            drawdown(
                x.net_ret
            ),

        "mean_turnover":
            float(
                x.turnover.mean()
            ),

        "total_transaction_cost_sum":
            float(
                x.transaction_cost.sum()
            ),

        "positive_alpha_years":
            sum(
                alpha > 0

                for _, alpha
                in annual
            ),

        "years":
            len(annual),

        "worst_year":
            worst_year,

        "worst_year_alpha":
            worst_alpha,
    }


rows = []

ew_cache = {}


for signal, spec in signals.items():

    for slip in SLIPS:

        ew = simulate(
            spec,
            1,
            1,
            slip,
            equal_weight=True,
        )


        ew_cache[
            (
                signal,
                slip,
            )
        ] = ew


        for (
            entry_k,
            exit_k,
        ) in ENTRY_EXIT:

            strat = simulate(
                spec,
                entry_k,
                exit_k,
                slip,
            )


            rows.append({
                "signal":
                    signal,

                "execution_horizon":
                    spec["h"],

                "entry_k":
                    entry_k,

                "exit_k":
                    exit_k,

                "slippage_bps":
                    slip,

                **evaluate(
                    strat,
                    ew,
                    spec["h"],
                ),
            })


leader = pd.DataFrame(
    rows
)


leader.to_csv(
    RESULTS
    / "v5_portfolio_leaderboard.csv",
    index=False,
)


robust = (
    leader[
        leader.slippage_bps
        == 10
    ]
    .sort_values(
        [
            "active_sharpe",
            "strategy_cagr",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


robust.to_csv(
    RESULTS
    / "v5_robust_10bps.csv",
    index=False,
)


# ------------------------------------------------------------
# Random baseline:
# 只對 10 bps 下最好的三個候選做 1000 次。
# 使用相同 universe / costs / holding / hysteresis。
# 唯一隨機化的是股票排名。
# ------------------------------------------------------------

rand_rows = []


for i, candidate in robust.head(
    3
).iterrows():

    spec = signals[
        candidate.signal
    ]


    ew = ew_cache[
        (
            candidate.signal,
            float(
                candidate.slippage_bps
            ),
        )
    ]


    totals = []
    sharpes = []


    for trial in range(
        args.random_trials
    ):

        random_strategy = (
            simulate(
                spec,

                int(
                    candidate.entry_k
                ),

                int(
                    candidate.exit_k
                ),

                float(
                    candidate.slippage_bps
                ),

                random_seed=
                    100000
                    + trial,
            )
        )


        m = evaluate(
            random_strategy,
            ew,
            spec["h"],
        )


        totals.append(
            m[
                "strategy_total_return"
            ]
        )


        sharpes.append(
            m[
                "active_sharpe"
            ]
        )


    totals = np.asarray(
        totals,
        float,
    )


    sharpes = np.asarray(
        sharpes,
        float,
    )


    rand_rows.append({
        "candidate_rank":
            i + 1,

        "signal":
            candidate.signal,

        "entry_k":
            int(
                candidate.entry_k
            ),

        "exit_k":
            int(
                candidate.exit_k
            ),

        "slippage_bps":
            float(
                candidate.slippage_bps
            ),

        "trials":
            args.random_trials,

        "model_total_return":
            float(
                candidate
                .strategy_total_return
            ),

        "random_total_median":
            float(
                np.median(
                    totals
                )
            ),

        "random_total_p95":
            float(
                np.quantile(
                    totals,
                    .95,
                )
            ),

        "model_total_percentile":
            float(
                np.mean(
                    totals
                    <= candidate
                    .strategy_total_return
                )
            ),

        "model_active_sharpe":
            float(
                candidate
                .active_sharpe
            ),

        "random_active_sharpe_p95":
            float(
                np.nanquantile(
                    sharpes,
                    .95,
                )
            ),

        "model_sharpe_percentile":
            float(
                np.nanmean(
                    sharpes
                    <= candidate
                    .active_sharpe
                )
            ),
    })


rand = pd.DataFrame(
    rand_rows
)


rand.to_csv(
    RESULTS
    / "v5_random_baseline.csv",
    index=False,
)


print()
print(
    "=" * 120
)

print(
    "LOBSTER V5 REALITY CHECK"
)

print(
    "=" * 120
)


print(
    f"Common OOS window: "
    f"{common_start.date()} "
    f"-> {common_end.date()}"
)


print(
    f"Seeds: {SEEDS}"
)


print(
    "Fee assumptions preserved "
    "from v4; "
    "slippage stress = "
    "0/5/10/20 bps"
)


cols = [
    "signal",
    "execution_horizon",
    "entry_k",
    "exit_k",
    "strategy_cagr",
    "0050_cagr",
    "ew_net_cagr",
    "active_sharpe",
    "period_max_drawdown",
    "mean_turnover",
    "positive_alpha_years",
    "worst_year",
    "worst_year_alpha",
]


print()
print(
    "TOP ROBUST STRATEGIES "
    "@ 10 BPS"
)


print(
    robust[
        cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print(
    "RANDOM BASELINE"
)


print(
    rand.to_string(
        index=False
    )
)


print()
print(
    "Saved:"
)

print(
    "results/"
    "v5_fold_metrics.csv"
)

print(
    "results/"
    "v5_portfolio_leaderboard.csv"
)

print(
    "results/"
    "v5_robust_10bps.csv"
)

print(
    "results/"
    "v5_random_baseline.csv"
)
