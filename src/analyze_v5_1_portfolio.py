from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd


p = argparse.ArgumentParser()

p.add_argument(
    "--seeds",
    default=(
        "1,7,42,123,2026,3407,8765,9999,"
        "31415,27182,16180,42424,"
        "13579,24680,55555,77777"
    ),
)

p.add_argument(
    "--random-trials",
    type=int,
    default=10000,
)

p.add_argument(
    "--bootstrap-trials",
    type=int,
    default=20000,
)

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

args = p.parse_args()


SEEDS = [
    int(x)
    for x in args.seeds.split(",")
]


HORIZONS = [
    10,
    20,
]


FEATURE_SETS = [
    "all",
    "no_absolute_returns",
    "no_relative",
    "no_market_context",
    "no_volatility",
    "no_trend",
]


TRAIN_WINDOWS = [
    "expanding",
    "3",
    "5",
    "8",
]


HYSTERESIS = [
    (5, 10),
    (10, 15),
    (10, 20),
    (15, 25),
]


SLIPPAGE_BPS = [
    0,
    5,
    10,
    20,
    30,
    50,
]


EXECUTION_DELAYS = [
    1,
    2,
]


RESULTS = Path(
    "results"
)

RESULTS.mkdir(
    exist_ok=True
)


def completed_run(
    h,
    fs,
    tw,
    seed,
):

    prefix = (
        f"v5_1_h{h}_"
        f"fs{fs}_"
        f"tw{tw}_"
        f"seed{seed}_*"
    )

    candidates = sorted(
        Path("runs").glob(
            prefix
        ),
        reverse=True,
    )

    for run in candidates:

        sf = run / "summary.json"
        pf = run / "predictions.parquet"

        if not (
            sf.exists()
            and pf.exists()
        ):
            continue

        try:
            summary = json.loads(
                sf.read_text()
            )
        except Exception:
            continue

        if int(
            summary.get(
                "years",
                0,
            )
        ) == 9:
            return run

    raise FileNotFoundError(
        f"missing complete run: "
        f"h={h} fs={fs} "
        f"tw={tw} seed={seed}"
    )


def load_ensemble(
    h,
    fs,
    tw,
):

    parts = []
    truth = None


    for seed in SEEDS:

        run = completed_run(
            h,
            fs,
            tw,
            seed,
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


        x["date"] = pd.to_datetime(
            x["date"]
        )


        if truth is None:

            truth = x[
                [
                    "date",
                    "ticker",
                    "future_alpha",
                ]
            ].copy()


        x["rank"] = (
            x
            .groupby(
                "date"
            )[
                "pred_alpha"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )


        parts.append(
            x[
                [
                    "date",
                    "ticker",
                    "rank",
                ]
            ]
        )


    all_pred = pd.concat(
        parts,
        ignore_index=True,
    )


    ens = (
        all_pred
        .groupby(
            [
                "date",
                "ticker",
            ],
            as_index=False,
        )
        .agg(
            score=(
                "rank",
                "mean",
            ),

            n_seeds=(
                "rank",
                "size",
            ),
        )
    )


    if (
        ens.n_seeds.min()
        != len(SEEDS)
    ):

        raise RuntimeError(
            f"incomplete ensemble "
            f"h={h} fs={fs} tw={tw}"
        )


    ens = ens.merge(
        truth,
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )


    return ens


def load_open_prices(
    tickers,
):

    prices = {}


    for ticker in tickers:

        f = (
            Path("data/raw")
            / f"{ticker}.parquet"
        )

        x = pd.read_parquet(
            f
        )

        x.index = pd.to_datetime(
            x.index
        )

        prices[ticker] = (
            x["Open"]
            .sort_index()
        )


    market_file = (
        Path("data/raw")
        / "0050.TW.parquet"
    )

    market = pd.read_parquet(
        market_file
    )

    market.index = pd.to_datetime(
        market.index
    )

    market_open = (
        market["Open"]
        .sort_index()
    )


    return (
        prices,
        market_open,
    )


def choose_target(
    day,
    previous,
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
            day.ticker
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
            in previous

            if (
                ticker in rank
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

        if ticker not in selected:

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
            1.0
            / entry_k

        for ticker
        in selected
    }


def price_return(
    series,
    start,
    end,
):

    try:

        a = float(
            series.loc[start]
        )

        b = float(
            series.loc[end]
        )

    except KeyError:

        raise RuntimeError(
            f"missing Open price "
            f"{start} -> {end}"
        )


    if (
        not np.isfinite(a)
        or
        not np.isfinite(b)
        or
        a <= 0
    ):

        raise RuntimeError(
            f"invalid Open price "
            f"{start} -> {end}"
        )


    return (
        b / a - 1
    )


def simulate(
    ensemble,
    horizon,
    prices,
    market_open,
    entry_k,
    exit_k,
    slip_bps,
    delay,
    random_seed=None,
    equal_weight=False,
):

    days = {
        date:
            g

        for date, g
        in ensemble.groupby(
            "date",
            sort=True,
        )
    }


    calendar = list(
        market_open.index
    )


    calendar_pos = {
        d:
            i

        for i, d
        in enumerate(
            calendar
        )
    }


    decision_dates = [
        d

        for d in sorted(
            days
        )

        if d in calendar_pos
    ]


    decision_dates = (
        decision_dates[
            ::horizon
        ]
    )


    if (
        len(decision_dates)
        < 3
    ):

        raise RuntimeError(
            "too few rebalance dates"
        )


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
        / 10000.0
    )


    previous_weights = {}

    rows = []


    for i in range(
        len(decision_dates) - 1
    ):

        decision = (
            decision_dates[i]
        )

        next_decision = (
            decision_dates[i + 1]
        )


        pos1 = (
            calendar_pos[decision]
            + delay
        )

        pos2 = (
            calendar_pos[
                next_decision
            ]
            + delay
        )


        if (
            pos1 >= len(calendar)
            or
            pos2 >= len(calendar)
        ):
            break


        execution = (
            calendar[pos1]
        )

        next_execution = (
            calendar[pos2]
        )


        day = days[
            decision
        ]


        if equal_weight:

            names = (
                day.ticker
                .tolist()
            )

            target = {
                t:
                    1.0
                    / len(names)

                for t in names
            }

        else:

            target = choose_target(
                day,
                set(
                    previous_weights
                ),
                entry_k,
                exit_k,
                rng,
            )


        if not target:
            continue


        union = (
            set(
                previous_weights
            )
            |
            set(target)
        )


        buy = 0.0
        sell = 0.0


        for ticker in union:

            delta = (
                target.get(
                    ticker,
                    0.0,
                )
                -
                previous_weights.get(
                    ticker,
                    0.0,
                )
            )


            if delta >= 0:

                buy += delta

            else:

                sell += -delta


        transaction_cost = (
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


        asset_returns = {}


        for ticker in target:

            asset_returns[ticker] = (
                price_return(
                    prices[ticker],
                    execution,
                    next_execution,
                )
            )


        gross_return = sum(
            target[ticker]
            * asset_returns[ticker]

            for ticker
            in target
        )


        net_return = (
            (1.0 - transaction_cost)
            * (1.0 + gross_return)
            - 1.0
        )


        market_return = (
            price_return(
                market_open,
                execution,
                next_execution,
            )
        )


        grown = {
            ticker:
                target[ticker]
                * (
                    1.0
                    + asset_returns[ticker]
                )

            for ticker
            in target
        }


        total = sum(
            grown.values()
        )


        previous_weights = {
            ticker:
                value / total

            for ticker, value
            in grown.items()
        }


        rows.append({
            "decision_date":
                decision,

            "execution_date":
                execution,

            "next_execution_date":
                next_execution,

            "gross_return":
                gross_return,

            "net_return":
                net_return,

            "market_return":
                market_return,

            "turnover":
                (
                    buy
                    + sell
                )
                / 2.0,

            "transaction_cost":
                transaction_cost,

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


def compound(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    return float(
        np.prod(
            1.0 + x
        )
        - 1.0
    )


def annualized_sharpe(
    x,
    horizon,
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
        /
        x.std(
            ddof=1
        )
        *
        math.sqrt(
            252.0
            / horizon
        )
    )


def rebalance_drawdown(
    x,
):

    equity = (
        1.0
        + pd.Series(x)
    ).cumprod()


    return float(
        (
            equity
            /
            equity.cummax()
            - 1.0
        ).min()
    )


def cagr(
    total_return,
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
        total_return <= -1
    ):

        return np.nan


    return float(
        (
            1.0
            + total_return
        )
        ** (
            1.0
            / years
        )
        - 1.0
    )


def evaluate(
    strategy,
    ew,
    horizon,
):

    x = strategy.merge(
        ew[
            [
                "execution_date",
                "net_return",
            ]
        ].rename(
            columns={
                "net_return":
                    "ew_net_return"
            }
        ),
        on="execution_date",
        how="inner",
        validate="one_to_one",
    )


    total = compound(
        x.net_return
    )

    gross = compound(
        x.gross_return
    )

    market = compound(
        x.market_return
    )

    ew_total = compound(
        x.ew_net_return
    )


    active = (
        x.net_return
        - x.market_return
    )


    start = (
        x.execution_date.min()
    )

    end = (
        x.next_execution_date.max()
    )


    y = x.copy()

    y["year"] = (
        pd.to_datetime(
            y.next_execution_date
        ).dt.year
    )


    annual = []


    for year, g in y.groupby(
        "year"
    ):

        strategy_y = compound(
            g.net_return
        )

        market_y = compound(
            g.market_return
        )

        annual.append(
            (
                int(year),
                strategy_y
                - market_y,
            )
        )


    worst_year, worst_alpha = min(
        annual,
        key=lambda z:
            z[1],
    )


    return {
        "periods":
            len(x),

        "start":
            start,

        "end":
            end,

        "total_return":
            total,

        "cagr":
            cagr(
                total,
                start,
                end,
            ),

        "gross_total_return":
            gross,

        "market_total_return":
            market,

        "market_cagr":
            cagr(
                market,
                start,
                end,
            ),

        "ew_net_total_return":
            ew_total,

        "ew_net_cagr":
            cagr(
                ew_total,
                start,
                end,
            ),

        "active_sharpe":
            annualized_sharpe(
                active,
                horizon,
            ),

        "strategy_sharpe":
            annualized_sharpe(
                x.net_return,
                horizon,
            ),

        "rebalance_max_drawdown":
            rebalance_drawdown(
                x.net_return
            ),

        "mean_turnover":
            float(
                x.turnover.mean()
            ),

        "cost_sum":
            float(
                x.transaction_cost.sum()
            ),

        "positive_alpha_years":
            int(
                sum(
                    a > 0
                    for _, a
                    in annual
                )
            ),

        "years":
            len(annual),

        "worst_year":
            worst_year,

        "worst_year_alpha":
            worst_alpha,
    }


def daily_ic(
    ensemble,
):

    values = []


    for _, g in ensemble.groupby(
        "date",
        sort=True,
    ):

        if len(g) < 10:
            continue

        v = g[
            "score"
        ].corr(
            g[
                "future_alpha"
            ],
            method="spearman",
        )

        if np.isfinite(v):

            values.append(
                float(v)
            )


    return np.asarray(
        values,
        dtype=float,
    )


def circular_block_bootstrap(
    values,
    block,
    trials,
    seed,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    n = len(x)

    blocks = int(
        math.ceil(
            n / block
        )
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )


    means = []

    batch = 250


    offsets = np.arange(
        block
    )


    for start in range(
        0,
        trials,
        batch,
    ):

        b = min(
            batch,
            trials - start,
        )


        starts = rng.integers(
            0,
            n,
            size=(
                b,
                blocks,
            ),
        )


        idx = (
            starts[
                :,
                :,
                None,
            ]
            +
            offsets[
                None,
                None,
                :,
            ]
        ) % n


        sampled = (
            x[idx]
            .reshape(
                b,
                -1,
            )[
                :,
                :n,
            ]
        )


        means.append(
            sampled.mean(
                axis=1
            )
        )


    means = np.concatenate(
        means
    )


    return {
        "observed_mean":
            float(
                x.mean()
            ),

        "ci_2_5":
            float(
                np.quantile(
                    means,
                    0.025,
                )
            ),

        "ci_97_5":
            float(
                np.quantile(
                    means,
                    0.975,
                )
            ),

        "bootstrap_p_mean_le_zero":
            float(
                np.mean(
                    means <= 0
                )
            ),

        "trials":
            trials,

        "block":
            block,
    }


# ------------------------------------------------------------
# Price cache
# ------------------------------------------------------------

sample = load_ensemble(
    20,
    "all",
    "expanding",
)

all_tickers = sorted(
    sample.ticker.unique()
)

prices, market_open = (
    load_open_prices(
        all_tickers
    )
)


portfolio_rows = []

primary_ensembles = {}


configs = (
    len(HORIZONS)
    * len(FEATURE_SETS)
    * len(TRAIN_WINDOWS)
)

done = 0


for horizon in HORIZONS:

    for feature_set in FEATURE_SETS:

        for train_window in TRAIN_WINDOWS:

            done += 1

            print(
                f"[{done:02d}/{configs}] "
                f"H={horizon} "
                f"FS={feature_set} "
                f"TW={train_window}",
                flush=True,
            )


            ensemble = load_ensemble(
                horizon,
                feature_set,
                train_window,
            )


            if (
                feature_set == "all"
                and
                train_window
                == "expanding"
            ):

                primary_ensembles[
                    horizon
                ] = ensemble


            ew_cache = {}


            for delay in EXECUTION_DELAYS:

                for slip in SLIPPAGE_BPS:

                    ew = simulate(
                        ensemble,
                        horizon,
                        prices,
                        market_open,
                        1,
                        1,
                        slip,
                        delay,
                        equal_weight=True,
                    )


                    ew_cache[
                        (
                            delay,
                            slip,
                        )
                    ] = ew


                    for (
                        entry_k,
                        exit_k,
                    ) in HYSTERESIS:

                        strategy = simulate(
                            ensemble,
                            horizon,
                            prices,
                            market_open,
                            entry_k,
                            exit_k,
                            slip,
                            delay,
                        )


                        metrics = evaluate(
                            strategy,
                            ew,
                            horizon,
                        )


                        portfolio_rows.append({
                            "horizon":
                                horizon,

                            "feature_set":
                                feature_set,

                            "train_window":
                                train_window,

                            "execution_delay":
                                delay,

                            "slippage_bps":
                                slip,

                            "entry_k":
                                entry_k,

                            "exit_k":
                                exit_k,

                            **metrics,
                        })


portfolio = pd.DataFrame(
    portfolio_rows
)


portfolio.to_csv(
    RESULTS
    / "v5_1_portfolio.csv",
    index=False,
)


# ------------------------------------------------------------
# Predeclared primary diagnostics.
#
# These are NOT selected from the overnight leaderboard:
#   H20 all / expanding / 10 -> 15
#   H10 all / expanding / 10 -> 15
# ------------------------------------------------------------

primary_rows = []


for horizon in [
    10,
    20,
]:

    e = primary_ensembles[
        horizon
    ]


    ic = daily_ic(
        e
    )


    boot = (
        circular_block_bootstrap(
            ic,
            horizon,
            args.bootstrap_trials,
            20260829 + horizon,
        )
    )


    boot["horizon"] = horizon

    primary_rows.append(
        boot
    )


pd.DataFrame(
    primary_rows
).to_csv(
    RESULTS
    / "v5_1_primary_ic_bootstrap.csv",
    index=False,
)


# ------------------------------------------------------------
# Random baseline for the predeclared H20 primary config.
# 10 bps, entry Top10, exit 15.
# Compare T+1 and T+2 execution.
# ------------------------------------------------------------

random_rows = []


primary_h20 = (
    primary_ensembles[20]
)


for delay in [
    1,
    2,
]:

    ew = simulate(
        primary_h20,
        20,
        prices,
        market_open,
        1,
        1,
        10,
        delay,
        equal_weight=True,
    )


    model_strategy = simulate(
        primary_h20,
        20,
        prices,
        market_open,
        10,
        15,
        10,
        delay,
    )


    model_metrics = evaluate(
        model_strategy,
        ew,
        20,
    )


    random_returns = []
    random_sharpes = []


    print(
        f"Random baseline H20 "
        f"delay={delay}: "
        f"{args.random_trials} trials",
        flush=True,
    )


    for trial in range(
        args.random_trials
    ):

        r = simulate(
            primary_h20,
            20,
            prices,
            market_open,
            10,
            15,
            10,
            delay,
            random_seed=(
                900000
                + delay
                * 100000
                + trial
            ),
        )


        m = evaluate(
            r,
            ew,
            20,
        )


        random_returns.append(
            m["total_return"]
        )

        random_sharpes.append(
            m["active_sharpe"]
        )


    rr = np.asarray(
        random_returns,
        dtype=float,
    )

    rs = np.asarray(
        random_sharpes,
        dtype=float,
    )


    random_rows.append({
        "horizon":
            20,

        "feature_set":
            "all",

        "train_window":
            "expanding",

        "entry_k":
            10,

        "exit_k":
            15,

        "slippage_bps":
            10,

        "execution_delay":
            delay,

        "trials":
            args.random_trials,

        "model_total_return":
            model_metrics[
                "total_return"
            ],

        "model_cagr":
            model_metrics[
                "cagr"
            ],

        "model_active_sharpe":
            model_metrics[
                "active_sharpe"
            ],

        "random_return_median":
            float(
                np.median(rr)
            ),

        "random_return_p95":
            float(
                np.quantile(
                    rr,
                    0.95,
                )
            ),

        "random_active_sharpe_p95":
            float(
                np.nanquantile(
                    rs,
                    0.95,
                )
            ),

        "model_return_percentile":
            float(
                np.mean(
                    rr
                    <= model_metrics[
                        "total_return"
                    ]
                )
            ),

        "model_sharpe_percentile":
            float(
                np.nanmean(
                    rs
                    <= model_metrics[
                        "active_sharpe"
                    ]
                )
            ),
    })


random_df = pd.DataFrame(
    random_rows
)


random_df.to_csv(
    RESULTS
    / "v5_1_primary_random.csv",
    index=False,
)


# ------------------------------------------------------------
# Human-readable summary
# ------------------------------------------------------------

robust_10 = (
    portfolio[
        (
            portfolio.slippage_bps
            == 10
        )
        &
        (
            portfolio.execution_delay
            == 1
        )
    ]
    .sort_values(
        [
            "active_sharpe",
            "cagr",
        ],
        ascending=False,
    )
)


primary_table = (
    portfolio[
        (
            portfolio.feature_set
            == "all"
        )
        &
        (
            portfolio.train_window
            == "expanding"
        )
        &
        (
            portfolio.entry_k
            == 10
        )
        &
        (
            portfolio.exit_k
            == 15
        )
        &
        (
            portfolio.slippage_bps
            == 10
        )
    ]
    .sort_values(
        [
            "horizon",
            "execution_delay",
        ]
    )
)


out = []

out.append(
    "=" * 120
)

out.append(
    "LOBSTER V5.1 REAL EXECUTION PORTFOLIO CHECK"
)

out.append(
    "=" * 120
)

out.append("")

out.append(
    "PREDECLARED ALL/EXPANDING 10->15 @ 10 BPS"
)

out.append(
    primary_table[
        [
            "horizon",
            "execution_delay",
            "cagr",
            "market_cagr",
            "ew_net_cagr",
            "active_sharpe",
            "rebalance_max_drawdown",
            "mean_turnover",
            "positive_alpha_years",
            "worst_year",
            "worst_year_alpha",
        ]
    ].to_string(
        index=False
    )
)

out.append("")

out.append(
    "TOP DIAGNOSTIC CONFIGS @ T+1 / 10 BPS "
    "(NOT A MODEL-SELECTION TABLE)"
)

out.append(
    robust_10[
        [
            "horizon",
            "feature_set",
            "train_window",
            "entry_k",
            "exit_k",
            "cagr",
            "market_cagr",
            "active_sharpe",
            "rebalance_max_drawdown",
            "mean_turnover",
            "worst_year",
            "worst_year_alpha",
        ]
    ]
    .head(25)
    .to_string(
        index=False
    )
)

out.append("")

out.append(
    "PRIMARY BLOCK BOOTSTRAP"
)

out.append(
    pd.DataFrame(
        primary_rows
    ).to_string(
        index=False
    )
)

out.append("")

out.append(
    "PRIMARY RANDOM BASELINE"
)

out.append(
    random_df.to_string(
        index=False
    )
)


text = "\n".join(
    out
)


(
    RESULTS
    / "v5_1_portfolio_summary.txt"
).write_text(
    text
    + "\n"
)


print(text)
