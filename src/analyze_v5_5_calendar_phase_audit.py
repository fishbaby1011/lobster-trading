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
    20,
]


FEATURE_SETS = [
    "all",
]


TRAIN_WINDOWS = [
    "expanding",
]


HYSTERESIS = [
    (10, 15),
]


SLIPPAGE_BPS = [
    0,
    10,
    20,
    30,
    50,
    100,
]


EXECUTION_DELAYS = [
    1,
    2,
    3,
    5,
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

    if (
        h != 20
        or fs != "all"
        or str(tw) != "expanding"
    ):
        raise ValueError(
            "v5.5 only accepts "
            "H20/all/expanding"
        )

    prefix = (
        f"v5_4b_repaired_h20_"
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
    phase=0,
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


    prediction_dates = sorted(
        days
    )


    if not prediction_dates:
        raise RuntimeError(
            "ensemble has no prediction dates"
        )


    prediction_start = (
        prediction_dates[0]
    )

    prediction_end = (
        prediction_dates[-1]
    )


    # IMPORTANT:
    #
    # Build the rebalance grid from the actual
    # market trading calendar, NOT from rows that
    # happen to exist in the prediction dataframe.
    #
    # Therefore a missing prediction date cannot
    # shift all subsequent rebalance phases.

    decision_calendar = [
        d

        for d in calendar

        if (
            d >= prediction_start
            and
            d <= prediction_end
        )
    ]


    if (
        phase < 0
        or phase >= horizon
    ):
        raise ValueError(
            f"invalid phase={phase} "
            f"for horizon={horizon}"
        )


    decision_dates = (
        decision_calendar[
            phase::horizon
        ]
    )


    missing_predictions = [
        d

        for d in decision_dates

        if d not in days
    ]


    if missing_predictions:

        preview = (
            missing_predictions[:10]
        )

        raise RuntimeError(
            "market-calendar scheduled "
            "decision date has no prediction: "
            f"{preview} "
            f"(total="
            f"{len(missing_predictions)})"
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

# ------------------------------------------------------------
# V5.5 CALENDAR-ANCHORED REPAIRED H20 PHASE AUDIT
#
# Frozen model:
#
#   H20
#   all features
#   expanding training window
#   repaired v5.4b 16-seed ensemble
#   Top10 -> Exit15
#
# Rebalance schedule:
#
#   anchored to 0050.TW market trading calendar
#
# NOT prediction-row calendar.
#
# Tests:
#
#   phases          = 0..19
#   execution delay = 1,2,3,5
#   slippage        = 0,10,20,30,50,100 bps
#
# Total:
#
#   20 * 4 * 6 = 480
# ------------------------------------------------------------


HORIZON = 20

FEATURE_SET = "all"

TRAIN_WINDOW = "expanding"

ENTRY_K = 10

EXIT_K = 15


print(
    "=" * 100
)

print(
    "LOBSTER V5.5 CALENDAR-ANCHORED "
    "REPAIRED H20 PHASE AUDIT"
)

print(
    "=" * 100
)

print(
    "Model         : repaired v5.4b "
    "H20 16-seed ensemble"
)

print(
    "Features      : all"
)

print(
    "Train window  : expanding"
)

print(
    "Portfolio     : Top10 -> Exit15"
)

print(
    "Calendar      : 0050.TW trading days"
)

print(
    f"Delays        : {EXECUTION_DELAYS}"
)

print(
    f"Slippage bps  : {SLIPPAGE_BPS}"
)

print()


# ============================================================
# Load repaired ensemble
# ============================================================

print(
    "Loading repaired H20 "
    "16-seed ensemble...",
    flush=True,
)


ensemble = load_ensemble(
    HORIZON,
    FEATURE_SET,
    TRAIN_WINDOW,
)


all_tickers = sorted(
    ensemble
    .ticker
    .unique()
)


prices, market_open = (
    load_open_prices(
        all_tickers
    )
)


# ============================================================
# Calendar completeness preflight
# ============================================================

prediction_dates = sorted(
    pd.to_datetime(
        ensemble["date"]
    ).unique()
)


prediction_dates = [
    pd.Timestamp(x)
    for x in prediction_dates
]


prediction_set = set(
    prediction_dates
)


prediction_start = (
    prediction_dates[0]
)

prediction_end = (
    prediction_dates[-1]
)


market_window = [
    d

    for d in market_open.index

    if (
        d >= prediction_start
        and
        d <= prediction_end
    )
]


missing_prediction_dates = [
    d

    for d in market_window

    if d not in prediction_set
]


extra_prediction_dates = [
    d

    for d in prediction_dates

    if d not in set(
        market_open.index
    )
]


print(
    "PRE-FLIGHT CALENDAR CHECK"
)

print(
    f"Prediction start : "
    f"{prediction_start.date()}"
)

print(
    f"Prediction end   : "
    f"{prediction_end.date()}"
)

print(
    f"Market days      : "
    f"{len(market_window)}"
)

print(
    f"Prediction days  : "
    f"{len(prediction_dates)}"
)

print(
    f"Missing preds    : "
    f"{len(missing_prediction_dates)}"
)

print(
    f"Extra preds      : "
    f"{len(extra_prediction_dates)}"
)


if missing_prediction_dates:

    print(
        "Missing market dates:"
    )

    for d in missing_prediction_dates[
        :30
    ]:
        print(
            "  ",
            d,
        )

    raise RuntimeError(
        "calendar completeness failed: "
        "market trading dates missing "
        "from repaired predictions"
    )


if extra_prediction_dates:

    raise RuntimeError(
        "prediction dates exist outside "
        "0050 trading calendar"
    )


print(
    "CALENDAR COMPLETENESS: PASS"
)

print()


# ============================================================
# Main 480 simulations
# ============================================================

rows = []


TOTAL = (
    HORIZON
    * len(EXECUTION_DELAYS)
    * len(SLIPPAGE_BPS)
)


job = 0


for phase in range(
    HORIZON
):

    for delay in EXECUTION_DELAYS:

        for slip in SLIPPAGE_BPS:

            job += 1

            print(
                f"[{job:03d}/{TOTAL}] "
                f"H=20 "
                f"phase={phase:02d} "
                f"delay={delay} "
                f"slip={slip:g}",
                flush=True,
            )


            ew = simulate(
                ensemble,
                HORIZON,
                prices,
                market_open,
                1,
                1,
                slip,
                delay,
                phase=phase,
                equal_weight=True,
            )


            strategy = simulate(
                ensemble,
                HORIZON,
                prices,
                market_open,
                ENTRY_K,
                EXIT_K,
                slip,
                delay,
                phase=phase,
            )


            metrics = evaluate(
                strategy,
                ew,
                HORIZON,
            )


            rows.append({
                "horizon":
                    HORIZON,

                "phase":
                    phase,

                "calendar_anchor":
                    "0050.TW",

                "model_source":
                    "v5_4_repaired",

                "feature_set":
                    FEATURE_SET,

                "train_window":
                    TRAIN_WINDOW,

                "entry_k":
                    ENTRY_K,

                "exit_k":
                    EXIT_K,

                "execution_delay":
                    delay,

                "slippage_bps":
                    slip,

                **metrics,
            })


df = pd.DataFrame(
    rows
)


df[
    "alpha_cagr"
] = (
    df["cagr"]
    -
    df["market_cagr"]
)


df.to_csv(
    RESULTS
    / "v5_5_calendar_phase_audit.csv",
    index=False,
)


# ============================================================
# Phase distribution summary
# ============================================================

summary_rows = []


for (
    delay,
    slip,
), g in df.groupby(
    [
        "execution_delay",
        "slippage_bps",
    ],
    sort=True,
):

    g = (
        g.sort_values(
            "phase"
        )
        .copy()
    )


    worst_sharpe = (
        g.loc[
            g[
                "active_sharpe"
            ].idxmin()
        ]
    )


    worst_alpha = (
        g.loc[
            g[
                "alpha_cagr"
            ].idxmin()
        ]
    )


    phase0 = (
        g[
            g.phase == 0
        ]
        .iloc[0]
    )


    summary_rows.append({

        "horizon":
            HORIZON,

        "execution_delay":
            delay,

        "slippage_bps":
            slip,

        "phases":
            int(
                g.phase.nunique()
            ),


        "cagr_min":
            g.cagr.min(),

        "cagr_median":
            g.cagr.median(),

        "cagr_max":
            g.cagr.max(),


        "market_cagr_median":
            g.market_cagr.median(),


        "alpha_cagr_min":
            g.alpha_cagr.min(),

        "alpha_cagr_median":
            g.alpha_cagr.median(),

        "alpha_cagr_max":
            g.alpha_cagr.max(),


        "beat_market_fraction":
            float(
                (
                    g.alpha_cagr
                    > 0
                ).mean()
            ),


        "active_sharpe_min":
            g.active_sharpe.min(),

        "active_sharpe_median":
            g.active_sharpe.median(),

        "active_sharpe_max":
            g.active_sharpe.max(),


        "positive_sharpe_fraction":
            float(
                (
                    g.active_sharpe
                    > 0
                ).mean()
            ),


        "worst_sharpe_phase":
            int(
                worst_sharpe.phase
            ),


        "worst_alpha_phase":
            int(
                worst_alpha.phase
            ),


        "rebalance_mdd_worst":
            g[
                "rebalance_max_drawdown"
            ].min(),


        "positive_alpha_years_min":
            int(
                g[
                    "positive_alpha_years"
                ].min()
            ),


        "worst_year_alpha_min":
            g[
                "worst_year_alpha"
            ].min(),


        "phase0_cagr":
            phase0.cagr,

        "phase0_active_sharpe":
            phase0.active_sharpe,

    })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    RESULTS
    / "v5_5_calendar_phase_summary.csv",
    index=False,
)


# ============================================================
# Predeclared primary:
#
# T+1 / 10 bps
# ============================================================

primary = (
    df[
        (
            df.execution_delay
            == 1
        )
        &
        (
            df.slippage_bps
            == 10
        )
    ]
    .sort_values(
        "phase"
    )
)


primary.to_csv(
    RESULTS
    / "v5_5_primary_all_phases.csv",
    index=False,
)


primary_summary = (
    summary[
        (
            summary.execution_delay
            == 1
        )
        &
        (
            summary.slippage_bps
            == 10
        )
    ]
    .iloc[0]
)


# ============================================================
# Diagnostic comparison against v5.3
#
# NOTE:
# This is NOT a pure causal decomposition.
#
# v5.3:
#   old ensemble + prediction calendar
#
# v5.5:
#   repaired ensemble + market calendar
#
# Therefore we only use this as a descriptive comparison.
# ============================================================

comparison = None


old_summary_path = (
    RESULTS
    / "v5_3_phase_summary.csv"
)


if old_summary_path.exists():

    old_summary = pd.read_csv(
        old_summary_path
    )


    old_primary = old_summary[
        (
            old_summary.horizon
            == 20
        )
        &
        (
            old_summary.execution_delay
            == 1
        )
        &
        (
            old_summary.slippage_bps
            == 10
        )
    ]


    if len(old_primary) == 1:

        old = (
            old_primary.iloc[0]
        )


        comparison = pd.DataFrame([
            {
                "metric":
                    "cagr_min",

                "v5_3":
                    old.cagr_min,

                "v5_5":
                    primary_summary.cagr_min,
            },

            {
                "metric":
                    "cagr_median",

                "v5_3":
                    old.cagr_median,

                "v5_5":
                    primary_summary.cagr_median,
            },

            {
                "metric":
                    "alpha_cagr_min",

                "v5_3":
                    old.alpha_cagr_min,

                "v5_5":
                    primary_summary.alpha_cagr_min,
            },

            {
                "metric":
                    "alpha_cagr_median",

                "v5_3":
                    old.alpha_cagr_median,

                "v5_5":
                    primary_summary.alpha_cagr_median,
            },

            {
                "metric":
                    "active_sharpe_min",

                "v5_3":
                    old.active_sharpe_min,

                "v5_5":
                    primary_summary.active_sharpe_min,
            },

            {
                "metric":
                    "active_sharpe_median",

                "v5_3":
                    old.active_sharpe_median,

                "v5_5":
                    primary_summary.active_sharpe_median,
            },

            {
                "metric":
                    "beat_market_fraction",

                "v5_3":
                    old.beat_market_fraction,

                "v5_5":
                    primary_summary.beat_market_fraction,
            },

            {
                "metric":
                    "positive_sharpe_fraction",

                "v5_3":
                    old.positive_sharpe_fraction,

                "v5_5":
                    primary_summary.positive_sharpe_fraction,
            },
        ])


        comparison[
            "delta"
        ] = (
            comparison["v5_5"]
            -
            comparison["v5_3"]
        )


        comparison.to_csv(
            RESULTS
            / "v5_5_vs_v5_3_primary.csv",
            index=False,
        )


# ============================================================
# Human-readable report
# ============================================================

out = []


out.append(
    "=" * 120
)

out.append(
    "LOBSTER V5.5 CALENDAR-ANCHORED "
    "REPAIRED H20 PHASE AUDIT"
)

out.append(
    "=" * 120
)

out.append("")

out.append(
    "FROZEN CONFIGURATION"
)

out.append(
    "H20 / all / expanding / repaired v5.4b "
    "16-seed ensemble / Top10 -> Exit15"
)

out.append(
    "Rebalance phases anchored to 0050.TW "
    "market trading calendar."
)

out.append("")

out.append(
    "CALENDAR PREFLIGHT"
)

out.append(
    f"prediction_start={prediction_start.date()}"
)

out.append(
    f"prediction_end={prediction_end.date()}"
)

out.append(
    f"market_days={len(market_window)}"
)

out.append(
    f"prediction_days={len(prediction_dates)}"
)

out.append(
    f"missing_prediction_days="
    f"{len(missing_prediction_dates)}"
)

out.append("")

out.append(
    "PRIMARY TEST: T+1 / 10 BPS / ALL 20 PHASES"
)

out.append(
    primary[
        [
            "phase",
            "cagr",
            "market_cagr",
            "alpha_cagr",
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
    "PRIMARY PHASE DISTRIBUTION"
)

out.append(
    pd.DataFrame(
        [primary_summary]
    )[
        [
            "horizon",
            "phases",

            "cagr_min",
            "cagr_median",
            "cagr_max",

            "market_cagr_median",

            "alpha_cagr_min",
            "alpha_cagr_median",
            "alpha_cagr_max",

            "beat_market_fraction",

            "active_sharpe_min",
            "active_sharpe_median",
            "active_sharpe_max",

            "positive_sharpe_fraction",

            "worst_sharpe_phase",
            "worst_alpha_phase",

            "rebalance_mdd_worst",

            "positive_alpha_years_min",
            "worst_year_alpha_min",

            "phase0_cagr",
            "phase0_active_sharpe",
        ]
    ].to_string(
        index=False
    )
)


if comparison is not None:

    out.append("")

    out.append(
        "DESCRIPTIVE V5.3 -> V5.5 COMPARISON"
    )

    out.append(
        "NOTE: model data and calendar definition "
        "both changed; this is not causal attribution."
    )

    out.append(
        comparison.to_string(
            index=False
        )
    )


out.append("")

out.append(
    "FULL DELAY / COST SUMMARY"
)

out.append(
    summary.to_string(
        index=False
    )
)


text = "\n".join(
    out
)


(
    RESULTS
    / "v5_5_calendar_audit_summary.txt"
).write_text(
    text
    + "\n"
)


print()

print(
    text
)

print()

print(
    "SAVED:"
)

print(
    "  results/v5_5_calendar_phase_audit.csv"
)

print(
    "  results/v5_5_calendar_phase_summary.csv"
)

print(
    "  results/v5_5_primary_all_phases.csv"
)

print(
    "  results/v5_5_vs_v5_3_primary.csv"
)

print(
    "  results/v5_5_calendar_audit_summary.txt"
)
