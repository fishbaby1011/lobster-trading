from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

STOCK = "2330.TW"
MARKET = "0050.TW"

START = "2012-01-01"
END = "2026-08-29"

# 預測未來幾個交易日
HORIZON = 5


# ------------------------------------------------------------
# 交易成本
#
# 目前先作研究用的保守估值。
# 之後請改成你實際券商的費率。
# ------------------------------------------------------------

BUY_FEE = 0.001425
SELL_FEE = 0.001425

# 這裡分開放，方便之後修改
STOCK_SELL_TAX = 0.003
ETF_SELL_TAX = 0.001


# ------------------------------------------------------------
# Threshold search
#
# 只能在 Validation Set 上選 threshold。
# 絕對不能偷看 Test Set。
# ------------------------------------------------------------

THRESHOLD_MIN = 0.35
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.01


# ============================================================
# PATHS
# ============================================================

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


# ============================================================
# UTILS
# ============================================================

def safe_auc(y_true, prob):
    """
    如果某個 test fold 剛好只有一個 class，
    roc_auc_score 會失敗，所以做保護。
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, prob)


def max_drawdown(equity):
    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    return drawdown.min()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download(symbol):
    print(
        f"Downloading {symbol} ..."
    )

    df = yf.download(
        symbol,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
        df.columns = (
            df.columns
            .get_level_values(0)
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    df = df[required].copy()

    df = df.sort_index()

    return df


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():
    stock = download(STOCK)
    market = download(MARKET)

    df = stock.copy()

    # --------------------------------------------------------
    # Align market data
    # --------------------------------------------------------

    df["market_open"] = (
        market["Open"]
        .reindex(df.index)
    )

    df["market_close"] = (
        market["Close"]
        .reindex(df.index)
    )


    # ========================================================
    # STOCK FEATURES
    # ========================================================

    df["ret_1"] = (
        df["Close"]
        .pct_change(1)
    )

    df["ret_5"] = (
        df["Close"]
        .pct_change(5)
    )

    df["ret_20"] = (
        df["Close"]
        .pct_change(20)
    )


    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["vol_5"] = (
        df["ret_1"]
        .rolling(5)
        .std()
    )

    df["vol_20"] = (
        df["ret_1"]
        .rolling(20)
        .std()
    )


    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    for n in [5, 20, 60]:
        ma = (
            df["Close"]
            .rolling(n)
            .mean()
        )

        df[f"ma{n}_ratio"] = (
            df["Close"] / ma - 1
        )


    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = (
        calculate_rsi(
            df["Close"],
            14
        )
        / 100
    )


    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_ma20 = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["Volume"]
        / volume_ma20
    )


    # --------------------------------------------------------
    # Intraday range
    # --------------------------------------------------------

    df["range_ratio"] = (
        (
            df["High"]
            - df["Low"]
        )
        / df["Close"]
    )


    # ========================================================
    # MARKET FEATURES
    # ========================================================

    df["market_ret_1"] = (
        df["market_close"]
        .pct_change(1)
    )

    df["market_ret_5"] = (
        df["market_close"]
        .pct_change(5)
    )

    df["market_ret_20"] = (
        df["market_close"]
        .pct_change(20)
    )

    market_ma20 = (
        df["market_close"]
        .rolling(20)
        .mean()
    )

    df["market_ma20_ratio"] = (
        df["market_close"]
        / market_ma20
        - 1
    )


    # ========================================================
    # RELATIVE STRENGTH FEATURES
    # ========================================================

    df["excess_ret_1"] = (
        df["ret_1"]
        - df["market_ret_1"]
    )

    df["excess_ret_5"] = (
        df["ret_5"]
        - df["market_ret_5"]
    )

    df["excess_ret_20"] = (
        df["ret_20"]
        - df["market_ret_20"]
    )


    # ========================================================
    # REALISTIC FUTURE RETURN
    #
    # Feature 使用 day T 收盤以前資料。
    #
    # Decision：
    # day T 收盤後
    #
    # Entry：
    # day T+1 open
    #
    # Exit：
    # 5 個交易日之後的 open
    #
    # 避免「看到今天收盤再用今天收盤價成交」。
    # ========================================================

    stock_entry = (
        df["Open"]
        .shift(-1)
    )

    stock_exit = (
        df["Open"]
        .shift(-(HORIZON + 1))
    )

    market_entry = (
        df["market_open"]
        .shift(-1)
    )

    market_exit = (
        df["market_open"]
        .shift(-(HORIZON + 1))
    )


    df["stock_future_ret"] = (
        stock_exit
        / stock_entry
        - 1
    )

    df["market_future_ret"] = (
        market_exit
        / market_entry
        - 1
    )


    # ========================================================
    # ALPHA TARGET
    #
    # 1:
    # 2330 未來五日跑贏 0050
    #
    # 0:
    # 0050 未來五日跑贏 / 持平
    # ========================================================

    df["future_alpha"] = (
        df["stock_future_ret"]
        - df["market_future_ret"]
    )

    df["target"] = np.where(
        df["future_alpha"].notna(),

        (
            df["future_alpha"] > 0
        ).astype(int),

        np.nan,
    )


    # ========================================================
    # FEATURES
    # ========================================================

    features = [
        "ret_1",
        "ret_5",
        "ret_20",

        "vol_5",
        "vol_20",

        "ma5_ratio",
        "ma20_ratio",
        "ma60_ratio",

        "rsi_14",

        "volume_ratio",

        "range_ratio",

        "market_ret_1",
        "market_ret_5",
        "market_ret_20",

        "market_ma20_ratio",

        "excess_ret_1",
        "excess_ret_5",
        "excess_ret_20",
    ]


    # ========================================================
    # CLEAN
    # ========================================================

    df = df.dropna(
        subset=
        features
        + [
            "stock_future_ret",
            "market_future_ret",
            "future_alpha",
            "target",
        ]
    )

    df["target"] = (
        df["target"]
        .astype(int)
    )


    return df, features


# ============================================================
# MODELS
# ============================================================

def make_logistic_plain():
    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=3000,
            )
        ),
    ])


def make_logistic_balanced():
    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
            )
        ),
    ])


def make_xgboost():
    return XGBClassifier(
        n_estimators=500,

        max_depth=4,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=42,
    )


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold(
    y_true,
    prob
):
    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + 0.0001,
        THRESHOLD_STEP,
    )

    rows = []

    for threshold in thresholds:
        pred = (
            prob >= threshold
        ).astype(int)

        accuracy = (
            accuracy_score(
                y_true,
                pred
            )
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_true,
                pred
            )
        )

        exposure = (
            pred.mean()
        )

        rows.append({
            "threshold":
                float(threshold),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "stock_exposure":
                exposure,
        })


    scan = pd.DataFrame(rows)


    # --------------------------------------------------------
    # 先最大化 balanced accuracy。
    # 若一樣，再比較 accuracy。
    # --------------------------------------------------------

    best = (
        scan
        .sort_values(
            [
                "balanced_accuracy",
                "accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )


    return (
        float(best["threshold"]),
        scan
    )


# ============================================================
# ROTATION BACKTEST
#
# Predict 1 -> hold 2330
# Predict 0 -> hold 0050
# ============================================================

def run_rotation_backtest(
    frame,
    probabilities,
    threshold,
):
    bt = frame.copy()

    bt["prob_2330_outperform"] = (
        probabilities
    )


    # --------------------------------------------------------
    # 每 HORIZON 日做一次新決策
    #
    # 因為 return 本身是 5-day horizon，
    # 避免大量 overlapping positions。
    # --------------------------------------------------------

    bt = (
        bt
        .iloc[::HORIZON]
        .copy()
    )


    bt["signal"] = (
        bt["prob_2330_outperform"]
        >= threshold
    ).astype(int)


    bt["asset"] = np.where(
        bt["signal"] == 1,
        "2330",
        "0050",
    )


    # ========================================================
    # Gross return
    # ========================================================

    bt["gross_ret"] = np.where(
        bt["asset"] == "2330",

        bt["stock_future_ret"],

        bt["market_future_ret"],
    )


    # ========================================================
    # Trading cost
    #
    # Initial:
    # buy fee
    #
    # Same asset:
    # no rebalance cost
    #
    # Switch:
    # sell previous asset
    # + tax
    # + buy new asset
    # ========================================================

    costs = []

    previous_asset = None

    for asset in bt["asset"]:
        if previous_asset is None:

            cost = BUY_FEE

        elif asset == previous_asset:

            cost = 0.0

        else:

            if previous_asset == "2330":
                sell_tax = (
                    STOCK_SELL_TAX
                )
            else:
                sell_tax = (
                    ETF_SELL_TAX
                )

            cost = (
                SELL_FEE
                + sell_tax
                + BUY_FEE
            )

        costs.append(cost)

        previous_asset = asset


    bt["transaction_cost"] = (
        costs
    )


    # --------------------------------------------------------
    # Apply cost
    # --------------------------------------------------------

    bt["strategy_ret"] = (
        (
            1
            + bt["gross_ret"]
        )
        *
        (
            1
            - bt["transaction_cost"]
        )
        - 1
    )


    # ========================================================
    # Benchmark returns
    # ========================================================

    bt["2330_ret"] = (
        bt["stock_future_ret"]
    )

    bt["0050_ret"] = (
        bt["market_future_ret"]
    )


    # 初始買入手續費
    if len(bt) > 0:

        first_index = (
            bt.index[0]
        )

        bt.loc[
            first_index,
            "2330_ret"
        ] = (
            (
                1
                + bt.loc[
                    first_index,
                    "2330_ret"
                ]
            )
            *
            (
                1
                - BUY_FEE
            )
            - 1
        )

        bt.loc[
            first_index,
            "0050_ret"
        ] = (
            (
                1
                + bt.loc[
                    first_index,
                    "0050_ret"
                ]
            )
            *
            (
                1
                - BUY_FEE
            )
            - 1
        )


    # ========================================================
    # Equity
    # ========================================================

    bt["strategy_equity"] = (
        (
            1
            + bt["strategy_ret"]
        )
        .cumprod()
    )

    bt["2330_equity"] = (
        (
            1
            + bt["2330_ret"]
        )
        .cumprod()
    )

    bt["0050_equity"] = (
        (
            1
            + bt["0050_ret"]
        )
        .cumprod()
    )


    # ========================================================
    # Metrics
    # ========================================================

    strategy_total = (
        bt["strategy_equity"]
        .iloc[-1]
        - 1
    )

    stock_total = (
        bt["2330_equity"]
        .iloc[-1]
        - 1
    )

    market_total = (
        bt["0050_equity"]
        .iloc[-1]
        - 1
    )


    # --------------------------------------------------------
    # Active returns vs 0050
    # --------------------------------------------------------

    active_ret = (
        bt["strategy_ret"]
        - bt["0050_ret"]
    )


    if (
        len(active_ret) > 1
        and active_ret.std(ddof=1) > 0
    ):

        annual_factor = (
            252 / HORIZON
        )

        active_sharpe = (
            active_ret.mean()
            / active_ret.std(ddof=1)
            * math.sqrt(
                annual_factor
            )
        )

    else:
        active_sharpe = np.nan


    switches = (
        bt["asset"]
        != bt["asset"].shift()
    ).sum()

    # 第一筆不是 switch
    switches = max(
        int(switches) - 1,
        0
    )


    metrics = {
        "strategy_return":
            strategy_total,

        "2330_return":
            stock_total,

        "0050_return":
            market_total,

        "alpha_vs_0050":
            strategy_total
            - market_total,

        "alpha_vs_2330":
            strategy_total
            - stock_total,

        "max_drawdown":
            max_drawdown(
                bt["strategy_equity"]
            ),

        "active_sharpe":
            active_sharpe,

        "stock_exposure":
            (
                bt["asset"]
                .eq("2330")
                .mean()
            ),

        "switches":
            switches,

        "periods":
            len(bt),
    }


    return bt, metrics


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def print_model_metrics(
    name,
    y_true,
    prob,
    threshold=0.5,
):
    pred = (
        prob >= threshold
    ).astype(int)

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Threshold:",
        round(
            threshold,
            4
        )
    )

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_true,
                pred
            ),
            4
        )
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(
                y_true,
                pred
            ),
            4
        )
    )

    auc = (
        safe_auc(
            y_true,
            prob
        )
    )

    print(
        "ROC-AUC:",
        (
            round(auc, 4)
            if not np.isnan(auc)
            else "N/A"
        )
    )

    print()

    print(
        classification_report(
            y_true,
            pred,
            digits=4,
            zero_division=0,
        )
    )


# ============================================================
# WALK-FORWARD
#
# Test year Y:
#
# Train:
# <= Y-2
#
# Validation:
# Y-1
#
# Test:
# Y
#
# Example:
#
# Test 2020
# Train 2012~2018
# Valid 2019
# Test  2020
# ============================================================

def walk_forward(
    df,
    features,
    start_year=2018,
    end_year=2026,
):
    rows = []

    print()
    print()
    print("#" * 70)
    print("WALK-FORWARD TEST")
    print("#" * 70)


    for test_year in range(
        start_year,
        end_year + 1
    ):

        valid_year = (
            test_year - 1
        )


        train = df[
            df.index
            <
            f"{valid_year}-01-01"
        ].copy()


        valid = df[
            (
                df.index
                >=
                f"{valid_year}-01-01"
            )
            &
            (
                df.index
                <
                f"{test_year}-01-01"
            )
        ].copy()


        test = df[
            (
                df.index
                >=
                f"{test_year}-01-01"
            )
            &
            (
                df.index
                <
                f"{test_year + 1}-01-01"
            )
        ].copy()


        if (
            len(train) < 500
            or len(valid) < 50
            or len(test) < 20
        ):

            continue


        X_train = (
            train[features]
        )

        y_train = (
            train["target"]
        )


        X_valid = (
            valid[features]
        )

        y_valid = (
            valid["target"]
        )


        X_test = (
            test[features]
        )

        y_test = (
            test["target"]
        )


        # ----------------------------------------------------
        # Logistic model
        # ----------------------------------------------------

        model = (
            make_logistic_plain()
        )


        model.fit(
            X_train,
            y_train
        )


        valid_prob = (
            model
            .predict_proba(
                X_valid
            )[:, 1]
        )


        threshold, _ = (
            choose_threshold(
                y_valid,
                valid_prob
            )
        )


        test_prob = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )


        test_pred = (
            test_prob
            >= threshold
        ).astype(int)


        auc = (
            safe_auc(
                y_test,
                test_prob
            )
        )


        accuracy = (
            accuracy_score(
                y_test,
                test_pred
            )
        )


        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                test_pred
            )
        )


        # ----------------------------------------------------
        # Backtest
        # ----------------------------------------------------

        bt, metrics = (
            run_rotation_backtest(
                test,
                test_prob,
                threshold,
            )
        )


        row = {
            "year":
                test_year,

            "train_samples":
                len(train),

            "validation_samples":
                len(valid),

            "test_samples":
                len(test),

            "threshold":
                threshold,

            "test_positive_rate":
                y_test.mean(),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "auc":
                auc,

            **metrics,
        }


        rows.append(row)


        print()
        print(
            f"{test_year}: "
            f"AUC={auc:.4f}  "
            f"BalAcc={balanced_accuracy:.4f}  "
            f"Threshold={threshold:.2f}  "
            f"Strategy={metrics['strategy_return']:.2%}  "
            f"0050={metrics['0050_return']:.2%}  "
            f"Alpha={metrics['alpha_vs_0050']:.2%}"
        )


    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Dataset
    # ========================================================

    df, features = (
        build_dataset()
    )


    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        "Samples:",
        len(df)
    )

    print(
        "Features:",
        len(features)
    )

    print()
    print(
        "Target:"
    )

    print(
        "1 = 2330 outperforms 0050"
    )

    print(
        "0 = 0050 outperforms / tie"
    )

    print()

    print(
        df["target"]
        .value_counts(
            normalize=True
        )
    )


    # ========================================================
    # Main split
    # ========================================================

    train = df[
        df.index
        <
        "2024-01-01"
    ].copy()


    valid = df[
        (
            df.index
            >=
            "2024-01-01"
        )
        &
        (
            df.index
            <
            "2026-01-01"
        )
    ].copy()


    test = df[
        df.index
        >=
        "2026-01-01"
    ].copy()


    print()
    print("=" * 70)
    print("MAIN TIME SPLIT")
    print("=" * 70)


    print(
        "Train:",
        train.index.min(),
        "->",
        train.index.max(),
        len(train)
    )


    print(
        "Valid:",
        valid.index.min(),
        "->",
        valid.index.max(),
        len(valid)
    )


    print(
        "Test:",
        test.index.min(),
        "->",
        test.index.max(),
        len(test)
    )


    X_train = (
        train[features]
    )

    y_train = (
        train["target"]
    )


    X_valid = (
        valid[features]
    )

    y_valid = (
        valid["target"]
    )


    X_test = (
        test[features]
    )

    y_test = (
        test["target"]
    )


    # ========================================================
    # DUMMY BASELINE
    #
    # 永遠猜 2330 會跑贏 0050
    # ========================================================

    always_2330 = (
        np.ones(
            len(y_test),
            dtype=int
        )
    )


    print()
    print("=" * 70)
    print("DUMMY BASELINE")
    print("=" * 70)


    print(
        "Test positive rate:",
        round(
            y_test.mean(),
            4
        )
    )


    print(
        "Always-2330 Accuracy:",
        round(
            accuracy_score(
                y_test,
                always_2330
            ),
            4
        )
    )


    print(
        "Always-2330 Balanced Accuracy:",
        round(
            balanced_accuracy_score(
                y_test,
                always_2330
            ),
            4
        )
    )


    # ========================================================
    # MODEL 1
    # Logistic plain
    # ========================================================

    logistic_plain = (
        make_logistic_plain()
    )


    logistic_plain.fit(
        X_train,
        y_train
    )


    plain_valid_prob = (
        logistic_plain
        .predict_proba(
            X_valid
        )[:, 1]
    )


    plain_test_prob = (
        logistic_plain
        .predict_proba(
            X_test
        )[:, 1]
    )


    print_model_metrics(
        "LOGISTIC PLAIN @ 0.50",
        y_test,
        plain_test_prob,
        0.5,
    )


    # ========================================================
    # MODEL 2
    # Logistic balanced
    # ========================================================

    logistic_balanced = (
        make_logistic_balanced()
    )


    logistic_balanced.fit(
        X_train,
        y_train
    )


    balanced_test_prob = (
        logistic_balanced
        .predict_proba(
            X_test
        )[:, 1]
    )


    print_model_metrics(
        "LOGISTIC BALANCED @ 0.50",
        y_test,
        balanced_test_prob,
        0.5,
    )


    # ========================================================
    # MODEL 3
    # XGBoost
    # ========================================================

    xgb = (
        make_xgboost()
    )


    xgb.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        verbose=False,
    )


    xgb_test_prob = (
        xgb
        .predict_proba(
            X_test
        )[:, 1]
    )


    print_model_metrics(
        "XGBOOST @ 0.50",
        y_test,
        xgb_test_prob,
        0.5,
    )


    # ========================================================
    # VALIDATION THRESHOLD SEARCH
    #
    # 只用 2024~2025。
    # ========================================================

    best_threshold, threshold_scan = (
        choose_threshold(
            y_valid,
            plain_valid_prob
        )
    )


    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 70)


    print(
        "Selected threshold:",
        round(
            best_threshold,
            4
        )
    )


    best_row = (
        threshold_scan[
            np.isclose(
                threshold_scan[
                    "threshold"
                ],
                best_threshold
            )
        ]
        .iloc[0]
    )


    print(
        "Validation Balanced Accuracy:",
        round(
            best_row[
                "balanced_accuracy"
            ],
            4
        )
    )


    print(
        "Validation Accuracy:",
        round(
            best_row[
                "accuracy"
            ],
            4
        )
    )


    print(
        "Validation Stock Exposure:",
        f"{best_row['stock_exposure']:.2%}"
    )


    # ========================================================
    # FINAL TEST USING LOCKED THRESHOLD
    # ========================================================

    print_model_metrics(
        "LOGISTIC PLAIN - VALIDATION SELECTED THRESHOLD",
        y_test,
        plain_test_prob,
        best_threshold,
    )


    # ========================================================
    # TEST BACKTEST
    # ========================================================

    bt, metrics = (
        run_rotation_backtest(
            test,
            plain_test_prob,
            best_threshold,
        )
    )


    print()
    print("=" * 70)
    print("2026 OUT-OF-SAMPLE ROTATION BACKTEST")
    print("=" * 70)


    print(
        "Threshold:",
        round(
            best_threshold,
            4
        )
    )


    print(
        "Strategy Return:",
        f"{metrics['strategy_return']:.2%}"
    )


    print(
        "2330 Return:",
        f"{metrics['2330_return']:.2%}"
    )


    print(
        "0050 Return:",
        f"{metrics['0050_return']:.2%}"
    )


    print(
        "Alpha vs 0050:",
        f"{metrics['alpha_vs_0050']:.2%}"
    )


    print(
        "Alpha vs 2330:",
        f"{metrics['alpha_vs_2330']:.2%}"
    )


    print(
        "Max Drawdown:",
        f"{metrics['max_drawdown']:.2%}"
    )


    print(
        "Active Sharpe:",
        (
            round(
                metrics[
                    "active_sharpe"
                ],
                4
            )
            if not np.isnan(
                metrics[
                    "active_sharpe"
                ]
            )
            else "N/A"
        )
    )


    print(
        "2330 Exposure:",
        f"{metrics['stock_exposure']:.2%}"
    )


    print(
        "Switches:",
        metrics["switches"]
    )


    print(
        "Periods:",
        metrics["periods"]
    )


    # ========================================================
    # WALK-FORWARD
    # ========================================================

    walk_forward_df = (
        walk_forward(
            df,
            features,
            start_year=2018,
            end_year=2026,
        )
    )


    # ========================================================
    # WALK-FORWARD SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)


    if len(walk_forward_df) > 0:

        display_columns = [
            "year",
            "threshold",
            "auc",
            "balanced_accuracy",
            "strategy_return",
            "0050_return",
            "alpha_vs_0050",
            "max_drawdown",
            "active_sharpe",
            "stock_exposure",
        ]


        print(
            walk_forward_df[
                display_columns
            ].to_string(
                index=False
            )
        )


        print()

        print(
            "Mean AUC:",
            round(
                walk_forward_df[
                    "auc"
                ].mean(),
                4
            )
        )


        print(
            "Median AUC:",
            round(
                walk_forward_df[
                    "auc"
                ].median(),
                4
            )
        )


        print(
            "Mean Balanced Accuracy:",
            round(
                walk_forward_df[
                    "balanced_accuracy"
                ].mean(),
                4
            )
        )


        positive_alpha_years = (
            (
                walk_forward_df[
                    "alpha_vs_0050"
                ]
                > 0
            )
            .sum()
        )


        print(
            "Positive Alpha Years:",
            f"{positive_alpha_years}"
            f"/{len(walk_forward_df)}"
        )


    # ========================================================
    # SAVE DATASET
    # ========================================================

    df.to_parquet(
        "data/2330_alpha_dataset.parquet"
    )


    # ========================================================
    # SAVE THRESHOLD SCAN
    # ========================================================

    threshold_scan.to_csv(
        "results/threshold_scan.csv",
        index=False,
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    test_predictions = (
        test.copy()
    )


    test_predictions[
        "logistic_plain_prob"
    ] = plain_test_prob


    test_predictions[
        "logistic_balanced_prob"
    ] = balanced_test_prob


    test_predictions[
        "xgb_prob"
    ] = xgb_test_prob


    test_predictions[
        "selected_threshold"
    ] = best_threshold


    test_predictions[
        "prediction"
    ] = (
        plain_test_prob
        >= best_threshold
    ).astype(int)


    test_predictions.to_csv(
        "results/test_predictions_v2.csv"
    )


    # ========================================================
    # SAVE BACKTEST
    # ========================================================

    bt.to_csv(
        "results/test_backtest_v2.csv"
    )


    # ========================================================
    # SAVE WALK-FORWARD
    # ========================================================

    walk_forward_df.to_csv(
        "results/walk_forward.csv",
        index=False,
    )


    # ========================================================
    # SAVE MODELS
    # ========================================================

    joblib.dump(
        logistic_plain,
        "models/logistic_plain_v2.joblib",
    )


    joblib.dump(
        logistic_balanced,
        "models/logistic_balanced_v2.joblib",
    )


    xgb.save_model(
        "models/xgb_v2.json"
    )


    # ========================================================
    # EQUITY CURVE
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )


    plt.plot(
        bt.index,
        bt["strategy_equity"],
        label="Lobster Strategy",
    )


    plt.plot(
        bt.index,
        bt["2330_equity"],
        label="2330",
    )


    plt.plot(
        bt.index,
        bt["0050_equity"],
        label="0050",
    )


    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Equity"
    )

    plt.title(
        "2026 Out-of-Sample Rotation Strategy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plt.savefig(
        "results/equity_curve_v2.png",
        dpi=150,
    )


    plt.close()


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)


    files = [
        "data/2330_alpha_dataset.parquet",
        "results/threshold_scan.csv",
        "results/test_predictions_v2.csv",
        "results/test_backtest_v2.csv",
        "results/walk_forward.csv",
        "results/equity_curve_v2.png",
        "models/logistic_plain_v2.joblib",
        "models/logistic_balanced_v2.joblib",
        "models/xgb_v2.json",
    ]


    for file in files:
        print(file)


if __name__ == "__main__":
    main()
