from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
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

HORIZON = 5

# 第一版先使用保守交易成本
BUY_FEE = 0.001425
SELL_FEE = 0.001425
SELL_TAX = 0.003

ROUND_TRIP_COST = (
    BUY_FEE
    + SELL_FEE
    + SELL_TAX
)


# ============================================================
# DOWNLOAD
# ============================================================

def download(symbol):
    print(f"Downloading {symbol} ...")

    df = yf.download(
        symbol,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
    )

    # yfinance 新版有時會回傳 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()


# ============================================================
# RSI
# ============================================================

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

    return 100 - (100 / (1 + rs))


# ============================================================
# DATA
# ============================================================

stock = download(STOCK)
market = download(MARKET)

df = stock.copy()

df["market_close"] = market["Close"].reindex(df.index)


# ============================================================
# FEATURES
# ============================================================

# Returns
df["ret_1"] = df["Close"].pct_change(1)
df["ret_5"] = df["Close"].pct_change(5)
df["ret_20"] = df["Close"].pct_change(20)

# Volatility
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

# Moving averages
for n in [5, 20, 60]:
    ma = (
        df["Close"]
        .rolling(n)
        .mean()
    )

    df[f"ma{n}_ratio"] = (
        df["Close"] / ma - 1
    )

# RSI
df["rsi_14"] = (
    calculate_rsi(
        df["Close"],
        14
    ) / 100
)

# Volume
volume_ma20 = (
    df["Volume"]
    .rolling(20)
    .mean()
)

df["volume_ratio"] = (
    df["Volume"]
    / volume_ma20
)

# Market features
df["market_ret_1"] = (
    df["market_close"]
    .pct_change(1)
)

df["market_ret_5"] = (
    df["market_close"]
    .pct_change(5)
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


FEATURES = [
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

    "market_ret_1",
    "market_ret_5",
    "market_ma20_ratio",
]


# ============================================================
# TARGET
# ============================================================

df["future_ret_5"] = (
    df["Close"].shift(-HORIZON)
    / df["Close"]
    - 1
)

df["target"] = np.where(
    df["future_ret_5"].notna(),
    (
        df["future_ret_5"] > 0
    ).astype(int),
    np.nan,
)


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=FEATURES
    + [
        "future_ret_5",
        "target",
    ]
)

df["target"] = (
    df["target"]
    .astype(int)
)

print()
print("=" * 60)
print("DATASET")
print("=" * 60)

print("Samples:", len(df))

print()
print("Target distribution:")
print(
    df["target"]
    .value_counts(normalize=True)
)


# ============================================================
# TIME SPLIT
# ============================================================

train = df[
    df.index < "2024-01-01"
].copy()

valid = df[
    (df.index >= "2024-01-01")
    &
    (df.index < "2026-01-01")
].copy()

test = df[
    df.index >= "2026-01-01"
].copy()


print()
print("=" * 60)
print("TIME SPLIT")
print("=" * 60)

print(
    "Train:",
    train.index.min(),
    "->",
    train.index.max(),
    len(train),
)

print(
    "Valid:",
    valid.index.min(),
    "->",
    valid.index.max(),
    len(valid),
)

print(
    "Test :",
    test.index.min(),
    "->",
    test.index.max(),
    len(test),
)


X_train = train[FEATURES]
y_train = train["target"]

X_valid = valid[FEATURES]
y_valid = valid["target"]

X_test = test[FEATURES]
y_test = test["target"]


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

print()
print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )
    ),
])

logistic.fit(
    X_train,
    y_train,
)

logistic_prob = (
    logistic
    .predict_proba(X_test)[:, 1]
)

logistic_pred = (
    logistic_prob >= 0.5
).astype(int)


print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            logistic_pred,
        ),
        4,
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            logistic_prob,
        ),
        4,
    )
)


# ============================================================
# MODEL 2: XGBOOST
# ============================================================

print()
print("=" * 60)
print("XGBOOST")
print("=" * 60)

model = XGBClassifier(
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


model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    verbose=False,
)


prob = (
    model
    .predict_proba(X_test)[:, 1]
)

pred = (
    prob >= 0.5
).astype(int)


print(
    "Accuracy:",
    round(
        accuracy_score(
            y_test,
            pred,
        ),
        4,
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            prob,
        ),
        4,
    )
)

print()
print(
    classification_report(
        y_test,
        pred,
        digits=4,
    )
)


# ============================================================
# BACKTEST
# ============================================================

test["prob_up"] = prob

# 因為 target 是未來五日，
# 第一版每五個交易日重新做一次決策，
# 避免 return window 大量重疊。
bt = (
    test
    .iloc[::HORIZON]
    .copy()
)


THRESHOLD = 0.55


bt["signal"] = (
    bt["prob_up"]
    >= THRESHOLD
).astype(int)


bt["strategy_ret"] = np.where(
    bt["signal"] == 1,

    (
        bt["future_ret_5"]
        - ROUND_TRIP_COST
    ),

    0.0,
)


bt["buyhold_ret"] = (
    bt["future_ret_5"]
)


bt["strategy_equity"] = (
    1 + bt["strategy_ret"]
).cumprod()


bt["buyhold_equity"] = (
    1 + bt["buyhold_ret"]
).cumprod()


# ============================================================
# METRICS
# ============================================================

def max_drawdown(equity):
    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    return drawdown.min()


strategy_return = (
    bt["strategy_equity"].iloc[-1]
    - 1
)

buyhold_return = (
    bt["buyhold_equity"].iloc[-1]
    - 1
)


print()
print("=" * 60)
print("2026 OUT-OF-SAMPLE BACKTEST")
print("=" * 60)


print(
    "Strategy Return:",
    f"{strategy_return:.2%}"
)


print(
    "2330 Buy/Hold:",
    f"{buyhold_return:.2%}"
)


print(
    "Max Drawdown:",
    f"{max_drawdown(bt['strategy_equity']):.2%}"
)


print(
    "Trades:",
    int(
        bt["signal"].sum()
    )
)


print(
    "Exposure:",
    f"{bt['signal'].mean():.2%}"
)


# ============================================================
# SAVE
# ============================================================

Path("data").mkdir(
    exist_ok=True
)

Path("results").mkdir(
    exist_ok=True
)

Path("models").mkdir(
    exist_ok=True
)


df.to_parquet(
    "data/2330_dataset.parquet"
)


bt.to_csv(
    "results/baseline_backtest.csv"
)


model.save_model(
    "models/xgb_2330.json"
)


print()
print("=" * 60)
print("SAVED")
print("=" * 60)

print(
    "data/2330_dataset.parquet"
)

print(
    "results/baseline_backtest.csv"
)

print(
    "models/xgb_2330.json"
)
