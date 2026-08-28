from pathlib import Path
from datetime import datetime
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf


START = "2012-01-01"
END = "2026-08-29"

HORIZON = 5
MARKET = "0050.TW"


TICKERS = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW",
    "2412.TW", "2303.TW", "3711.TW", "3034.TW", "2357.TW",

    "2379.TW", "3231.TW", "6669.TW", "3017.TW", "2327.TW",
    "2345.TW", "2395.TW", "3008.TW", "2408.TW", "3037.TW",

    "2002.TW", "1301.TW", "1303.TW", "1326.TW", "1101.TW",
    "1102.TW", "1216.TW", "2207.TW", "2603.TW", "2609.TW",

    "2615.TW", "2618.TW", "2880.TW", "2881.TW", "2882.TW",
    "2883.TW", "2884.TW", "2885.TW", "2886.TW", "2887.TW",

    "2891.TW", "2892.TW", "5880.TW", "5871.TW", "5876.TW",
    "2912.TW", "3045.TW", "4904.TW", "1590.TW", "9910.TW",
]


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RUN_DIR = Path("runs") / (
    "build_universe_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)

DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


LOG_FILE = RUN_DIR / "run.log"


def log(msg):
    text = (
        f"{datetime.now().isoformat(timespec='seconds')} | "
        f"{msg}"
    )

    print(text, flush=True)

    with LOG_FILE.open("a") as f:
        f.write(text + "\n")


def download(symbol):
    cache = RAW_DIR / f"{symbol}.parquet"

    if cache.exists():
        log(f"CACHE    {symbol}")
        return pd.read_parquet(cache)

    log(f"DOWNLOAD {symbol}")

    df = yf.download(
        symbol,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    df = (
        df[required]
        .copy()
        .sort_index()
    )

    df = df[
        ~df.index.duplicated(keep="first")
    ]

    df.to_parquet(cache)

    time.sleep(0.2)

    return df


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


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
    "range_ratio",

    "market_ret_1",
    "market_ret_5",
    "market_ret_20",

    "market_ma20_ratio",

    "excess_ret_1",
    "excess_ret_5",
    "excess_ret_20",
]


log("Loading benchmark 0050.TW")

market = download(MARKET)


def build_stock(ticker, stock):
    df = stock.copy()

    df["market_open"] = (
        market["Open"].reindex(df.index)
    )

    df["market_close"] = (
        market["Close"].reindex(df.index)
    )

    # ========================================================
    # Stock features
    # ========================================================

    df["ret_1"] = df["Close"].pct_change(
        1,
        fill_method=None,
    )

    df["ret_5"] = df["Close"].pct_change(
        5,
        fill_method=None,
    )

    df["ret_20"] = df["Close"].pct_change(
        20,
        fill_method=None,
    )

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

    for n in [5, 20, 60]:
        ma = (
            df["Close"]
            .rolling(n)
            .mean()
        )

        df[f"ma{n}_ratio"] = (
            df["Close"] / ma - 1
        )

    df["rsi_14"] = (
        rsi(df["Close"], 14)
        / 100
    )

    volume_ma20 = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["Volume"]
        / volume_ma20
    )

    df["range_ratio"] = (
        (
            df["High"]
            - df["Low"]
        )
        / df["Close"]
    )

    # ========================================================
    # Market features
    # ========================================================

    df["market_ret_1"] = (
        df["market_close"]
        .pct_change(
            1,
            fill_method=None,
        )
    )

    df["market_ret_5"] = (
        df["market_close"]
        .pct_change(
            5,
            fill_method=None,
        )
    )

    df["market_ret_20"] = (
        df["market_close"]
        .pct_change(
            20,
            fill_method=None,
        )
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
    # Relative strength
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
    # Future return
    #
    # T close   -> model sees data
    # T+1 open  -> enter
    # T+6 open  -> exit
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

    df["future_alpha"] = (
        df["stock_future_ret"]
        - df["market_future_ret"]
    )

    # ========================================================
    # Label end date
    #
    # 用來避免 Walk-forward boundary leakage
    # ========================================================

    date_series = pd.Series(
        df.index,
        index=df.index,
    )

    df["label_end_date"] = (
        date_series
        .shift(-(HORIZON + 1))
    )

    df["ticker"] = ticker

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

    keep = (
        [
            "ticker",
            "label_end_date",

            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
        + FEATURES
        + [
            "stock_future_ret",
            "market_future_ret",
            "future_alpha",
        ]
    )

    return df[keep].copy()


all_data = []

success = []
failed = []


for i, ticker in enumerate(
    TICKERS,
    start=1,
):
    log(
        f"[{i:02d}/{len(TICKERS)}] "
        f"{ticker}"
    )

    try:
        stock = download(ticker)

        if len(stock) < 250:
            raise RuntimeError(
                f"too few rows: {len(stock)}"
            )

        result = build_stock(
            ticker,
            stock,
        )

        all_data.append(result)

        success.append(ticker)

        log(
            f"OK       {ticker}: "
            f"{len(result):,} samples"
        )

    except Exception as exc:
        failed.append(ticker)

        log(
            f"FAILED   {ticker}: "
            f"{repr(exc)}"
        )


if not all_data:
    raise RuntimeError(
        "No usable ticker data"
    )


dataset = (
    pd.concat(all_data)
    .sort_index()
)

dataset.index.name = "date"


output = (
    DATA_DIR
    / "universe_dataset.parquet"
)

dataset.to_parquet(output)


metadata = {
    "created_at":
        datetime.now().isoformat(),

    "start":
        START,

    "end":
        END,

    "horizon":
        HORIZON,

    "market":
        MARKET,

    "features":
        FEATURES,

    "requested_tickers":
        TICKERS,

    "successful_tickers":
        success,

    "failed_tickers":
        failed,

    "rows":
        len(dataset),

    "stocks":
        len(success),

    "warning":
        (
            "Fixed present-day research universe. "
            "This is NOT a historical point-in-time constituent set; "
            "survivorship/selection bias remains."
        ),
}


with (
    DATA_DIR
    / "universe_metadata.json"
).open("w") as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


with (
    RUN_DIR
    / "metadata.json"
).open("w") as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


log("=" * 70)
log("BUILD COMPLETE")
log("=" * 70)

log(
    f"Rows     : {len(dataset):,}"
)

log(
    f"Stocks   : {len(success)}"
)

log(
    f"Failed   : {len(failed)}"
)

log(
    f"Date     : "
    f"{dataset.index.min()} "
    f"-> "
    f"{dataset.index.max()}"
)

log(
    f"Dataset  : {output}"
)

log(
    f"Run log  : {LOG_FILE}"
)

if failed:
    log(
        f"Failed tickers: {failed}"
    )
