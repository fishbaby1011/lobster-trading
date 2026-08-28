from pathlib import Path
from datetime import datetime
import json

import pandas as pd


HORIZONS = [1, 3, 5, 10, 20]

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

BASE_FILE = DATA_DIR / "universe_dataset.parquet"
OUTPUT_FILE = DATA_DIR / "universe_multihorizon.parquet"
META_FILE = DATA_DIR / "universe_multihorizon_metadata.json"

RUN_DIR = (
    Path("runs")
    / (
        "build_multihorizon_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = RUN_DIR / "run.log"


def log(msg):
    text = (
        f"{datetime.now().isoformat(timespec='seconds')} | "
        f"{msg}"
    )

    print(text, flush=True)

    with LOG_FILE.open("a") as f:
        f.write(text + "\n")


log("Loading existing feature dataset")

base = pd.read_parquet(
    BASE_FILE
)

base.index = pd.to_datetime(
    base.index
)

base = base.sort_index()


with (
    DATA_DIR
    / "universe_metadata.json"
).open() as f:
    old_meta = json.load(f)


FEATURES = old_meta["features"]


market_file = (
    RAW_DIR / "0050.TW.parquet"
)

if not market_file.exists():
    raise FileNotFoundError(
        market_file
    )


market = pd.read_parquet(
    market_file
).sort_index()


all_data = []


tickers = sorted(
    base["ticker"].unique()
)


for i, ticker in enumerate(
    tickers,
    start=1,
):

    log(
        f"[{i:02d}/{len(tickers)}] "
        f"{ticker}"
    )

    stock_file = (
        RAW_DIR
        / f"{ticker}.parquet"
    )

    if not stock_file.exists():
        raise FileNotFoundError(
            stock_file
        )

    stock = pd.read_parquet(
        stock_file
    ).sort_index()

    stock.index = pd.to_datetime(
        stock.index
    )


    # 與原始 builder 保持一致：
    # benchmark 對齊個股交易日
    market_open = (
        market["Open"]
        .reindex(stock.index)
    )


    targets = pd.DataFrame(
        index=stock.index
    )


    date_series = pd.Series(
        stock.index,
        index=stock.index,
    )


    for h in HORIZONS:

        stock_entry = (
            stock["Open"]
            .shift(-1)
        )

        stock_exit = (
            stock["Open"]
            .shift(-(h + 1))
        )


        market_entry = (
            market_open
            .shift(-1)
        )

        market_exit = (
            market_open
            .shift(-(h + 1))
        )


        stock_ret = (
            stock_exit
            / stock_entry
            - 1
        )

        market_ret = (
            market_exit
            / market_entry
            - 1
        )


        targets[
            f"stock_future_ret_h{h}"
        ] = stock_ret


        targets[
            f"market_future_ret_h{h}"
        ] = market_ret


        targets[
            f"future_alpha_h{h}"
        ] = (
            stock_ret
            - market_ret
        )


        targets[
            f"label_end_date_h{h}"
        ] = (
            date_series
            .shift(-(h + 1))
        )


    current = (
        base[
            base["ticker"]
            == ticker
        ]
        .copy()
        .sort_index()
    )


    # 舊 h5 target 不再需要，避免混淆
    drop_old = [
        "stock_future_ret",
        "market_future_ret",
        "future_alpha",
        "label_end_date",
    ]


    current = current.drop(
        columns=[
            c
            for c in drop_old
            if c in current.columns
        ]
    )


    current = current.join(
        targets,
        how="left",
    )


    all_data.append(
        current
    )


dataset = (
    pd.concat(all_data)
    .sort_index()
)

dataset.index.name = "date"


dataset.to_parquet(
    OUTPUT_FILE
)


metadata = {
    "created_at":
        datetime.now().isoformat(),

    "horizons":
        HORIZONS,

    "features":
        FEATURES,

    "rows":
        len(dataset),

    "stocks":
        dataset[
            "ticker"
        ].nunique(),

    "date_min":
        str(dataset.index.min()),

    "date_max":
        str(dataset.index.max()),

    "source":
        str(BASE_FILE),

    "warning":
        (
            "Fixed present-day research universe. "
            "Survivorship/selection bias remains."
        ),
}


with META_FILE.open("w") as f:
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
log("MULTI-HORIZON BUILD COMPLETE")
log("=" * 70)

log(
    f"Rows     : {len(dataset):,}"
)

log(
    f"Stocks   : "
    f"{dataset['ticker'].nunique()}"
)

log(
    f"Horizons : {HORIZONS}"
)

log(
    f"Dataset  : {OUTPUT_FILE}"
)
