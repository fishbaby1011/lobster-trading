from pathlib import Path
import json
import shutil
import time

import numpy as np
import pandas as pd
import requests


RAW = Path("data/raw")
BACKUP = Path("data/raw_backup_20250801")
RESULTS = Path("results")

BACKUP.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


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


BEFORE = pd.Timestamp("2025-07-31")
TARGET = pd.Timestamp("2025-08-01")
AFTER = pd.Timestamp("2025-08-04")


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 Chrome/131 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.twse.com.tw/",
})


def clean_number(x):

    s = str(x).strip()

    s = (
        s.replace(",", "")
        .replace("\u3000", "")
    )

    if s in {
        "",
        "--",
        "---",
        "nan",
    }:
        return np.nan

    # TWSE sometimes embeds HTML markup.
    if "<" in s:
        import re
        s = re.sub(
            r"<[^>]+>",
            "",
            s,
        )

    try:
        return float(s)

    except ValueError:
        return np.nan


def find_quote_table(j):

    # Newer TWSE response format.
    if isinstance(j, dict) and "tables" in j:

        for table in j["tables"]:

            fields = table.get(
                "fields",
                [],
            )

            data = table.get(
                "data",
                [],
            )

            if (
                "證券代號" in fields
                and "開盤價" in fields
                and "收盤價" in fields
            ):
                return fields, data


    # Older exchangeReport format:
    # fields9 / data9 etc.
    if isinstance(j, dict):

        for key, fields in j.items():

            if not str(key).startswith(
                "fields"
            ):
                continue

            if not isinstance(
                fields,
                list,
            ):
                continue

            if (
                "證券代號" not in fields
                or "開盤價" not in fields
                or "收盤價" not in fields
            ):
                continue

            suffix = str(key)[
                len("fields"):
            ]

            data_key = (
                "data"
                + suffix
            )

            data = j.get(
                data_key,
                [],
            )

            return fields, data


    raise RuntimeError(
        "Could not locate TWSE daily quote table"
    )


def fetch_market(date):

    date_str = (
        pd.Timestamp(date)
        .strftime("%Y%m%d")
    )

    endpoints = [
        "https://www.twse.com.tw/"
        "rwd/zh/afterTrading/MI_INDEX",

        "https://www.twse.com.tw/"
        "exchangeReport/MI_INDEX",
    ]


    last_error = None


    for attempt in range(6):

        endpoint = endpoints[
            attempt
            % len(endpoints)
        ]

        try:

            print(
                f"FETCH {date_str} "
                f"attempt={attempt + 1}",
                flush=True,
            )

            r = session.get(
                endpoint,
                params={
                    "response": "json",
                    "date": date_str,
                    "type": "ALLBUT0999",
                },
                timeout=45,
            )


            if r.status_code == 428:

                wait = min(
                    30 * (
                        2 ** attempt
                    ),
                    180,
                )

                print(
                    f"  TWSE 428; "
                    f"sleep {wait}s",
                    flush=True,
                )

                time.sleep(wait)
                continue


            r.raise_for_status()


            if not r.text.strip():

                raise RuntimeError(
                    "empty TWSE response"
                )


            j = r.json()


            fields, data = (
                find_quote_table(j)
            )


            frame = pd.DataFrame(
                data,
                columns=fields,
            )


            required = [
                "證券代號",
                "成交股數",
                "開盤價",
                "最高價",
                "最低價",
                "收盤價",
            ]


            for col in required:

                if col not in frame.columns:

                    raise RuntimeError(
                        f"missing field {col}"
                    )


            out = pd.DataFrame({
                "ticker":
                    frame[
                        "證券代號"
                    ]
                    .astype(str)
                    .str.strip(),

                "Volume":
                    frame[
                        "成交股數"
                    ].map(
                        clean_number
                    ),

                "Open":
                    frame[
                        "開盤價"
                    ].map(
                        clean_number
                    ),

                "High":
                    frame[
                        "最高價"
                    ].map(
                        clean_number
                    ),

                "Low":
                    frame[
                        "最低價"
                    ].map(
                        clean_number
                    ),

                "Close":
                    frame[
                        "收盤價"
                    ].map(
                        clean_number
                    ),
            })


            out = (
                out.drop_duplicates(
                    "ticker"
                )
                .set_index(
                    "ticker"
                )
            )


            print(
                f"  OK: "
                f"{len(out)} securities"
            )


            return out


        except Exception as e:

            last_error = e

            wait = min(
                15 * (
                    2 ** attempt
                ),
                120,
            )

            print(
                f"  ERROR: {e}"
            )

            print(
                f"  sleep {wait}s",
                flush=True,
            )

            time.sleep(wait)


    raise RuntimeError(
        f"Unable to download "
        f"{date_str}: "
        f"{last_error}"
    )


# --------------------------------------------------
# Only THREE market-wide data downloads.
# --------------------------------------------------

day_before = fetch_market(
    BEFORE
)

time.sleep(3)

day_target = fetch_market(
    TARGET
)

time.sleep(3)

day_after = fetch_market(
    AFTER
)


# --------------------------------------------------
# Validate all research tickers exist.
# --------------------------------------------------

codes = [
    t.split(".")[0]
    for t in TICKERS
]


for name, frame in [
    ("BEFORE", day_before),
    ("TARGET", day_target),
    ("AFTER", day_after),
]:

    missing = [
        code
        for code in codes
        if code not in frame.index
    ]

    if missing:

        raise RuntimeError(
            f"{name} TWSE missing: "
            f"{missing}"
        )


# --------------------------------------------------
# Repair / verify every ticker.
# --------------------------------------------------

manifest = []


for i, ticker in enumerate(
    TICKERS,
    1,
):

    code = ticker.split(".")[0]

    print(
        f"[{i:02d}/50] {ticker}",
        flush=True,
    )


    path = (
        RAW
        / f"{ticker}.parquet"
    )

    backup = (
        BACKUP
        / f"{ticker}.parquet"
    )


    df = pd.read_parquet(
        path
    )

    df.index = pd.to_datetime(
        df.index
    )


    if (
        BEFORE not in df.index
        or AFTER not in df.index
    ):
        raise RuntimeError(
            f"{ticker}: Yahoo reference "
            f"date missing"
        )


    # Create backup if we haven't already.
    if not backup.exists():

        shutil.copy2(
            path,
            backup,
        )


    yahoo_before = float(
        df.loc[
            BEFORE,
            "Close",
        ]
    )

    yahoo_after = float(
        df.loc[
            AFTER,
            "Close",
        ]
    )


    raw_before = float(
        day_before.loc[
            code,
            "Close",
        ]
    )

    raw_after = float(
        day_after.loc[
            code,
            "Close",
        ]
    )


    factor_before = (
        yahoo_before
        / raw_before
    )

    factor_after = (
        yahoo_after
        / raw_after
    )


    factor_diff = abs(
        factor_before
        - factor_after
    ) / max(
        (
            abs(factor_before)
            + abs(factor_after)
        )
        / 2,
        1e-12,
    )


    # Corporate action / mismatch guard.
    if factor_diff > 0.005:

        raise RuntimeError(
            f"{ticker}: adjustment "
            f"factor mismatch "
            f"{factor_before:.8f} vs "
            f"{factor_after:.8f}"
        )


    factor = (
        factor_before
        + factor_after
    ) / 2


    raw = day_target.loc[
        code
    ]


    repaired = {
        "Open":
            float(
                raw["Open"]
            )
            * factor,

        "High":
            float(
                raw["High"]
            )
            * factor,

        "Low":
            float(
                raw["Low"]
            )
            * factor,

        "Close":
            float(
                raw["Close"]
            )
            * factor,

        "Volume":
            float(
                raw["Volume"]
            ),
    }


    action = "verify"


    if TARGET not in df.index:

        row = pd.DataFrame(
            [repaired],
            index=[
                TARGET
            ],
        )


        # Preserve exact existing column order.
        row = row[
            df.columns
        ]


        df = pd.concat(
            [
                df,
                row,
            ]
        )


        df = (
            df.sort_index()
        )

        df = df[
            ~df.index.duplicated(
                keep="first"
            )
        ]


        df.to_parquet(
            path
        )


        action = "insert"


    else:

        # First 23 may already have been repaired.
        existing = float(
            df.loc[
                TARGET,
                "Open",
            ]
        )

        expected = repaired[
            "Open"
        ]

        rel = abs(
            existing - expected
        ) / max(
            abs(expected),
            1e-12,
        )

        if rel > 0.001:

            raise RuntimeError(
                f"{ticker}: existing "
                f"repair inconsistent "
                f"existing={existing} "
                f"expected={expected}"
            )


    manifest.append({
        "ticker":
            ticker,

        "action":
            action,

        "factor_before":
            factor_before,

        "factor_after":
            factor_after,

        "factor_diff":
            factor_diff,

        "adjustment_factor":
            factor,

        "open_20250801":
            repaired[
                "Open"
            ],

        "close_20250801":
            repaired[
                "Close"
            ],
    })


    print(
        f"   {action.upper()} "
        f"factor={factor:.8f}"
    )


manifest = pd.DataFrame(
    manifest
)


manifest.to_csv(
    RESULTS
    / "data_repair_20250801.csv",
    index=False,
)


# --------------------------------------------------
# Final verification.
# --------------------------------------------------

valid = 0


for ticker in TICKERS:

    x = pd.read_parquet(
        RAW
        / f"{ticker}.parquet"
    )

    x.index = pd.to_datetime(
        x.index
    )


    if TARGET not in x.index:

        raise RuntimeError(
            f"{ticker}: target "
            f"still missing"
        )


    values = (
        x.loc[
            TARGET,
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ],
        ]
        .astype(float)
    )


    if not np.isfinite(
        values
    ).all():

        raise RuntimeError(
            f"{ticker}: invalid "
            f"target row"
        )


    valid += 1


print()

print(
    "=" * 80
)

print(
    f"VALID 2025-08-01: "
    f"{valid} / 50"
)

print(
    "MARKET-WIDE TWSE REPAIR: PASS"
)

print(
    "=" * 80
)
