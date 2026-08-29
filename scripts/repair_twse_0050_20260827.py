from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import requests


RAW = Path("data/raw/0050.TW.parquet")

BACKUP_DIR = Path(
    "data/raw_backup_0050_20260827"
)

RESULTS = Path("results")

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS.mkdir(
    exist_ok=True,
)


BEFORE = pd.Timestamp(
    "2026-08-26"
)

TARGET = pd.Timestamp(
    "2026-08-27"
)

AFTER = pd.Timestamp(
    "2026-08-28"
)


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "Chrome/131 Safari/537.36"
    ),
    "Accept":
        "application/json,text/plain,*/*",
    "Referer":
        "https://www.twse.com.tw/",
})


def num(x):

    s = (
        str(x)
        .replace(",", "")
        .strip()
    )

    if s in {
        "",
        "--",
        "---",
    }:
        return np.nan

    return float(s)


def fetch_0050_august():

    url = (
        "https://www.twse.com.tw/"
        "rwd/zh/afterTrading/STOCK_DAY"
    )

    last_error = None

    for attempt in range(6):

        try:

            print(
                f"TWSE fetch attempt "
                f"{attempt + 1}/6",
                flush=True,
            )

            r = session.get(
                url,
                params={
                    "date":
                        "20260801",
                    "stockNo":
                        "0050",
                    "response":
                        "json",
                },
                timeout=45,
            )

            if r.status_code == 428:

                wait = min(
                    20 * (
                        2 ** attempt
                    ),
                    120,
                )

                print(
                    f"TWSE 428; "
                    f"sleep {wait}s"
                )

                time.sleep(wait)
                continue

            r.raise_for_status()

            if not r.text.strip():

                raise RuntimeError(
                    "empty response"
                )

            j = r.json()

            if j.get("stat") != "OK":

                raise RuntimeError(
                    f"TWSE stat="
                    f"{j.get('stat')}"
                )

            rows = []

            for row in j["data"]:

                roc = row[0]

                y, m, d = [
                    int(x)
                    for x in roc.split("/")
                ]

                date = pd.Timestamp(
                    year=y + 1911,
                    month=m,
                    day=d,
                )

                rows.append({
                    "Date":
                        date,

                    "Volume":
                        num(row[1]),

                    "Open":
                        num(row[3]),

                    "High":
                        num(row[4]),

                    "Low":
                        num(row[5]),

                    "Close":
                        num(row[6]),
                })

            out = (
                pd.DataFrame(rows)
                .set_index("Date")
                .sort_index()
            )

            return out

        except Exception as e:

            last_error = e

            wait = min(
                10 * (
                    2 ** attempt
                ),
                60,
            )

            print(
                f"ERROR: {e}"
            )

            print(
                f"sleep {wait}s"
            )

            time.sleep(wait)

    raise RuntimeError(
        f"TWSE fetch failed: "
        f"{last_error}"
    )


if not RAW.exists():

    raise FileNotFoundError(
        RAW
    )


yahoo = pd.read_parquet(
    RAW
)

yahoo.index = pd.to_datetime(
    yahoo.index
)

yahoo = yahoo.sort_index()


print(
    "Yahoo has target before repair:",
    TARGET in yahoo.index,
)


for d in [
    BEFORE,
    AFTER,
]:

    if d not in yahoo.index:

        raise RuntimeError(
            f"Yahoo reference missing: {d}"
        )


twse = fetch_0050_august()


print()
print(
    "===== TWSE 0050 ====="
)

print(
    twse.loc[
        BEFORE:AFTER
    ].to_string()
)


for d in [
    BEFORE,
    TARGET,
    AFTER,
]:

    if d not in twse.index:

        raise RuntimeError(
            f"TWSE date missing: {d}"
        )


# Yahoo data was downloaded using
# auto_adjust=True.
#
# Infer adjustment factor using both
# neighboring valid trading days.

factor_before = (
    float(
        yahoo.loc[
            BEFORE,
            "Close",
        ]
    )
    /
    float(
        twse.loc[
            BEFORE,
            "Close",
        ]
    )
)


factor_after = (
    float(
        yahoo.loc[
            AFTER,
            "Close",
        ]
    )
    /
    float(
        twse.loc[
            AFTER,
            "Close",
        ]
    )
)


factor_diff = abs(
    factor_before
    -
    factor_after
) / max(
    (
        abs(factor_before)
        +
        abs(factor_after)
    )
    / 2,
    1e-12,
)


print()
print(
    "factor_before:",
    factor_before,
)

print(
    "factor_after :",
    factor_after,
)

print(
    "relative diff:",
    factor_diff,
)


if factor_diff > 0.005:

    raise RuntimeError(
        "Adjustment factor mismatch. "
        "Do not interpolate across "
        "possible corporate action."
    )


factor = (
    factor_before
    +
    factor_after
) / 2


backup = (
    BACKUP_DIR
    / "0050.TW.parquet"
)


if not backup.exists():

    shutil.copy2(
        RAW,
        backup,
    )


raw = twse.loc[
    TARGET
]


repaired = pd.DataFrame(
    {
        "Open": [
            float(
                raw["Open"]
            )
            * factor
        ],

        "High": [
            float(
                raw["High"]
            )
            * factor
        ],

        "Low": [
            float(
                raw["Low"]
            )
            * factor
        ],

        "Close": [
            float(
                raw["Close"]
            )
            * factor
        ],

        "Volume": [
            float(
                raw["Volume"]
            )
        ],
    },
    index=[
        TARGET
    ],
)


repaired = repaired[
    yahoo.columns
]


if TARGET in yahoo.index:

    existing = (
        yahoo.loc[
            TARGET,
            repaired.columns,
        ]
        .astype(float)
    )

    expected = (
        repaired.loc[
            TARGET
        ]
        .astype(float)
    )

    rel = (
        (
            existing
            -
            expected
        )
        .abs()
        /
        expected
        .abs()
        .clip(
            lower=1e-12
        )
    )

    if (
        rel[
            [
                "Open",
                "High",
                "Low",
                "Close",
            ]
        ]
        .max()
        > 0.001
    ):

        raise RuntimeError(
            "Existing repair does not "
            "match TWSE-derived values"
        )

    action = "VERIFY"

else:

    yahoo = pd.concat(
        [
            yahoo,
            repaired,
        ]
    )

    yahoo = (
        yahoo.sort_index()
    )

    yahoo = yahoo[
        ~yahoo.index.duplicated(
            keep="first"
        )
    ]

    yahoo.to_parquet(
        RAW
    )

    action = "INSERT"


check = pd.read_parquet(
    RAW
)

check.index = pd.to_datetime(
    check.index
)


if TARGET not in check.index:

    raise RuntimeError(
        "Repair write verification failed"
    )


manifest = pd.DataFrame([
    {
        "ticker":
            "0050.TW",

        "date":
            TARGET,

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

        "open":
            float(
                repaired.Open.iloc[0]
            ),

        "high":
            float(
                repaired.High.iloc[0]
            ),

        "low":
            float(
                repaired.Low.iloc[0]
            ),

        "close":
            float(
                repaired.Close.iloc[0]
            ),

        "volume":
            float(
                repaired.Volume.iloc[0]
            ),
    }
])


manifest.to_csv(
    RESULTS
    / "data_repair_0050_20260827.csv",
    index=False,
)


print()
print(
    "=" * 80
)

print(
    f"{action} 0050.TW "
    f"{TARGET.date()}"
)

print(
    check.loc[
        TARGET
    ].to_string()
)

print()

print(
    "0050 2026-08-27 REPAIR: PASS"
)

print(
    "=" * 80
)
