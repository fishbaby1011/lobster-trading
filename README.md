# Lobster Trading Research

> A reproducible quantitative research project for cross-sectional stock ranking, walk-forward validation, portfolio simulation, and robustness testing on Taiwan equities.

**Repository:** https://github.com/fishbaby1011/lobster-trading

> [!WARNING]
> This project is research software, not investment advice and not a production trading system.  
> Current backtests still contain important limitations such as survivorship/selection bias, reused development data, simplified transaction-cost assumptions, and incomplete real-market execution constraints.

---

## 1. What is this project?

Lobster Trading Research is an experimental quantitative research pipeline designed to answer a narrower and more useful question than:

> "Will this stock go up tomorrow?"

Instead, the current model asks:

> **Among a universe of stocks, which ones are more likely to outperform the market over the next 10–20 trading days?**

The project currently focuses on **cross-sectional ranking** rather than exact price forecasting.

For a stock `i`:

```text
future_alpha
=
stock_future_return
-
0050_future_return
```

The model learns to rank stocks by expected future alpha.

This means the prediction score itself is not interpreted as a future percentage return. What matters is whether the relative ordering of stocks contains persistent predictive information.

---

## 2. Research philosophy

The project is intentionally built around a few rules:

1. **The model does not directly control trading decisions.** Predictions are passed into deterministic portfolio and risk logic.
2. **Time-series leakage must be avoided.** Training, validation, refit, and test periods follow chronological walk-forward splits.
3. **Labels must be purged at fold boundaries.** A training sample is only allowed if its entire future-return label is known before the split date.
4. **Prediction quality and trading profitability are different problems.** A positive IC does not automatically imply a profitable strategy after turnover, fees, tax, and slippage.
5. **Single-seed results are not trusted.** Neural-network results are tested across many deterministic seeds and combined using rank ensembles.
6. **Backtests should become progressively more realistic.** Each project version removes another source of optimism or methodological weakness.

---

## 3. Current universe and data

The current research universe contains **50 Taiwan-listed equities**, with **0050.TW** used as the market benchmark.

Current dataset metadata:

| Item | Value |
|---|---:|
| Research universe | 50 stocks |
| Benchmark | `0050.TW` |
| Raw data source | Yahoo Finance via `yfinance` |
| Approx. history | 2012–2026 |
| Multi-horizon rows | ~173k |
| Primary horizons | 10 and 20 trading days |
| Features | 18 |
| OOS development period | 2018–2026 |

### Important limitation

The 50-stock universe is a **fixed present-day research universe**, not a historical point-in-time constituent universe.

Therefore, current results are affected by:

- survivorship bias,
- selection bias,
- historical constituent bias.

This is one of the largest remaining research limitations and must be addressed before treating the strategy as production-ready.

---

## 4. Features

The current price-only model uses 18 features.

### Returns

```text
ret_1
ret_5
ret_20
```

### Volatility

```text
vol_5
vol_20
```

### Trend / moving-average structure

```text
ma5_ratio
ma20_ratio
ma60_ratio
rsi_14
```

### Volume / intraday range

```text
volume_ratio
range_ratio
```

### Market context

```text
market_ret_1
market_ret_5
market_ret_20
market_ma20_ratio
```

### Relative strength

```text
excess_ret_1
excess_ret_5
excess_ret_20
```

At this stage, the core model is deliberately **price/volume only**.

News, financial statements, earnings-call text, LLM signals, and event extraction are planned for later phases.

---

## 5. Label and execution timing

The model observes information available at the **close of day T**.

For an `H`-day target:

```text
T close
  ↓
model observes features
  ↓
T+1 open
  ↓
theoretical entry
  ↓
T+H+1 open
  ↓
target exit
```

The label is:

```text
stock_return(T+1 open → T+H+1 open)
-
0050_return(T+1 open → T+H+1 open)
```

This timing prevents the model from using the same closing price both as an input and as an impossible same-instant execution price.

---

## 6. Model

The current neural model is a multilayer perceptron used for **cross-sectional ranking**.

Default hidden layers:

```text
18
↓
1024
↓
512
↓
256
↓
1 score
```

The training objective is a pairwise ranking loss.

Conceptually, the network learns relationships such as:

```text
Stock A should rank above Stock B
Stock B should rank above Stock C
...
```

rather than trying to predict an exact return such as:

```text
2330.TW will return exactly +4.73%
```

---

## 7. Evaluation metric: Information Coefficient

The primary ranking diagnostic is daily cross-sectional **Spearman Information Coefficient (IC)**.

Conceptually:

```text
IC > 0
model ranking agrees with future realized ranking

IC ≈ 0
little ranking information

IC < 0
model ranking is directionally wrong
```

Small IC values can still matter in quantitative research if they are persistent, statistically robust, present across many dates and securities, and monetizable after costs.

---

## 8. Walk-forward methodology

For each test year `Y`, the v5 methodology works like this:

```text
Older history
    │
    ├── Stage training
    │
Y-1 ├── Validation
    │       ↓
    │   choose best epoch count
    │
    ├── discard stage model
    │
All purged pre-Y data
    │
    ├── fresh refit for selected epoch count
    │
Y   └── out-of-sample prediction
```

Example for test year 2026:

```text
Pre-2025 data  → stage training
2025           → validation / epoch selection

Pre-2026 data  → fresh refit
2026           → OOS prediction
```

### Purging

A row near a split boundary is excluded from training if its future label extends beyond the split date.

This is enforced using:

```text
label_end_date < split_date
```

and is intended to prevent look-ahead leakage caused by overlapping future-return labels.

---

## 9. Seed ensemble

Neural-network initialization can materially change predictions.

Instead of selecting whichever seed produced the best backtest, v5 uses multiple predefined seeds.

Each seed produces a daily cross-sectional percentile rank:

```text
seed 1
seed 7
seed 42
seed 123
...
```

The final signal is based on the average percentile rank across the ensemble.

This reduces dependence on one lucky neural-network initialization.

---

## 10. Hysteresis / turnover control

A ranking model can generate excessive turnover when adjacent stocks frequently swap positions.

The portfolio simulator therefore supports different entry and exit thresholds.

Example:

```text
Entry: Top 10
Exit : worse than Top 15
```

A stock already held at rank 12 is not immediately sold simply because it fell outside the Top 10.

This creates a no-trade band and reduces unnecessary turnover.

---

# Research history

## v1 — Single-stock directional baseline

Initial experiments focused on predicting the direction of `2330.TW`.

This was useful as a pipeline sanity check, but it was not considered a sufficiently strong research formulation.

## v2 — Future alpha

The target changed from simple direction prediction to:

```text
stock future return
-
0050 future return
```

The research question became relative performance rather than absolute market direction.

## v3 — Cross-sectional ranking and seed robustness

The project moved from one-stock prediction toward multi-stock cross-sectional ranking.

Major additions included MLP ranking models, multiple random seeds, walk-forward testing, ranking IC analysis, and Top-K portfolio experiments.

A key result was that apparently strong single-seed MLP performance was not sufficiently stable across seeds.

## v4 — Multi-horizon research

Horizons tested:

```text
H1
H3
H5
H10
H20
```

A useful pattern emerged:

- short-horizon models contained some ranking information,
- frequent trading caused poor portfolio economics,
- H10 and H20 were materially more interesting after transaction costs.

This motivated focusing v5 research on H10 and H20.

---

# v5 — Refit + ensemble + continuous reality check

v5 introduced several major methodological improvements:

- 8-seed rank ensemble,
- validation-based epoch selection,
- fresh pre-test refit,
- continuous multi-year portfolio simulation,
- holdings carried across year boundaries,
- drift-aware turnover,
- hysteresis,
- slippage stress tests,
- equal-weight net benchmark,
- random-portfolio baselines.

## Selected v5 diagnostic result

One of the strongest **10 bps slippage** configurations was:

```text
Signal : H20 ensemble
Entry  : Top 10
Exit   : Top 15
Period : 2018-01-02 → 2026-07-02
```

| Metric | Result |
|---|---:|
| Strategy total return | +1998.4% |
| Strategy CAGR | 43.09% |
| 0050 CAGR | 24.39% |
| Net equal-weight CAGR | 24.88% |
| Active Sharpe | 0.799 |
| Period-sampled max drawdown | -24.90% |
| Positive-alpha years | 8 / 9 |
| Worst year | 2024 |
| Worst-year alpha | -47.05% |

### 1,000 random-portfolio diagnostic

For the same H20 / Top10→15 / 10 bps configuration:

| Metric | Result |
|---|---:|
| Model total return | +1998.4% |
| Random median | +271.0% |
| Random 95th percentile | +461.0% |
| Model return percentile | 100% |
| Model active Sharpe | 0.799 |
| Random active-Sharpe 95th percentile | -0.129 |
| Model Sharpe percentile | 100% |

These results are encouraging, but **must not be interpreted as proof of a production alpha strategy**.

### Why these numbers may still be optimistic

The v5 results are affected by several research issues:

- the same 2018–2026 period has been inspected repeatedly across project versions,
- the universe is not point-in-time,
- portfolio configurations were compared on development data,
- the 1,000-random baseline is an exploratory diagnostic rather than a formal multiple-testing-adjusted p-value,
- v5 transaction costs are research assumptions rather than broker-specific live costs,
- v5 drawdown is sampled at rebalance periods rather than reconstructed from a full daily NAV path.

---

# v5.1 — Overnight Robustness Gauntlet

v5.1 is designed to attack the strongest v5 result rather than simply optimize it further.

The experiment matrix contains:

```text
2 horizons
×
6 feature configurations
×
4 training windows
×
16 seeds
=
768 full walk-forward/refit experiments
```

## Horizons

```text
H10
H20
```

## Feature ablations

```text
all
no_absolute_returns
no_relative
no_market_context
no_volatility
no_trend
```

The purpose is to identify which feature families actually contribute to the ranking signal.

## Training windows

```text
expanding
3 years
5 years
8 years
```

This tests whether the model depends on very old market history or adapts better to recent regimes.

## Seeds

16 predefined seeds are used.

Large fold seeds are normalized deterministically for compatibility with NumPy's legacy 32-bit RNG while preserving reproducibility.

## Real-execution simulation

v5.1 adds a more explicit portfolio execution model.

Instead of directly treating target-horizon labels as portfolio PnL, the simulator uses raw adjusted Open prices:

```text
signal on T close
↓
execution at T+1 Open
or T+2 Open
↓
hold position
↓
next rebalance execution Open
```

This makes target construction and realized portfolio PnL separate systems.

## Execution-delay stress test

v5.1 compares:

```text
T+1 execution
T+2 execution
```

If a signal disappears after only one additional trading-day delay, it is likely too fragile for practical use.

## Slippage stress

Planned stress levels:

```text
0 bps
5 bps
10 bps
20 bps
30 bps
50 bps
```

## Statistical diagnostics

The v5.1 analysis also includes:

- block bootstrap for mean daily IC,
- predefined primary configurations,
- 10,000 random portfolio trials,
- turnover and cost analysis,
- explicit Open-to-Open portfolio simulation.

The goal is not to find the most profitable row in a leaderboard.

The goal is to determine whether the signal **survives hostile testing**.

---

# Repository structure

```text
lobster-trading/
├── data/
│   ├── raw/
│   ├── universe_dataset.parquet
│   ├── universe_metadata.json
│   ├── universe_multihorizon.parquet
│   └── universe_multihorizon_metadata.json
│
├── models/
├── results/
├── runs/
│
├── scripts/
│   ├── rerun_baselines.sh
│   ├── run_v2_parallel.sh
│   ├── run_v3_robustness.sh
│   ├── run_v4_multihorizon.sh
│   ├── run_v5_parallel.sh
│   ├── run_v5_1_overnight.sh
│   └── watch_v5_existing.sh
│
└── src/
    ├── baseline.py
    ├── baseline_v2.py
    ├── build_universe.py
    ├── build_multihorizon.py
    ├── train_cross_section.py
    ├── train_cross_section_v2.py
    ├── train_cross_section_v3.py
    ├── train_cross_section_v4.py
    ├── train_cross_section_v5.py
    ├── train_cross_section_v5_1.py
    ├── aggregate_v3.py
    ├── aggregate_v4.py
    ├── aggregate_v5_1.py
    ├── analyze_v5.py
    └── analyze_v5_1_portfolio.py
```

---

# Quick start

## 1. Clone

```bash
git clone https://github.com/fishbaby1011/lobster-trading.git
cd lobster-trading
```

## 2. Create an environment

```bash
python3 -m venv ~/venvs/lobster
source ~/venvs/lobster/bin/activate
python -m pip install --upgrade pip
```

Install the main research dependencies:

```bash
pip install \
  numpy \
  pandas \
  pyarrow \
  scipy \
  scikit-learn \
  xgboost \
  yfinance \
  matplotlib \
  joblib \
  tqdm
```

Install a PyTorch build appropriate for your CPU/CUDA environment separately.

Verify CUDA:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

---

# Build the dataset

## Base universe

```bash
python src/build_universe.py
```

Primary output:

```text
data/universe_dataset.parquet
```

## Multi-horizon targets

```bash
python src/build_multihorizon.py
```

Primary output:

```text
data/universe_multihorizon.parquet
```

---

# Run v5

Example two-GPU launcher:

```bash
GPU_LIST="0 1" \
JOBS_PER_GPU=2 \
./scripts/run_v5_parallel.sh
```

Then:

```bash
python src/analyze_v5.py \
  --random-trials 1000
```

Main outputs:

```text
results/v5_fold_metrics.csv
results/v5_portfolio_leaderboard.csv
results/v5_robust_10bps.csv
results/v5_random_baseline.csv
```

---

# Run the v5.1 overnight gauntlet

The launcher supports multiple GPUs and multiple workers per GPU.

```bash
GPU_LIST="0 1" \
JOBS_PER_GPU=2 \
./scripts/run_v5_1_overnight.sh
```

For a detached long-running session:

```bash
screen -dmS lobster-v5-1 \
bash -lc '
set -o pipefail

cd "$HOME/lobster-trading"

GPU_LIST="0 1" \
JOBS_PER_GPU=2 \
./scripts/run_v5_1_overnight.sh \
2>&1 | tee runs/v5_1_overnight_master.log
'
```

Monitor progress:

```bash
watch -n 30 '
DONE=$(awk "NR>1 {n++} END {print n+0}" \
  runs/v5_1_launcher_logs/status.csv 2>/dev/null)

BAD=$(awk -F, "NR>1 && \$3 != 0 {n++} END {print n+0}" \
  runs/v5_1_launcher_logs/status.csv 2>/dev/null)

ACTIVE=$(pgrep -fc "[t]rain_cross_section_v5_1.py" || true)

echo "Active    : $ACTIVE"
echo "Processed : $DONE / 768"
echo "Failed    : $BAD"

echo

nvidia-smi \
  --query-gpu=index,utilization.gpu,memory.used,power.draw \
  --format=csv
'
```

Expected final outputs include:

```text
results/v5_1_all_runs.csv
results/v5_1_all_folds.csv
results/v5_1_robustness.csv
results/v5_1_yearly_ic.csv
results/v5_1_ablation.csv
results/v5_1_overnight_summary.txt
results/v5_1_portfolio.csv
results/v5_1_portfolio_summary.txt
results/v5_1_primary_random.csv
results/v5_1_primary_ic_bootstrap.csv
```

---

# Hardware

The project can perform data preparation and some baseline work on CPU. GPU acceleration is used for neural-network experiments.

A current development configuration has used:

```text
2 × NVIDIA Tesla V100-SXM2-32GB
PyTorch + CUDA
4 concurrent workers
```

The experiment launcher accepts:

```bash
GPU_LIST="0 1"
JOBS_PER_GPU=2
```

so the same research framework can be scaled to larger multi-GPU systems.

---

# What the project is NOT

This repository is **not** currently:

- an automatic broker,
- a live trading bot,
- a proven alpha source,
- a production portfolio-management system,
- a guarantee of future returns,
- a replacement for financial risk management.

The project is currently a **quantitative research laboratory**.

---

# Major remaining limitations

Before paper trading or real-money use, the following are high priority.

### 1. Point-in-time universe

Replace the fixed present-day 50-stock list with historical constituent and investability data.

### 2. Untouched future holdout

2018–2026 has already been repeatedly examined and should be treated as development data.

The strongest validation will come from genuinely unseen future observations.

### 3. Real Taiwan transaction costs

Research cost assumptions should be replaced with verified broker-specific fees, taxes, discounts, spread assumptions, and realistic slippage.

### 4. Liquidity and execution constraints

The simulator should eventually model liquidity, position capacity, limit-up / limit-down behavior, suspended securities, tradability, round-lot / odd-lot rules, and unavailable Open executions.

### 5. Full daily NAV

Portfolio risk should ultimately be evaluated on a daily mark-to-market NAV path rather than only at rebalance points.

### 6. Multiple-testing control

As more model configurations are explored, apparent winners become easier to discover by chance.

Future evaluation should explicitly account for the entire model-selection process.

### 7. Paper trading

A long-running paper-trading system using only information available at each decision time is required before any live deployment.

---

# Long-term architecture

The current price-ranking model is intended to become only one component of a larger research system.

```text
Price / Volume Signal
        │
        ├── momentum
        ├── volatility
        ├── relative strength
        └── market regime

News / NLP Signal
        │
        ├── company announcements
        ├── earnings calls
        ├── financial news
        ├── event extraction
        └── sentiment / impact

Fundamental Signal
        │
        ├── revenue
        ├── margins
        ├── earnings
        └── valuation

        ↓

Cross-signal ensemble

        ↓

Portfolio construction

        ↓

Hard-coded risk engine

        ↓

Paper trading

        ↓

Potential broker integration
```

The long-term design intentionally keeps LLM reasoning separate from deterministic portfolio and risk controls.

---

# Roadmap

- [x] single-stock baseline
- [x] future-alpha target
- [x] cross-sectional universe
- [x] Ridge / XGBoost / MLP comparison
- [x] ranking loss
- [x] multi-seed robustness
- [x] multi-horizon experiments
- [x] walk-forward validation
- [x] purged labels
- [x] validation-based epoch selection
- [x] pre-test refit
- [x] rank ensemble
- [x] continuous portfolio simulation
- [x] hysteresis
- [x] slippage stress testing
- [x] random-portfolio diagnostics
- [x] v5.1 feature ablation framework
- [x] v5.1 rolling training windows
- [x] explicit Open-to-Open execution simulator
- [ ] complete v5.1 overnight robustness results
- [ ] point-in-time historical universe
- [ ] liquidity / capacity filters
- [ ] full daily NAV simulation
- [ ] stronger multiple-testing controls
- [ ] market-regime model
- [ ] paper trading
- [ ] fundamental-data pipeline
- [ ] financial NLP / event pipeline
- [ ] LLM research-agent integration
- [ ] production risk engine
- [ ] broker integration

---

# Reproducibility

Experiment outputs store configuration and metadata under `runs/`.

Important research artifacts include:

```text
config.json
summary.json
fold_metrics.csv
predictions.parquet
run.log
```

Git commit identifiers are stored with later experiment configurations so a result can be associated with the code that generated it.

For long-running sweeps, launcher logs and status files are stored under:

```text
runs/*_launcher_logs/
```

The v5.1 overnight launcher also supports resuming already completed experiments.

---

# Research status

The strongest current interpretation is:

> The project appears to contain a non-trivial cross-sectional ranking signal worth investigating further, but current backtest evidence is not yet sufficient to justify live trading.

The purpose of the next stages is not to maximize historical return.

It is to determine how much of the observed signal survives:

```text
different seeds
different horizons
different training windows
feature removal
execution delay
higher slippage
realistic costs
point-in-time universes
new unseen market data
```

Only signals that survive those tests should be considered candidates for paper trading.

---

## Disclaimer

This repository is for research and educational purposes only.

Historical backtest performance does not imply future performance. Results may be materially affected by data quality, model-selection bias, transaction costs, market impact, liquidity, execution assumptions, survivorship bias, and regime changes.

Do not use the results in this repository as the sole basis for real-money investment decisions.
