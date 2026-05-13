# Modular Stock Backtester

A Python research framework for daily stock strategies. Write a strategy as one self-contained module conforming to a single interface; the same engine runs **standard backtests**, **grid parameter optimization**, and **walk-forward optimization (WFO)** with no engine changes per strategy.

Designed for local quantitative research and reproducible experiments.

---

## What you can do with it

1. **Backtest** one strategy on one symbol over a date range.
2. **Batch-backtest** the same strategy across many symbols.
3. **Grid-optimize** parameters (e.g., find the best fast/slow window pair) against an objective (sharpe / sortino / calmar / total_return).
4. **Walk-forward optimize** — repeatedly re-optimize on a rolling in-sample window and validate on the next out-of-sample window. Only OOS results are stitched into the headline summary, with parameter stability tracked per window.
5. **Drop in a new strategy** from a template — one file, one registry line, and it's runnable.

---

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/junxie1138-spec/Modular-Backtester.git
cd Modular-Backtester
pip install -e .[dev]
python scripts/generate_sample_data.py   # creates bundled SPY/AAPL sample CSVs
```

This installs `pandas`, `numpy`, `pyyaml`, `pyarrow`, and `pytest`.

---

## The three workflows

Every run is driven by a YAML config and writes a deterministic artifact bundle under `output/runs/<timestamp>_<run_name>/`.

### 1. Single backtest

```bash
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```

Output bundle:

```
output/runs/20260514_0319_sma_cross_spy/
├── config_resolved.yaml    # exact config used
├── summary.json            # headline metrics (sharpe, drawdown, win rate, etc.)
├── trades.csv              # every fill
├── positions.csv           # per-bar position state
├── equity_curve.csv        # per-bar cash + position_value + equity
└── logs.txt
```

### 2. Grid parameter optimization

```bash
python -m backtester.runners.run_optimize --config configs/optimize/sma_cross_grid.yaml
```

Runs every parameter combination, picks the best by the configured objective, and additionally writes `grid_results.json` with every combo's score.

### 3. Walk-forward optimization

```bash
python -m backtester.runners.run_wfo --config configs/wfo/sma_cross_wfo.yaml
```

For each (train_bars, test_bars, step_bars) window:
- Grid-optimize on the **train** slice (in-sample).
- Run the chosen parameters on the **test** slice (out-of-sample).
- Stitch the OOS equity curves into one continuous series.

Output additionally includes `window_results.json` (per-window summary + best params), `oos_equity_curve.csv`, `oos_trades.csv`, `oos_positions.csv`, and `parameter_stability` in `summary.json`.

### 4. Multi-symbol batch

```bash
python -m backtester.runners.run_batch --config <your-batch-config>.yaml
```

Iterates `data.symbols`, runs the strategy per symbol, writes per-symbol artifacts plus a top-level `batch_summary.json`.

---

## Writing a strategy

A strategy is one Python file in `strategies/`. It defines:

1. A `@dataclass(slots=True)` for parameters.
2. A class inheriting from `BaseStrategy[ParamsType]` with three methods: `params_type()`, `indicators(data, params)`, `generate_signals(data, indicators, ctx, params)`.

Minimal example (the bundled SMA cross strategy):

```python
from dataclasses import dataclass
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SMACrossParams:
    fast: int = 20
    slow: int = 50
    size: float = 1.0


class SMACrossStrategy(BaseStrategy[SMACrossParams]):
    strategy_id = "sma_cross"

    @classmethod
    def params_type(cls):
        return SMACrossParams

    def warmup_bars(self, params): return max(params.fast, params.slow)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["fast_sma"] = data["close"].rolling(params.fast).mean()
        out["slow_sma"] = data["close"].rolling(params.slow).mean()
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df.loc[indicators["fast_sma"] > indicators["slow_sma"], "signal"] = 1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)  # trade next bar
        df["size"] = params.size
        return SignalFrame(data=df)
```

Then register it in `backtester/strategies/registry.py`:

```python
from strategies.sma_cross import SMACrossStrategy
register_strategy(SMACrossStrategy)
```

That's it — `strategy: sma_cross` in any YAML config will route to your class. Full rules in [`docs/strategy_contract.md`](docs/strategy_contract.md).

There's also a [template file](backtester/strategies/templates/strategy_template.py) you can copy.

---

## Anatomy of a config

```yaml
run_name: sma_cross_spy
strategy: sma_cross
strategy_params:
  fast: 20
  slow: 50
  size: 1.0

data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: "csv"            # csv | parquet
  root: "data/raw"

execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false

portfolio:
  sizing_mode: "percent_equity"
  size: 0.95               # use 95% of equity per entry

# Optional — enables grid optimization:
optimization:
  objective: sharpe        # sharpe | sortino | total_return | annualized_return | calmar
  param_space:
    fast: [10, 20, 30]
    slow: [50, 100, 200]

# Optional — enables walk-forward:
wfo:
  enabled: true
  train_bars: 756          # 3 trading years
  test_bars: 252           # 1 trading year
  step_bars: 252
```

See [`docs/examples.md`](docs/examples.md) for more.

---

## Bringing your own data

The framework reads OHLCV from `data/raw/{SYMBOL}.csv` or `.parquet`. Required columns: `open`, `high`, `low`, `close`, `volume` (case-insensitive), with a parseable date index. Invariants are enforced by `backtester.data.validators.validate_ohlcv` before any run.

Full schema: [`docs/data_contract.md`](docs/data_contract.md).

Bundled sample CSVs (`SPY.csv`, `AAPL.csv`) are deterministic synthetic data produced by `scripts/generate_sample_data.py` — useful for testing the framework but not real market data.

---

## Execution model

- **Long-only** for the MVP.
- Order types: **MARKET**, **LIMIT**, **STOP**. Strategies emit `signal` (0/1) and optionally a `limit_price` column to use LIMIT orders.
- Signals are typically **shifted by one bar** — the strategy emits intent at bar N's close; the order fills at bar N+1's open (plus configurable slippage in basis points).
- Commission and slippage are both in bps and applied per fill.

---

## Reproducibility

- Configs are YAML and round-trip through `config_resolved.yaml`.
- Synthetic sample data is deterministic given the seed.
- Two runs of the same config produce **byte-identical** `total_return`, `sharpe`, `max_drawdown`, and `n_trades`.

---

## Repository layout

```
backtester/                # framework (do not modify per-strategy)
  config/                  # YAML loader + dataclass models + validation
  core/                    # shared types, enums, exceptions, constants
  data/                    # OHLCV loaders + validators
  strategies/              # BaseStrategy ABC, registry, authoring template
  engine/                  # orders, fills, position, broker, portfolio, BacktestEngine
  analytics/               # metrics, drawdown, trades, exposure
  optimize/                # parameter space, objectives, grid search
  wfo/                     # splitter, runner, stitcher
  runners/                 # run_backtest, run_optimize, run_wfo, run_batch
  io/                      # artifact writer, logging, serialization

strategies/                # user/AI-authored strategies (sma_cross, rsi_mean_reversion, breakout_20d)
configs/                   # YAML configs for runs (backtests/, optimize/, wfo/)
data/raw/                  # OHLCV inputs
output/runs/               # per-run artifact bundles (gitignored)
docs/                      # contracts and runbook
tests/                     # unit + integration tests
scripts/                   # sample data generator
```

---

## Documentation

- [`docs/strategy_contract.md`](docs/strategy_contract.md) — strategy interface, rules, signal semantics.
- [`docs/data_contract.md`](docs/data_contract.md) — OHLCV schema and validation invariants.
- [`docs/runbook.md`](docs/runbook.md) — CLI commands and output bundle format.
- [`docs/examples.md`](docs/examples.md) — example configs for each workflow.

---

## Testing

```bash
python -m pytest -q
```

The test suite is **135 tests** covering every public surface — types, exceptions, data loaders, validators, the engine, analytics, all three sample strategies, the optimizer, WFO, and the four CLIs as end-to-end integration tests.

---

## Status

`v0.1.0` — MVP complete: backtest, optimize, and WFO workflows with three sample strategies. Long-only execution with MARKET/LIMIT/STOP orders.

Deferred to future versions: short selling, intraday timeframes, parallel grid search, portfolio-level constraints, HTML report generation, dynamic plugin discovery (currently strategies are explicit-registry only — by design, for predictability).
