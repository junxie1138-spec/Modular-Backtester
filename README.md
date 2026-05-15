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
pip install -e .[dev]                    # core install + test runner
pip install -e .[data]                   # adds yfinance for the v0.4.0 cache-on-miss loader
python scripts/generate_sample_data.py   # writes synthetic SPY/AAPL to data/synth/
```

Core install pulls `pandas`, `numpy`, `pyyaml`, `pyarrow`, and `pytest`. The `data` extras adds `yfinance` (only needed if you run `source: yfinance` configs or the universe-screening CLI).

---

## Running on macOS

The backtester and the Strategy Factory both run on macOS unchanged — all file handling uses `pathlib` with explicit UTF-8, and the factory shells out to `git` rather than to any OS-specific tooling. Install is the same as above, with three macOS notes:

- **Python 3.11+.** macOS does not ship a recent enough Python. Install one with Homebrew (`brew install python@3.11`) or from python.org, then work inside a virtual environment:

  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -e '.[dev]'        # quote the extras
  pip install -e '.[data]'
  ```

  Quote the extras: macOS's default shell is `zsh`, which expands `.[dev]` as a glob and fails an unquoted `pip install -e .[dev]` with `zsh: no matches found`.

- **git 2.28+** is required for the distributed factory — `factory/sync.py` uses `git init -b`. The git bundled with current macOS / Xcode Command Line Tools is new enough; `brew install git` also works. Check with `git --version`.

- **The `claude` CLI** is just `claude` on the `PATH` on macOS, so the factory's default `claude_cmd = "claude"` works as-is — the Windows `claude.CMD` wrinkle does not apply.

Everything else is identical: `python -m pytest -q` runs the full suite, and the factory dashboard still serves on `http://127.0.0.1:8787`.

**Mixed Windows + macOS factory pool.** When the distributed factory is enabled (`[sync] enabled = true`), it syncs generated strategies and per-machine result/dedup shards across machines through git. The repo ships a `.gitattributes` that pins those paths to LF line endings, so a pool that mixes Windows and macOS nodes stays conflict-free and `sync_pull` never sees spurious line-ending churn. No action is needed — just give each machine its own `node_id` in `factory/config/settings.local.toml`.

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
  source: "csv"            # csv | parquet | yfinance (v0.4.0)
  root: "data/raw"

execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
  allow_fractional: false
  allow_short: false         # set true to enable -1 signals (short positions)

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

`data/raw/` ships with **real** 2015-2025 OHLCV for SPY, AAPL, `^VIX`, and the 15-name mean-reversion universe (TSLA, NVDA, AMD, COIN, GOOGL, MSTR, XPEV, NIO, PLTR, SMCI, SHOP, W, META, NFLX). The synthetic generator (`scripts/generate_sample_data.py`) now writes to `data/synth/` — used by the backwards-compat test to preserve v0.3.0 golden numerics. Real-data fetches happen via `source: yfinance` (cache-on-miss; the loader writes the full available history to a CSV on first fetch and reads from cache thereafter).

---

## Execution model

- **Long, flat, and short** positions are all supported. Strategies emit `signal` in `{-1, 0, 1}`. Set `execution.allow_short: true` in your config to enable `-1`; otherwise the simulator raises `ShortNotAllowedError` on the first short signal. Default is `false`, so long-only strategies and configs run unchanged.
- Order types: **MARKET**, **LIMIT**, **STOP**. Strategies emit `signal` and optionally a `limit_price` column to use LIMIT orders. LIMIT is honored only when entering from flat (flat → long or flat → short); flips through zero (long → short and short → long) and exits to flat are always MARKET, emitted as a single combined order that closes the prior leg and opens the new one in one fill.
- Signals are typically **shifted by one bar** — the strategy emits intent at bar N's close; the order fills at bar N+1's open (plus configurable slippage in basis points).
- Commission and slippage are both in bps and applied per fill.
- **Trailing stops** (v0.3.0). Set `execution.trailing_stop_pct: 0.05`
  (or `execution.trailing_stop_atr_mult: 3.0` with
  `trailing_stop_atr_period: 14`) to attach a trailing stop to every
  position. The stop trails the running peak (long) or trough (short)
  since entry and fires as a STOP order on the next bar. Stop-out fills
  are tagged `reason="trailing_stop"` in `trades.csv`; signal-driven
  fills are tagged `reason="signal"`. Trailing-stop hits take priority
  over the strategy signal on the same bar. A 5% trailing stop does NOT
  always improve drawdown — see `docs/runbook.md` for limitations.
- Short-position accounting has known limitations (no borrow cost, no margin call, no per-symbol bans). See [`docs/runbook.md`](docs/runbook.md) for the full list.

---

## v0.4.0 — Multi-symbol, regime gates, tranche stops

v0.3.0 introduced execution-layer trailing stops. v0.4.0 adds:

- **Two-phase tranche stop** (`execution.hard_stop_atr_mult` +
  `execution.runner_atr_mult`) for strategies that scale out — fixed hard
  stop while the full position is held; close-basis runner trail with an
  optional breakeven floor once tranche 1 fills.
- **Three-gate regime policy** — SPY 200-EMA, VIX hysteresis, and a rolling
  20-day strategy-PnL circuit breaker. Any tripping gate flattens the entire
  book to cash; configurable hysteresis on each.
- **Multi-symbol portfolio simulator** with shared cash, cross-symbol risk-
  budget enforcement, sector caps, and volatility-targeted position sizing.
  Strategies opt in via `uses_multi_symbol = True`.
- **yfinance data loader** (`source: yfinance`) with cache-on-miss to local
  CSVs, adjusted OHLC contract, and explicit invalidation on date-range
  mismatch.
- **Universe screening CLI** (`scripts/screen_universe.py`) using the
  range/ATR ratio and a 200-day OLS-slope trend filter.
- **`mean_reversion_atr`** — defense-first swing-trading strategy implementing
  the full PRD spec.

The single-symbol v0.3.0 path remains the default; existing strategies are
unaffected.

---

## Reproducibility

- Configs are YAML and round-trip through `config_resolved.yaml`.
- Synthetic sample data is deterministic given the seed.
- Two runs of the same config produce **byte-identical** `total_return`, `sharpe`, `max_drawdown`, and `n_trades`.

---

## Repository layout

```
backtester/                # framework (do not modify per-strategy)
  config/                  # YAML loader + dataclass models + validation + universe loader
  core/                    # shared types, enums, exceptions, constants
  data/                    # OHLCV loaders (csv, parquet, yfinance) + validators
  strategies/              # BaseStrategy ABC, registry, authoring template
  engine/                  # orders, fills, position, broker, portfolio, BacktestEngine,
                           # trailing_stop, atr, tranche_stop, regime, risk_budget,
                           # sector_cap, multi_portfolio, multi_backtest_engine
  analytics/               # metrics, drawdown, trades, exposure
  optimize/                # parameter space, objectives, grid search, lhs_sampler
  wfo/                     # splitter, runner, stitcher
  runners/                 # run_backtest, run_optimize, run_wfo, run_batch
  io/                      # artifact writer, logging, serialization

strategies/                # user/AI-authored strategies (sma_cross, rsi_mean_reversion,
                           # breakout_20d, rsi_long_short, momentum_streak, mean_reversion_atr,
                           # gen_1715800000)
configs/                   # YAML configs for runs (backtests/, optimize/, wfo/, universe.yaml)
data/raw/                  # real OHLCV fixtures (15-name universe + SPY + AAPL + ^VIX)
data/synth/                # deterministic synthetic OHLCV (backwards-compat test fixture)
data/sector_map.csv        # ticker → sector lookup for the universe loader
output/runs/               # per-run artifact bundles (gitignored)
docs/                      # contracts, runbook, and design specs/plans under superpowers/
tests/                     # unit + integration tests
scripts/                   # sample data generator + screen_universe.py
factory/                   # Strategy Factory v0.2.0 — unattended loop that generates
                           # strategies via `claude -p` and runs them through the pipeline
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

The test suite is **352 passing + 5 xfail-by-default** strategy-performance gates (4 stress windows + held-out 2022-2025). It covers every public surface — types, exceptions, data loaders (CSV + yfinance), validators, the engine (signed-qty Position arithmetic, tri-state simulator transitions, trailing stops, tranche stops, regime gates, risk-budget + sector-cap enforcement, multi-symbol simulator), analytics, all six sample strategies, the optimizer (grid + discrete-LHS), WFO, and the four CLIs as end-to-end integration tests. The xfail markers can be flipped to hard asserts once the strategy is tuned to clear PRD performance thresholds — see [`docs/runbook.md`](docs/runbook.md) for the workflow.

---

## Strategy Factory

A separate, opt-in subsystem that lives at `factory/` and wraps the backtester. It is an unattended loop that mass-produces SPY strategy ideas via `claude -p`, validates each one (static AST checks + functional smoke test against the real backtester package), writes it to disk, and runs the full backtest → optimize → WFO pipeline. Generated strategies are auto-discovered from `strategies/gen_*.py` at registry import time — the factory does not edit any backtester source file. A configurable screening gate (the `[screening]` section in `factory/config/settings.toml`) skips the expensive WFO stage when a strategy's best in-sample optimize score falls below a floor — OOS Sharpe is almost always below the best in-sample score, so a hopeless optimize would not clear the shortlist anyway. Screened cycles are recorded as `complete` with `screened_out=true` and shown as `complete (screened)` on the dashboard. Strategies that clear WFO are re-run through a held-out promotion gate on alternate tickers before being flagged. Hits are surfaced via a local Flask dashboard and Telegram alerts.

**Critical context**: at ~24 strategies/day on one asset over one historical path, some strategies will post OOS Sharpe > 1.0 on luck alone. WFO mitigates but does not eliminate multiple-comparisons risk. The dashboard's "good" flag and the Telegram alert are explicitly labelled **shortlist signals, not verdicts** — a held-out gate (different symbol or fully unseen period) is required before treating any row as a real candidate. The held-out promotion gate now runs automatically on every strategy that clears WFO.

```bash
python -m factory.loop                                    # run the continuous loop
python -m factory.dashboard.server                        # local dashboard at http://127.0.0.1:8787
python -m factory.scripts.endurance_check --cycles 100    # validate 100 unattended cycles
python -m factory.scripts.telegram_smoke                  # verify Telegram credentials
```

**Distributed mode (multi-machine).** The factory can run on several machines at once, all contributing into one shared strategy pool coordinated entirely through this git repository — no extra servers, services, or databases. Each machine has a `node_id`, and every machine-owned file is keyed by it: strategy ids are `gen_<node_id>_<ts>`, and the results and dedup stores are per-machine shard directories. Because no two machines ever write the same file, `git pull --rebase` is always conflict-free. `factory/sync.py` pulls the shared `factory-pool` branch before each cycle and pushes after; generated strategies are picked up everywhere by registry auto-discovery. Distributed mode is off by default (`[sync] enabled = false` in `factory/config/settings.toml`) — with it off, the factory behaves exactly as the single-machine version. See [`docs/superpowers/specs/2026-05-16-distributed-factory-design.md`](docs/superpowers/specs/2026-05-16-distributed-factory-design.md) for the design.

Quickstart and configuration live in [`factory/README.md`](factory/README.md). The factory has its own fast unit suite plus slow integration tests, and edits no backtester source file — generated strategies are registered by auto-discovery (`strategies/gen_*.py` globbed at registry import time).

---

## Status

`v0.4.0` — Mean-reversion ATR + multi-symbol framework. Adds the multi-symbol portfolio simulator (shared cash, cross-symbol risk-budget and sector-cap enforcement, volatility-targeted sizing), two-phase tranche stop (HARD→RUNNER state machine with close-basis trail and breakeven floor), three-gate regime policy (SPY 200-EMA, VIX hysteresis, strategy circuit breaker), yfinance loader with cache-on-miss and adjusted-OHLC contract, universe-screening CLI (`scripts/screen_universe.py`), discrete-LHS optimizer mode, and the `mean_reversion_atr` strategy. The v0.3.0 single-symbol path is byte-identical and existing strategies are unaffected. See [`docs/superpowers/specs/2026-05-14-mean-reversion-atr-design.md`](docs/superpowers/specs/2026-05-14-mean-reversion-atr-design.md) for the design and [`docs/superpowers/plans/2026-05-14-mean-reversion-atr.md`](docs/superpowers/plans/2026-05-14-mean-reversion-atr.md) for the 48-task implementation plan. The v0.4.0 line also advances the Strategy Factory: an `exit_rule` generation slot, an optimize→WFO screening gate, the held-out promotion stage on shortlisted strategies, and a distributed multi-machine mode (a git-coordinated shared strategy pool, off by default).

`v0.3.0` — Trailing stops. Adds an execution-layer trailing stop with
two distance modes (percentage of peak/trough, or multiple of recent
ATR). Stop-out exits are tagged `reason="trailing_stop"` in
`trades.csv`. Long-only and long/short configs are unchanged when both
trailing fields are unset. See
[`docs/superpowers/plans/2026-05-14-trailing-stops.md`](docs/superpowers/plans/2026-05-14-trailing-stops.md)
for the design.

`v0.2.0` — Long/short execution. Adds short-position support end-to-end: signed-quantity `Position` arithmetic, tri-state portfolio simulator with combined-order flips through zero, a symmetric `rsi_long_short` sample strategy, and matching WFO + CLI integration coverage. Long-only configs are unchanged; `execution.allow_short` defaults to `false`. See [`docs/superpowers/plans/2026-05-14-short-positions.md`](docs/superpowers/plans/2026-05-14-short-positions.md) for the design.

`v0.1.0` — MVP: backtest, optimize, and WFO workflows with three long-only sample strategies (`sma_cross`, `rsi_mean_reversion`, `breakout_20d`) over MARKET/LIMIT/STOP orders.

Deferred to future versions: borrow-cost / hard-to-borrow modeling, margin-call simulation, per-symbol short bans, intraday timeframes, parallel grid search, multi-symbol WFO, per-symbol parameter overrides applied through the engine, LHS sampler wired into `GridSearchOptimizer.optimize()`, phased circuit-breaker re-entry, HTML report generation, dynamic plugin discovery (currently strategies are explicit-registry only — by design, for predictability).
