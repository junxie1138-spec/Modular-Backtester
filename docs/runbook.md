# Runbook

## Install

```
pip install -e .[dev]
python scripts/generate_sample_data.py
```

## Commands

```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
python -m backtester.runners.run_optimize --config configs/optimize/sma_cross_grid.yaml
python -m backtester.runners.run_wfo      --config configs/wfo/sma_cross_wfo.yaml
python -m backtester.runners.run_batch    --config <multi-symbol config>
```

## Output bundle

Every run writes a folder under `output/runs/`:

```
output/runs/<timestamp>_<run_name>/
  config_resolved.yaml      # exact config used
  summary.json              # headline metrics
  trades.csv                # fill log
  positions.csv             # per-bar position
  equity_curve.csv          # per-bar cash + position_value + equity
  window_results.json       # WFO only
  oos_equity_curve.csv      # WFO only
  oos_trades.csv            # WFO only
  oos_positions.csv         # WFO only
  grid_results.json         # optimize only
  logs.txt
```

## Reproducibility

- Configs are YAML and round-trip through `config_resolved.yaml`.
- The sample data generator (`scripts/generate_sample_data.py`) is
  deterministic — given the same seed it produces byte-identical CSVs.
- Strategies must not access the environment, network, or local files.

## Testing

```
pytest -q
```

## Limitations (v0.2.0)

Short-position support (`execution.allow_short: true`) intentionally omits
several real-broker features that should be added as follow-up phases:

- **No borrow cost / hard-to-borrow modeling.** Realized PnL on a short
  does not accrue a daily borrow fee. See the `TODO(short-positions)`
  marker in `backtester/engine/position.py`.
- **No margin call simulation.** The simulator assumes unlimited margin
  headroom. A short losing more than the account equity simply produces
  a negative equity series.
- **No leverage cap beyond `portfolio.size <= 1.0`.** When shorts are
  enabled, an instantaneous long → short flip momentarily produces ~2×
  gross exposure (the SELL closes the long and opens the short in one
  fill). If you want a hard gross-exposure cap, reduce `portfolio.size`
  (e.g., `0.5` ensures at most 1× gross around a flip).
- **No short interest / locate / hard-to-borrow availability checks.**
  Every symbol is assumed shortable on every bar.
- **No per-symbol short bans.** There is no mechanism to disable
  shorting on a specific ticker.

If your strategy or backtest depends on any of these effects, treat the
results as an upper bound on real-world performance.

## Trailing-stop limitations (v0.3.0)

The execution-layer trailing stop intentionally omits several features
that should be added as follow-up phases:

- **Only percentage and ATR-multiple distance modes.** Fixed-dollar
  trailing stops are out of scope for v0.3.0 (one-line addition once a
  use case justifies it).
- **Same-bar precedence is hard-coded.** A trailing-stop hit always
  cancels the same-bar signal-driven order. There is no configurable
  ordering between strategy intent and stop trigger.
- **No partial exits.** Stops always close the full position (`qty =
  abs(pos.qty)`). There is no "trail half, hold the rest" mechanism.
- **No grid- or WFO-searchable trailing parameters.** `trailing_stop_*`
  fields are not first-class entries in `OptimizationConfig.param_space`
  yet. To tune them you must hand-run multiple configs.
- **No re-entry cooldown.** After a stop fires, if the strategy signal
  still requests a position on the very next bar, the simulator
  re-enters immediately. Strategies that want a "wait N bars after a
  stop-out" rule must implement it themselves.
- **Drawdown is not guaranteed to improve.** A tight trailing stop can
  *increase* drawdown when paired with a noisy entry signal (whipsaw
  effect: stop exits during normal pullbacks, then re-enters at higher
  prices). E.g., on 2015-2024 SPY data, sma_cross with a 5% trailing
  stop produced a *larger* max drawdown than the no-stop baseline
  (-0.77 vs -0.37). Trailing stops require strategy-specific tuning;
  they are not a free improvement.
- **No interaction with borrow-cost accounting** (which is itself a
  documented v0.2.0 limitation). A short stopped out by a rally still
  pays no borrow during the holding period.
