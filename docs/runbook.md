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
