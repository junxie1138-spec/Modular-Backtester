# Modular Stock Backtester

Python research framework for daily stock strategies. Strategies are
self-contained modules conforming to one ABC contract; the same engine
runs standard backtests, grid optimization, and walk-forward optimization.

## Quick start

```
pip install -e .[dev]
python scripts/generate_sample_data.py
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```

## Layout

```
backtester/       # framework (contracts, engine, analytics, optimize, wfo, runners, io)
strategies/       # user/AI-authored strategies
configs/          # YAML configs for runs
data/raw/         # OHLCV inputs
output/runs/      # deterministic per-run artifact bundles
docs/             # contracts and runbook
tests/            # unit + integration tests
scripts/          # one-off helpers (sample data generator)
```

## Documentation

- `docs/strategy_contract.md` — how to write a strategy.
- `docs/data_contract.md` — OHLCV schema and invariants.
- `docs/runbook.md` — commands, output structure, reproducibility notes.
- `docs/examples.md` — example configs.

## Testing

```
pytest -q
```
