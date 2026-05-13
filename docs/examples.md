# Examples

## 1. Backtest one strategy on one symbol

```yaml
# configs/backtests/sma_cross_spy.yaml
run_name: sma_cross_spy
strategy: sma_cross
strategy_params: {fast: 20, slow: 50}
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2015-01-02"
  end: "2024-12-31"
  source: csv
execution: {initial_cash: 100000, commission_bps: 2, slippage_bps: 5}
portfolio: {size: 0.95}
```

```
python -m backtester.runners.run_backtest --config configs/backtests/sma_cross_spy.yaml
```

## 2. Grid optimize the same strategy

```yaml
# configs/optimize/sma_cross_grid.yaml
optimization:
  objective: sharpe
  param_space:
    fast: [10, 20, 30]
    slow: [50, 100, 200]
```

## 3. Walk-forward optimization

```yaml
# configs/wfo/sma_cross_wfo.yaml
wfo:
  enabled: true
  train_bars: 756   # 3 trading years
  test_bars: 252    # 1 trading year
  step_bars: 252
```
