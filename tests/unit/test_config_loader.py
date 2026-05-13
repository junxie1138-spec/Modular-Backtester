from __future__ import annotations

from pathlib import Path

import pytest

from backtester.config.loader import load_run_config, dump_run_config
from backtester.core.exceptions import ConfigError


YAML_BASIC = """
run_name: sma_cross_spy
strategy: sma_cross
strategy_params:
  fast: 20
  slow: 50
data:
  symbols: ["SPY"]
  timeframe: "1d"
  start: "2020-01-01"
  end: "2024-01-01"
  source: "csv"
execution:
  initial_cash: 100000
  commission_bps: 2
  slippage_bps: 5
portfolio:
  sizing_mode: "percent_equity"
  size: 0.95
"""

YAML_WFO = """
run_name: x
strategy: sma_cross
strategy_params: {fast: 10, slow: 30}
data: {symbols: ["SPY"], timeframe: "1d", start: "2020-01-01", end: "2024-01-01"}
execution: {}
portfolio: {}
optimization:
  objective: sharpe
  param_space:
    fast: [10, 20]
    slow: [50, 100]
wfo:
  enabled: true
  train_bars: 252
  test_bars: 63
  step_bars: 63
"""


def test_load_basic(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML_BASIC)
    rc = load_run_config(p)
    assert rc.run_name == "sma_cross_spy"
    assert rc.data.symbols == ["SPY"]
    assert rc.execution.commission_bps == 2
    assert rc.optimization is None
    assert rc.wfo is None


def test_load_wfo(tmp_path: Path):
    p = tmp_path / "w.yaml"
    p.write_text(YAML_WFO)
    rc = load_run_config(p)
    assert rc.wfo is not None and rc.wfo.enabled is True
    assert rc.wfo.train_bars == 252
    assert rc.optimization is not None
    assert rc.optimization.param_space["fast"] == [10, 20]


def test_load_missing_required_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_name: x\nstrategy: y\n")
    with pytest.raises(ConfigError):
        load_run_config(p)


def test_dump_round_trip(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML_BASIC)
    rc = load_run_config(p)
    out = tmp_path / "out.yaml"
    dump_run_config(rc, out)
    rc2 = load_run_config(out)
    assert rc2.run_name == rc.run_name
    assert rc2.strategy_params == rc.strategy_params
