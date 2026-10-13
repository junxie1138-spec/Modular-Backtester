from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    anchor_bars: int = 60
    flow_bars: int = 18
    recovery_fraction: float = 0.60
    repayment_fraction: float = 0.75
    minimum_depth: float = 0.03

    def __post_init__(self):
        if self.anchor_bars < 2 or self.flow_bars < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0.5 <= self.recovery_fraction < 1.0):
            raise ValueError("Recovery fraction must be in [0.5, 1).")
        if not (self.repayment_fraction > 0 and 0 < self.minimum_depth < 1):
            raise ValueError("Invalid repayment fraction or minimum depth.")


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791292664"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.anchor_bars + 1, 2 * params.flow_bars + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] >= 0)
        out["return"] = close.pct_change(fill_method=None)
        out["impulse"] = (out["return"] * volume).replace([np.inf, -np.inf], np.nan)
        out["flow"] = out["impulse"].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out["previous_flow"] = out["flow"].shift(params.flow_bars)
        out["peak"] = close.shift(1).rolling(params.anchor_bars, min_periods=params.anchor_bars).max()
        out["trough"] = close.shift(1).rolling(params.anchor_bars, min_periods=params.anchor_bars).min()
        out["span"] = (out["peak"] - out["trough"]).where(out["peak"] > out["trough"])
        out["recovery"] = (close - out["trough"]) / out["span"]
        out["drawdown"] = 1.0 - close / out["peak"]
        out["drawup"] = close / out["trough"] - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams,
    ) -> SignalFrame:
        x = indicators
        # Primitive 1: reclaim a depressed range, or unwind an elevated range.
        price_long = (x["drawdown"] >= params.minimum_depth) & (x["recovery"] >= params.recovery_fraction)
        price_short = (x["drawup"] >= params.minimum_depth) & (x["recovery"] <= 1.0 - params.recovery_fraction)
        # Primitive 2: recent signed turnover repays the preceding opposite flow.
        volume_long = (x["previous_flow"] < 0) & (x["flow"] >= -params.repayment_fraction * x["previous_flow"])
        volume_short = (x["previous_flow"] > 0) & (x["flow"] <= -params.repayment_fraction * x["previous_flow"])
        valid = np.isfinite(x[["drawdown", "drawup", "recovery", "flow", "previous_flow"]]).all(axis=1)
        long_entry = (valid & price_long & volume_long).fillna(False).to_numpy(dtype=bool)
        short_entry = (valid & price_short & volume_short).fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            direction = 1 if long_entry[i] else (-1 if short_entry[i] else 0)
            if position == 0:
                position = direction
            elif direction == -position:
                # Opposing AND-confirmation exits to flat; neutral evidence holds.
                position = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    x = pd.DataFrame({
        "drawdown": [0.05] * 8,
        "drawup": [0.05] * 8,
        "recovery": [0.8, 0.8, 0.2, 0.2, 0.2, 0.8, 0.5, np.nan],
        "flow": [1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0, np.nan],
        "previous_flow": [-1.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, np.nan],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, x, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()
    index = pd.date_range("2020-01-01", periods=160)
    close = 100.0 + 10.0 * np.sin(np.arange(160) / 8.0)
    bars = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000.0,
    }, index=index)
    full = strategy.indicators(bars, params)
    assert full.index.equals(bars.index)
    assert full.iloc[strategy.warmup_bars(params) - 1].notna().all()
    prefix = strategy.indicators(bars.iloc[:100], params)
    pd.testing.assert_frame_equal(full.iloc[:100], prefix)
    signals = strategy.generate_signals(bars, full, ctx, params).data
    partial = strategy.generate_signals(bars.iloc[:100], prefix, ctx, params).data
    pd.testing.assert_frame_equal(signals.iloc[:100], partial)


if __name__ == "__main__":
    _self_check()
