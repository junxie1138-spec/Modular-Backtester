from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fast_bars: int = 2
    slow_bars: int = 6
    percentile_bars: int = 63
    entry_quantile: float = 0.80
    profit_target: float = 0.012
    hold_bars: int = 4

    def __post_init__(self):
        if not 1 <= self.fast_bars < self.slow_bars:
            raise ValueError("Require 1 <= fast_bars < slow_bars")
        if self.percentile_bars < 2 or not 0.5 < self.entry_quantile < 1.0:
            raise ValueError("Invalid percentile settings")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1791205031"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(params.slow_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["fast_roc"] = close.pct_change(params.fast_bars, fill_method=None) / params.fast_bars
        out["slow_roc"] = close.pct_change(params.slow_bars, fill_method=None) / params.slow_bars
        out["fast_accel"] = out["fast_roc"].diff()
        out["slow_accel"] = out["slow_roc"].diff()
        out["pulse"] = out["fast_accel"] + out["slow_accel"]
        out["pulse_magnitude"] = out["pulse"].abs()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["pulse_magnitude"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_quantile).shift(1)
        out["previous_opposition"] = (
            out["fast_accel"].shift(1) * out["slow_accel"].shift(1) < 0
        )
        ready = out[["fast_roc", "slow_roc", "fast_accel", "slow_accel", "threshold"]].notna().all(axis=1)
        released = ready & out["previous_opposition"] & (out["pulse_magnitude"] > out["threshold"])
        out["long_entry"] = (
            released & (out["fast_accel"] > 0) & (out["slow_accel"] > 0)
            & (out["fast_roc"] > 0) & (out["slow_roc"] > 0)
        ).fillna(False)
        out["short_entry"] = (
            released & (out["fast_accel"] < 0) & (out["slow_accel"] < 0)
            & (out["fast_roc"] < 0) & (out["slow_roc"] < 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # An entry decided yesterday fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif np.isfinite(closes[i]) and closes[i] > 0:
                position = 1 if longs[i] else (-1 if shorts[i] else 0)
                if position:
                    entry_bar = i + 1
                    entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 97.0,
        "close": [100.0, 102.0, 100.0, 100.0, 100.0, 98.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    entries = pd.DataFrame({
        "long_entry": [True, False, False, False, False, False, False, False],
        "short_entry": [False, False, False, False, True, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    params = WaveParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, -1, 0, 0]
    assert (result["size"] > 0).all()
    data["close"] = 100.0
    result = strategy.generate_signals(data, entries, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert not strategy.indicators(data, params)[["long_entry", "short_entry"]].to_numpy().any()


if __name__ == "__main__":
    _self_check()
