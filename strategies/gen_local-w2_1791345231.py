from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    correlation_window: int = 20
    rank_window: int = 126
    flow_window: int = 10
    atr_window: int = 14
    rank_threshold: float = 0.80
    flow_threshold: float = 0.12
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791345231"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.correlation_window + params.rank_window + 1,
                       params.flow_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.correlation_window, params.rank_window,
               params.flow_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not (0.0 < params.rank_threshold <= 1.0
                and 0.0 <= params.flow_threshold < 1.0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0.0
                and params.max_hold > 0):
            raise ValueError("Invalid entry, stop, or holding parameters.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"])
        out["location"] = ((2.0 * data["close"] - data["high"] - data["low"])
                           / out["range"]).clip(-1.0, 1.0)
        out["lag_location"] = out["location"].shift(1)
        out["correlation"] = out["location"].rolling(
            params.correlation_window).corr(out["lag_location"]).clip(-1.0, 1.0)
        out["correlation_rank"] = out["correlation"].rolling(
            params.rank_window).rank(pct=True)
        out["body"] = ((data["close"] - data["open"])
                       / out["range"]).clip(-1.0, 1.0)
        out["signed_volume"] = out["body"] * data["volume"].where(data["volume"] >= 0)
        out["volume_sum"] = data["volume"].where(data["volume"] >= 0).rolling(
            params.flow_window).sum()
        out["signed_volume_sum"] = out["signed_volume"].rolling(params.flow_window).sum()
        out["flow"] = out["signed_volume_sum"] / out["volume_sum"].where(
            out["volume_sum"] > 0)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        entry = ((indicators["correlation"] > 0.10)
                 & (indicators["correlation_rank"] >= params.rank_threshold)
                 & (indicators["flow"] >= params.flow_threshold)).fillna(False)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = entry.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = stop = -np.inf
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                if not valid:
                    active = False
                    continue
                # First held close is on the bar after the entry decision.
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or i - entry_bar >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
            elif i >= warmup and valid and entries[i]:
                active = True
                entry_bar = i
                high_water = stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    p = GeneratedParams(correlation_window=3, rank_window=3, flow_window=3, atr_window=3)
    index = pd.date_range("2020-01-01", periods=32)
    data = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 95.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"correlation": 0.5, "correlation_rank": 1.0,
                            "flow": 0.5, "atr": 1.0}, index=index)
    derived.loc[index[8:11], "atr"] = [1.0, 10.0, 10.0]
    data.loc[index[8:11], "close"] = [110.0, 109.0, 106.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=7)
    result = strategy.generate_signals(data, derived, ctx, p).data
    assert result["signal"].iloc[:8].eq(0).all()
    assert result["signal"].iloc[8:11].eq(1).all()
    assert result["signal"].iloc[11] == 0  # ATR expansion must not lower the stop.
    derived["flow"] = -0.5
    assert strategy.generate_signals(data, derived, ctx, p).data["signal"].eq(0).all()
    derived["flow"] = 0.5
    data["close"] = 100.0
    derived["atr"] = 1.0
    result = strategy.generate_signals(data, derived, ctx, p).data
    assert result["signal"].iloc[8:26].eq(1).all() and result["signal"].iloc[26] == 0
    flat = data.assign(high=100.0, low=100.0)
    assert strategy.generate_signals(flat, strategy.indicators(flat, p), ctx, p).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
