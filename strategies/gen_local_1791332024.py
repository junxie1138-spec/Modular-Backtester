from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    lookback: int = 8
    impact_percentile: float = 0.30
    volume_percentile: float = 0.80
    profit_target: float = 0.006
    hold_bars: int = 2

    def __post_init__(self):
        if not 3 <= self.lookback <= 10 or not 1 <= self.hold_bars <= 2:
            raise ValueError("lookback must be 3-10; hold_bars must be 1-2")
        if not 0 < self.impact_percentile < self.volume_percentile <= 1:
            raise ValueError("invalid percentile thresholds")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1791332024"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params):
        return params.lookback

    def indicators(self, data, params):
        opening = data["open"].where(data["open"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        body = data["close"].div(opening).sub(1)
        impact = body.abs().div(volume).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["body_return"] = body
        out["impact"] = impact
        out["impact_rank"] = impact.rolling(params.lookback, min_periods=params.lookback).rank(pct=True)
        out["volume_rank"] = volume.rolling(params.lookback, min_periods=params.lookback).rank(pct=True)
        out["mean_close"] = data["close"].rolling(params.lookback, min_periods=params.lookback).mean()
        out["entry"] = (
            out["impact_rank"].le(params.impact_percentile)
            & out["volume_rank"].ge(params.volume_percentile)
            & body.lt(0)
            & data["close"].lt(out["mean_close"])
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Read the fill-day open only after that day has arrived.
                if i == entry_bar:
                    entry_price = opening[i]
                valid_fill = np.isfinite(entry_price) and entry_price > 0
                # Gross open-to-close target, observed at close; exit fills next open.
                target_hit = valid_fill and np.isfinite(closing[i]) and closing[i] >= entry_price * (1 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.hold_bars
                if not valid_fill or target_hit or time_up:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closing[i]) and closing[i] > 0:
                entry_bar = i + 1
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AbsorptionParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    timed = strategy.generate_signals(data, entries, ctx, params).data
    assert timed["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 101.0
    targeted = strategy.generate_signals(data, entries, ctx, params).data
    assert targeted["signal"].tolist() == [0, 1, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
    assert strategy.warmup_bars(AbsorptionParams(lookback=10)) == 10


if __name__ == "__main__":
    _self_check()
