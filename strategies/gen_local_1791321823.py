from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSeamParams:
    channel_bars: int = 40
    gap_fraction: float = 0.04
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapSeamParams]):
    strategy_id = "gen_local_1791321823"

    @classmethod
    def params_type(cls):
        return GapSeamParams

    def warmup_bars(self, params):
        return max(200, params.channel_bars + 1)

    def indicators(self, data, params):
        if params.channel_bars < 2 or params.hold_bars < 1:
            raise ValueError("Lookback must be >= 2 and holding period >= 1.")
        if not (0 < params.gap_fraction < 1 and 0 < params.profit_target < 1):
            raise ValueError("Gap fraction and profit target must be between 0 and 1.")
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["channel_low"] = data["low"].rolling(params.channel_bars, min_periods=params.channel_bars).min().shift(1)
        out["channel_high"] = data["high"].rolling(params.channel_bars, min_periods=params.channel_bars).max().shift(1)
        out["channel_width"] = (out["channel_high"] - out["channel_low"]).where(lambda s: s > 0)
        out["prior_position"] = (data["close"].shift(1) - out["channel_low"]) / out["channel_width"]
        out["open_position"] = (data["open"] - out["channel_low"]) / out["channel_width"]
        out["close_position"] = (data["close"] - out["channel_low"]) / out["channel_width"]
        out["gap_fraction"] = (data["open"] - data["close"].shift(1)) / out["channel_width"]
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1)
        out["long_setup"] = valid & (data["close"] > out["ma200"]) & (out["prior_position"] < 0.5) & (out["open_position"] >= 0.5) & (out["close_position"] >= 0.5) & (out["gap_fraction"] >= params.gap_fraction)
        out["short_setup"] = valid & (data["close"] < out["ma200"]) & (out["prior_position"] > 0.5) & (out["open_position"] <= 0.5) & (out["close_position"] <= 0.5) & (out["gap_fraction"] <= -params.gap_fraction)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if side:
                # A decision from the previous close fills at this bar's open.
                if entry_bar == -1:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                # Targets are evaluated at the close; exits fill next session.
                gain = side * (closes[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    side = 0
                    entry_bar = -1
            elif np.isfinite(closes[i]) and closes[i] > 0:
                side = 1 if longs[i] else (-1 if shorts[i] else 0)
                entry_bar = -1
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    setups = pd.DataFrame({"long_setup": [True, False, False, False, False, False], "short_setup": False}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    params = GapSeamParams(hold_bars=3)
    result = strategy.generate_signals(data, setups, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    for side, target_close in ((1, 105.0), (-1, 95.0)):
        setups["long_setup"] = [side == 1, False, False, False, False, False]
        setups["short_setup"] = [side == -1, False, False, False, False, False]
        data.loc[index[2], "close"] = target_close
        result = strategy.generate_signals(data, setups, ctx, params).data
        assert result["signal"].tolist() == [0, side, side, 0, 0, 0]
    assert strategy.warmup_bars(GapSeamParams(channel_bars=220)) == 221
    assert not strategy.indicators(data, params)[["long_setup", "short_setup"]].to_numpy().any()


if __name__ == "__main__":
    _self_check()
