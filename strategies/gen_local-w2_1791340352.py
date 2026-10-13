from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    recovery_bars: int = 2
    depth_atr: float = 1.5
    stop_atr: float = 1.5
    hold_days: int = 4
    reference_bars: int = 20
    atr_bars: int = 14


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791340352"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(200, params.reference_bars + params.recovery_bars + 1,
                       params.atr_bars + params.recovery_bars + 1))

    def indicators(self, data, params):
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        delta = close.diff()
        up = delta.gt(0)
        down = delta.lt(0)
        out = pd.DataFrame(index=data.index)
        out["delta"] = delta
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1).where(close.shift(1).notna())
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["peak"] = close.rolling(params.reference_bars, min_periods=params.reference_bars).max()
        out["trough"] = close.rolling(params.reference_bars, min_periods=params.reference_bars).min()
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()
        for side, extremum in (("up", "peak"), ("down", "trough")):
            start = out[side + "_streak"].eq(1)
            out[side + "_anchor"] = close.shift(1).where(start).ffill()
            out[side + "_extremum"] = out[extremum].shift(1).where(start).ffill()
            out[side + "_atr"] = out["atr"].shift(1).where(start).ffill()
        out["drawdown"] = out["up_extremum"] - out["up_anchor"]
        out["rally"] = out["down_anchor"] - out["down_extremum"]
        out["long_recovery"] = (close - out["up_anchor"]) / out["drawdown"].where(out["drawdown"].gt(0))
        out["short_recovery"] = (out["down_anchor"] - close) / out["rally"].where(out["rally"].gt(0))
        out["long_setup"] = (
            close.gt(out["ma200"]) &
            out["up_streak"].eq(params.recovery_bars) &
            out["drawdown"].ge(params.depth_atr * out["up_atr"]) &
            out["long_recovery"].between(0.2, 0.6)
        )
        out["short_setup"] = (
            close.lt(out["ma200"]) &
            out["down_streak"].eq(params.recovery_bars) &
            out["rally"].ge(params.depth_atr * out["down_atr"]) &
            out["short_recovery"].between(0.2, 0.6)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_setup"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_setup"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry, stop, distance = np.nan, np.nan, np.nan
        for i in range(len(data)):
            if position:
                if np.isnan(entry):
                    # This open is observed only on the bar after the entry decision.
                    entry = opens[i]
                    stop = entry - position * distance
                    age = 0
                age += 1
                invalid = not (np.isfinite(entry) and np.isfinite(close[i]))
                stopped = (position == 1 and close[i] < stop) or (position == -1 and close[i] > stop)
                if invalid or stopped or age >= params.hold_days:
                    position = 0
                    entry = np.nan
            elif np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                if longs[i] or shorts[i]:
                    position = 1 if longs[i] else -1
                    distance = params.stop_atr * atr[i]
                    entry = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 96.0,
                         "close": [100., 99., 97., 100., 100., 102., 100., 100.],
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "long_setup": [True] + [False] * 7,
                        "short_setup": [False] * 4 + [True] + [False] * 3}, index=index)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    ctx = StrategyContext("SPY", "1d", 200)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, 0, 0]
    data["close"] = 100.0
    ind["short_setup"] = False
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["long_setup"].eq(False).all()


if __name__ == "__main__":
    _self_check()
