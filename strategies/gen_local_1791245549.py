from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    propagation_ratio: float = 1.15
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local_1791245549"

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        return 24

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        if not np.isfinite(params.propagation_ratio) or params.propagation_ratio <= 1:
            raise ValueError("propagation_ratio must exceed one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive")
        close = data["close"]
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        change = close.diff()
        out = pd.DataFrame(index=data.index)
        out["true_range"] = tr
        out["atr_fast"] = tr.rolling(5, min_periods=5).mean()
        out["atr_slow"] = tr.rolling(20, min_periods=20).mean()
        out["atr"] = tr.rolling(10, min_periods=10).mean()
        out["close_change"] = change
        out["close_volatility"] = change.rolling(5, min_periods=5).std(ddof=0)
        out["prior_fast_atr"] = out["atr_fast"].shift(3)
        out["prior_slow_atr"] = out["atr_slow"].shift(3)
        out["prior_close_volatility"] = out["close_volatility"].shift(3)
        out["displacement"] = close.diff(5)
        out["travel"] = change.abs().rolling(5, min_periods=5).sum()
        out["efficiency"] = out["displacement"].abs().div(out["travel"].replace(0, np.nan))
        out["wave"] = (
            (out["prior_fast_atr"] > params.propagation_ratio * out["prior_slow_atr"])
            & (out["atr_fast"] < out["prior_fast_atr"])
            & (out["prior_close_volatility"] > 0)
            & (out["close_volatility"] > params.propagation_ratio * out["prior_close_volatility"])
            & (out["efficiency"] > 0.55)
            & (out["displacement"].abs() > out["atr"])
            & (out["atr"] > 0)
        ).fillna(False)
        direction = np.sign(out["displacement"].fillna(0)).astype(int)
        eligible = direction.where(out["wave"], 0)
        out["entry"] = eligible.where(eligible.ne(eligible.shift(1).fillna(0)), 0).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                position = 0
                age = 0
                extreme = stop = np.nan
                continue
            if position:
                age += 1
                # The first held bar is the bar after the entry decision.
                if age == 1:
                    extreme = close
                    stop = close - position * params.trail_atr * atr
                elif position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                else:
                    # Symmetric rolling-low trailing stop for short positions.
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                breached = close <= stop if position == 1 else close >= stop
                if breached or age >= 8:
                    position = 0
                    age = 0
                    extreme = stop = np.nan
                signals[i] = position
                continue
            if entries[i]:
                position = int(entries[i])
                age = 0
                extreme = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ShockwaveParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    for side, prices in ((1, [200, 100, 110, 105, 110, 110]), (-1, [50, 100, 90, 95, 90, 90])):
        data = pd.DataFrame({"close": prices}, index=index)
        derived = pd.DataFrame({"atr": 1.0, "entry": [side, 0, 0, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(data, derived, None, params)
        assert result.data["signal"].tolist() == [0, side, side, side, 0, 0]
