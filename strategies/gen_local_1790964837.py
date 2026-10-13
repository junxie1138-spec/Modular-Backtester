from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_length: int = 3
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790964837"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(15, int(params.streak_length) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["down_close"] = data["close"].lt(out["previous_close"])
        groups = (~out["down_close"]).cumsum()
        out["streak"] = out["down_close"].astype(int).groupby(groups).cumsum()
        out["range"] = data["high"] - data["low"]
        out["close_location"] = ((data["close"] - data["low"]) / out["range"].where(out["range"] > 0)).fillna(0.0)
        out["inside"] = data["low"].ge(data["low"].shift(1)) & data["high"].le(data["high"].shift(1))
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.streak_length < 2 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("streak_length must be at least 2 and trail_atr must be positive")

        candidates = (
            indicators["streak"].ge(params.streak_length)
            & indicators["inside"]
            & indicators["close_location"].ge(0.6)
            & indicators["atr"].gt(0)
            & np.isfinite(indicators["atr"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = 0.0
        peak = 0.0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]):
                    active = True
                    armed = False
                    entry = opens[i]
                    peak = entry
                    stop = entry - 2.0 * atrs[i - 1]
                    held = 0

            if active:
                held += 1
                if np.isfinite(closes[i]) and closes[i] > 0:
                    peak = max(peak, closes[i])
                    if peak >= entry * 1.01:
                        armed = True
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atrs[i])
                    if closes[i] <= stop or held >= 4:
                        active = False
                elif held >= 4:
                    active = False
                signals[i] = int(active)
            elif candidates[i]:
                pending = True
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    close = np.array([100.0] * 15 + [99.0, 98.0, 97.0, 96.8, 98.0, 99.0, 99.0, 99.0, 99.0])
    sample = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000000.0}, index=pd.date_range("2020-01-01", periods=len(close)))
    sample.iloc[17, sample.columns.get_loc("low")] = 95.8
    sample.iloc[17, sample.columns.get_loc("high")] = 98.0
    sample.iloc[18, sample.columns.get_loc("low")] = 96.0
    sample.iloc[18, sample.columns.get_loc("high")] = 97.0
    strategy = GeneratedStrategy()
    params = StrategyParams()
    result = strategy.generate_signals(sample, strategy.indicators(sample, params), None, params).data
    assert result["signal"].iloc[:19].eq(0).all()
    assert result["signal"].iloc[19:23].eq(1).all()
    assert result["signal"].iloc[23] == 0
    assert result["size"].gt(0).all()
