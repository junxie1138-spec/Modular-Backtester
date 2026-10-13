from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    regime_window: int = 40
    atr_window: int = 14
    streak_bars: int = 3
    correlation_floor: float = 0.15
    trailing_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791244283"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.regime_window, params.atr_window) + params.streak_bars + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.regime_window < 2 or params.atr_window < 1
                or params.streak_bars < 1 or params.holding_bars < 1
                or not 0.0 < params.correlation_floor < 1.0
                or not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range_fraction"] = (data["high"] - data["low"]) / data["close"].replace(0, np.nan)
        out["volume_change"] = data["volume"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["range_change"] = out["range_fraction"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["correlation"] = out["volume_change"].rolling(
            params.regime_window, min_periods=params.regime_window
        ).corr(out["range_change"])
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["regime"] = np.select([
            out["correlation"] >= params.correlation_floor,
            out["correlation"] <= -params.correlation_floor,
        ], [1, -1], default=0)
        out["up_close"] = data["close"] > out["previous_close"]
        out["event"] = out["up_close"] & (
            ((out["regime"] == 1) & (out["volume_change"] > 0) & (out["range_change"] > 0))
            | ((out["regime"] == -1) & (out["volume_change"] < 0) & (out["range_change"] > 0))
        )
        out["streak_reset"] = (~out["event"]) | out["regime"].ne(out["regime"].shift(1))
        out["streak"] = out["event"].astype(int).groupby(out["streak_reset"].cumsum()).cumsum()
        out["eligible"] = (
            (out["streak"] >= params.streak_bars)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
            & np.isfinite(data["close"])
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = (
            eligible & eligible.shift(1, fill_value=False)
            & indicators["regime"].eq(indicators["regime"].shift(1))
        ).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                # The first bar after the entry decision is the first invested bar.
                age += 1
                if valid:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trailing_atr * atr)
                if not valid or close <= stop or age >= params.holding_bars:
                    active = False
                else:
                    signal[i] = 1
            elif valid and confirmed[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100, 101, 102, 103, 102, 100, 100, 100]}, index=index)
    indicators = pd.DataFrame({
        "eligible": [True, True, False, False, False, False, False, False],
        "regime": [1] * 8,
        "atr": [1, 1, 1, 1, 10, 10, 1, 1],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    result = strategy.generate_signals(data, indicators, None, Params(holding_bars=2)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    indicators.loc[index[1], "regime"] = -1
    assert not strategy.generate_signals(data, indicators, None, Params()).data["signal"].any()
