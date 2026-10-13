from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 60
    trough_window: int = 20
    breakout_window: int = 5
    atr_window: int = 14
    min_depth: float = 0.03
    max_depth: float = 0.12
    repair_fraction: float = 0.65
    trail_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791276247"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.peak_window + 2,
                       params.trough_window + 2,
                       params.breakout_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.trough_window,
               params.breakout_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not (0 < params.min_depth < params.max_depth < 1
                and 0 < params.repair_fraction < 1
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Invalid depth, repair fraction, or ATR multiplier")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.peak_window,
                                    min_periods=params.peak_window).max().shift(1)
        out["trough"] = close.rolling(params.trough_window,
                                      min_periods=params.trough_window).min().shift(1)
        out["depth"] = (out["peak"] - out["trough"]) / out["peak"].where(out["peak"] > 0)
        out["recovery_barrier"] = out["trough"] + params.repair_fraction * (out["peak"] - out["trough"])
        out["breakout_level"] = close.rolling(params.breakout_window,
                                              min_periods=params.breakout_window).max().shift(1)
        out["stable_extrema"] = (out["peak"].eq(out["peak"].shift(1))
                                  & out["trough"].eq(out["trough"].shift(1)))
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window,
                                                min_periods=params.atr_window).mean()
        out["entry"] = (
            (close > out["ma200"])
            & out["depth"].between(params.min_depth, params.max_depth)
            & out["stable_extrema"]
            & (out["previous_close"] <= out["recovery_barrier"])
            & (close > out["recovery_barrier"])
            & (close > out["breakout_level"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = stop = -np.inf
        age = 0
        for i in range(len(data)):
            if not (np.isfinite(close[i]) and close[i] > 0
                    and np.isfinite(atr[i]) and atr[i] > 0):
                active = False
                high_water = stop = -np.inf
                age = 0
                continue
            if active:
                age += 1
                # The first tracked close belongs to the next-bar entry session.
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
            elif entries[i]:
                active = True
                high_water = stop = -np.inf
                age = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100., 110., 109., 107., 100.]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1., 1., 10., 10., 1.],
        "entry": [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, RecoveryParams()).data
    # ATR expansion cannot lower the stop; the exit takes effect one bar later.
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    expiry = strategy.generate_signals(
        data, indicators, None, RecoveryParams(hold_bars=2)
    ).data
    assert expiry["signal"].tolist() == [0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
