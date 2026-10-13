from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBrakeParams:
    peak_window: int = 20
    atr_window: int = 10
    min_depth_atr: float = 2.0
    min_braking: float = 0.5
    trail_atr: float = 0.7
    max_hold: int = 2

    def __post_init__(self):
        if self.peak_window < 2 or self.atr_window < 1:
            raise ValueError("Lookback windows must be positive.")
        if self.min_depth_atr <= 0 or self.trail_atr <= 0:
            raise ValueError("ATR multiples must be positive.")
        if not 0 < self.min_braking < 1 or self.max_hold not in (1, 2):
            raise ValueError("Invalid braking threshold or holding horizon.")


class GeneratedStrategy(BaseStrategy[DrawdownBrakeParams]):
    strategy_id = "gen_local-w2_1791293425"

    @classmethod
    def params_type(cls):
        return DrawdownBrakeParams

    def warmup_bars(self, params):
        return int(max(params.peak_window + 2, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["peak"] = close.rolling(params.peak_window).max()
        out["depth"] = out["peak"] - close
        out["depth_atr"] = out["depth"] / out["atr"].where(out["atr"] > 0)
        out["depth_step"] = out["depth"].diff()
        out["prior_step"] = out["depth_step"].shift(1)
        out["braking"] = 1.0 - out["depth_step"] / out["prior_step"].where(out["prior_step"] > 0)
        width = data["high"] - data["low"]
        out["close_location"] = (close - data["low"]) / width.where(width > 0)
        out["stable_peak"] = out["peak"].eq(out["peak"].shift(1)) & out["peak"].eq(out["peak"].shift(2))
        out["strength"] = (
            (out["depth_atr"] / (2.0 * params.min_depth_atr)).clip(0.0, 1.0)
            * out["braking"].clip(0.0, 1.0)
        ).fillna(0.0)
        out["entry"] = (
            out["stable_peak"]
            & out["depth_atr"].ge(params.min_depth_atr)
            & out["depth_step"].gt(0)
            & out["prior_step"].gt(0)
            & out["braking"].ge(params.min_braking)
            & out["close_location"].ge(0.6)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        active = False
        age = 0
        high_water = 0.0
        stop = 0.0
        entry_size = 0.2
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strengths = indicators["strength"].fillna(0.0).to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                active = False
                continue
            if active:
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                if close <= stop or age >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = entry_size
            elif entries[i]:
                active = True
                age = 0
                high_water = close
                stop = close - params.trail_atr * atr
                entry_size = float(0.2 + 0.75 * np.clip(strengths[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = DrawdownBrakeParams()
    index = pd.date_range("2020-01-01", periods=4)
    ind = pd.DataFrame({
        "atr": [1.0] * 4,
        "entry": [True, False, False, False],
        "strength": [0.8] * 4,
    }, index=index)
    for prices, expected in [
        ([100.0, 98.0, 99.0, 99.0], [0, 1, 0, 0]),
        ([100.0, 101.0, 102.0, 103.0], [0, 1, 1, 0]),
    ]:
        data = pd.DataFrame({"close": prices}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
        assert np.isclose(result["size"].iloc[1], 0.8)


if __name__ == "__main__":
    _self_check()
