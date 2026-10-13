from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    z_threshold: float = 1.2
    atr_window: int = 14
    trail_atr: float = 2.0
    hold_days: int = 4

    def __post_init__(self):
        if self.ma_window < 3 or self.atr_window < 1:
            raise ValueError("Invalid rolling window")
        if not 3 <= self.hold_days <= 5:
            raise ValueError("hold_days must be between 3 and 5")
        if not np.isfinite(self.z_threshold) or self.z_threshold <= 0:
            raise ValueError("z_threshold must be finite and positive")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791262951"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.ma_window + 2, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = close.shift(1)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["regime_ma"] = close.rolling(200).mean()
        out["dispersion"] = close.rolling(params.ma_window).std(ddof=0)
        out["distance"] = close - out["ma"]
        out["z"] = out["distance"] / out["dispersion"].replace(0.0, np.nan)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["contracting"] = (
            (out["dispersion"] < out["dispersion"].shift(1))
            & (out["dispersion"].shift(1) < out["dispersion"].shift(2))
        )
        long_entry = (
            out["contracting"]
            & (out["z"] >= params.z_threshold)
            & (out["z"].shift(1) < params.z_threshold)
            & (out["distance"] > out["distance"].shift(1))
            & (close > out["regime_ma"])
        )
        short_entry = (
            out["contracting"]
            & (out["z"] <= -params.z_threshold)
            & (out["z"].shift(1) > -params.z_threshold)
            & (out["distance"] < out["distance"].shift(1))
            & (close < out["regime_ma"])
        )
        out["entry"] = np.select(
            [long_entry.fillna(False), short_entry.fillna(False)],
            [1, -1], default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = stop = -np.inf
        rows = zip(
            data["close"].to_numpy(),
            indicators["atr"].to_numpy(),
            indicators["regime_ma"].to_numpy(),
            indicators["entry"].to_numpy(),
        )
        for i, (price, atr, regime, entry) in enumerate(rows):
            if not np.isfinite([price, atr, regime]).all() or atr <= 0:
                position = 0
                signals[i] = 0
                continue
            if position:
                age += 1
                # Long: highest close since fill; short: mirrored lowest close.
                # The decision-bar close is excluded because entry fills next bar.
                watermark = max(watermark, position * price)
                stop = max(stop, watermark - params.trail_atr * atr)
                regime_failed = position * (price - regime) <= 0
                if position * price <= stop or age >= params.hold_days or regime_failed:
                    position = 0
                # An exit consumes this decision bar; never immediately re-enter.
            elif entry in (-1, 1) and entry * (price - regime) > 0:
                position = int(entry)
                age = 0
                watermark = stop = -np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    cases = [
        ([110, 111, 115, 112, 111, 110], [1] * 6, [0, 1, 1, 1, 0, 0]),
        ([110, 111, 115, 113, 112, 111], [1, 1, 1, 10, 10, 10], [0, 1, 1, 1, 0, 0]),
        ([110, 111, 112, 113, 114, 115], [1] * 6, [0, 1, 1, 1, 1, 0]),
    ]
    for prices, atr, expected in cases:
        for direction in (1, -1):
            index = pd.date_range("2020-01-01", periods=6)
            data = pd.DataFrame({"close": np.array(prices) * direction + 300}, index=index)
            ind = pd.DataFrame(
                {"atr": atr, "regime_ma": 300, "entry": [direction, 0, 0, 0, 0, 0]},
                index=index,
            )
            result = GeneratedStrategy.generate_signals(data, ind, None, params).data
            assert result["signal"].tolist() == [direction * x for x in expected]
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
