from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    compression_bars: int = 6
    atr_bars: int = 20
    compression_atr: float = 2.2
    gap_atr: float = 0.35
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = "gen_local_1790923867"

    @classmethod
    def params_type(cls) -> type[GapCompressionParams]:
        return GapCompressionParams

    def warmup_bars(self, params: GapCompressionParams) -> int:
        return int(max(200, params.atr_bars + 1, params.compression_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapCompressionParams) -> pd.DataFrame:
        if min(params.compression_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError("Lookbacks and holding limit must be positive")
        if not all(np.isfinite(x) and x > 0 for x in (params.compression_atr, params.gap_atr, params.trail_atr)):
            raise ValueError("ATR multipliers must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["prior_atr"] = out["atr"].shift(1).where(lambda s: s > 0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["box_high"] = data["high"].rolling(params.compression_bars, min_periods=params.compression_bars).max().shift(1)
        out["box_low"] = data["low"].rolling(params.compression_bars, min_periods=params.compression_bars).min().shift(1)
        out["compression"] = (out["box_high"] - out["box_low"]) / out["prior_atr"]
        out["gap"] = (data["open"] - out["prior_close"]) / out["prior_atr"]
        out["entry"] = (
            (out["compression"] <= params.compression_atr)
            & (out["gap"] >= params.gap_atr)
            & (data["open"] > out["box_high"])
            & (data["close"] >= data["open"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapCompressionParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        regimes = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            close, atr, regime = closes[i], atrs[i], regimes[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0 and np.isfinite(regime)
            if active:
                age += 1
                if valid:
                    # The first held close is on the bar after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close < stop or close <= regime or age >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif valid and entries[i] and close > regime:
                active = True
                high_water = -np.inf
                stop = -np.inf
                age = 0
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100.0, 101.0, 105.0, 104.0, 102.0, 100.0]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 1.0, 10.0, 1.0],
        "ma200": 90.0,
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, GapCompressionParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
