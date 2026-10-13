from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidenceParams:
    minimum_depth_atr: float = 2.0
    trailing_atr: float = 2.8


class GeneratedStrategy(BaseStrategy[ResidenceParams]):
    strategy_id = "gen_local_1791337611"

    @classmethod
    def params_type(cls) -> type[ResidenceParams]:
        return ResidenceParams

    @staticmethod
    def warmup_bars(params: ResidenceParams) -> int:
        return 84

    @staticmethod
    def indicators(data: pd.DataFrame, params: ResidenceParams) -> pd.DataFrame:
        if not np.isfinite(params.minimum_depth_atr) or params.minimum_depth_atr <= 0:
            raise ValueError("minimum_depth_atr must be finite and positive")
        if not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0:
            raise ValueError("trailing_atr must be finite and positive")
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean().replace(0.0, np.nan)
        peak = close.rolling(63, min_periods=63).max()
        depth = ((peak - close) / peak.replace(0.0, np.nan)).clip(lower=0.0)
        depth_sum = depth.rolling(21, min_periods=21).sum()
        centroid = depth.pow(2).rolling(21, min_periods=21).sum() / depth_sum.replace(0.0, np.nan)
        depth_atr = (peak - close) / atr
        price_change = close.diff()
        stable_peak = peak.diff().eq(0.0)
        ready = pd.concat([close, atr, peak, depth, centroid, depth.shift(1), centroid.shift(1)], axis=1).apply(np.isfinite).all(axis=1)
        eligible = ready & stable_peak & depth_atr.ge(params.minimum_depth_atr)
        long_entry = eligible & depth.lt(centroid) & depth.shift(1).ge(centroid.shift(1)) & price_change.gt(0.0)
        short_entry = eligible & depth.gt(centroid) & depth.shift(1).le(centroid.shift(1)) & price_change.lt(0.0)
        return pd.DataFrame({
            "true_range": true_range,
            "atr": atr,
            "peak": peak,
            "depth": depth,
            "depth_sum": depth_sum,
            "centroid": centroid,
            "depth_atr": depth_atr,
            "price_change": price_change,
            "stable_peak": stable_peak,
            "ready": ready,
            "long_entry": long_entry.fillna(False),
            "short_entry": short_entry.fillna(False),
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResidenceParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        for i in range(len(data)):
            if not ready[i] or not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                continue
            if position:
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                if position == 1:
                    stop = max(stop, high_water - params.trailing_atr * atr[i])
                    stopped = close[i] <= stop
                else:
                    stop = min(stop, low_water + params.trailing_atr * atr[i])
                    stopped = close[i] >= stop
                if stopped or i - entry_bar >= 18:
                    position = 0
                    continue
            else:
                position = 1 if longs[i] else (-1 if shorts[i] else 0)
                if position:
                    entry_bar = i
                    high_water = close[i]
                    low_water = close[i]
                    stop = close[i] - position * params.trailing_atr * atr[i]
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100.0, 105.0, 102.0, 100.0, 100.0, 95.0, 99.0, 100.0]}, index=index)
    derived = pd.DataFrame({
        "atr": 1.0,
        "ready": True,
        "long_entry": [True, False, False, False, False, False, False, False],
        "short_entry": [False, False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, ResidenceParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result.data["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
