from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransmissionParams:
    min_correlation: float = 0.25
    trail_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[GapTransmissionParams]):
    strategy_id = "gen_local_1791237545"

    @classmethod
    def params_type(cls) -> type[GapTransmissionParams]:
        return GapTransmissionParams

    @staticmethod
    def warmup_bars(params: GapTransmissionParams) -> int:
        return 22

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTransmissionParams) -> pd.DataFrame:
        previous = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        gap = opening / previous - 1.0
        daytime = data["close"] / opening - 1.0
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=14).mean()
        variance = gap.rolling(20, min_periods=20).var().replace(0.0, np.nan)
        result = pd.DataFrame({
            "gap": gap,
            "daytime": daytime,
            "correlation": gap.rolling(20, min_periods=20).corr(daytime).shift(1),
            "beta": (gap.rolling(20, min_periods=20).cov(daytime) / variance).shift(1),
            "atr": atr,
            "prior_atr": atr.shift(1),
            "gap_atr": (opening - previous) / atr.shift(1).replace(0.0, np.nan),
        }, index=data.index)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapTransmissionParams,
    ) -> SignalFrame:
        if not (0.0 <= params.min_correlation <= 1.0):
            raise ValueError("min_correlation must be between zero and one")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("trail_atr must be finite and positive")
        ready = indicators.notna().all(axis=1)
        entries = (
            ready
            & (indicators["correlation"] >= params.min_correlation)
            & (indicators["beta"] >= 0.5)
            & indicators["gap_atr"].between(0.1, 0.8)
            & (indicators["daytime"] > 0.0)
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        prior_atrs = indicators["prior_atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_close = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                if not (np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0):
                    active = False
                    continue
                if age == 0:
                    if not (np.isfinite(opens[i]) and np.isfinite(prior_atrs[i])):
                        active = False
                        continue
                    # Seed protection at the actual entry open; subsequently only ratchet upward.
                    stop = opens[i] - params.trail_atr * prior_atrs[i]
                high_close = max(high_close, closes[i])
                stop = max(stop, high_close - params.trail_atr * atrs[i])
                age += 1
                if closes[i] <= stop or age >= 2:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i]:
                active = True
                age = 0
                high_close = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 102.0, 103.0],
        "high": [101.0, 103.0, 104.0, 105.0],
        "low": [99.0, 99.0, 101.0, 102.0],
        "close": [100.0, 102.0, 103.0, 104.0],
        "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "gap": 0.005, "daytime": 0.01, "correlation": 0.5,
        "beta": 0.8, "atr": 1.0, "prior_atr": 1.0, "gap_atr": 0.5,
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapTransmissionParams()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    data.loc[index[1], "close"] = 98.0
    ind.loc[index[1]:, "daytime"] = -0.01
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
