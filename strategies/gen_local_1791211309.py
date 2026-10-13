from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    rank_window: int = 60
    compression_pct: float = 0.30
    volume_pct: float = 0.55
    close_location_min: float = 0.65
    atr_window: int = 14
    trail_atr: float = 1.5
    hold_days: int = 4

    def __post_init__(self):
        if self.rank_window < 2 or self.atr_window < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < self.compression_pct < 1 and 0 < self.volume_pct < 1):
            raise ValueError("Percentiles must be between zero and one.")
        if not (0 <= self.close_location_min <= 1 and self.trail_atr > 0):
            raise ValueError("Invalid close location or trailing distance.")
        if not 3 <= self.hold_days <= 5:
            raise ValueError("Holding horizon must be three to five days.")


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1791211309"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return int(max(params.rank_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["relative_range"] = width / previous.where(previous > 0)
        out["range_rank"] = out["relative_range"].rolling(params.rank_window).rank(pct=True)
        out["volume_rank"] = data["volume"].where(data["volume"] > 0).rolling(params.rank_window).rank(pct=True)
        out["close_location"] = (data["close"] - data["low"]) / width
        out["true_range"] = pd.concat(
            [width, (data["high"] - previous).abs(), (data["low"] - previous).abs()],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["rising_close"] = data["close"].gt(previous)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            out["range_rank"].le(params.compression_pct)
            & out["volume_rank"].ge(params.volume_pct)
            & out["close_location"].ge(params.close_location_min)
            & out["rising_close"].fillna(False)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AbsorptionParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                active = False
                continue
            if active:
                age += 1
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if closes[i] <= stop or age >= params.hold_days:
                    active = False
            elif i > 0 and setups[i] and setups[i - 1]:
                active = True
                age = 0
                # The next bar is the entry bar; start tracking its closing high.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AbsorptionParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100, 101, 102, 103, 100.25, 100, 101, 102]}, index=index)
    features = pd.DataFrame(
        {"atr": [1, 1, 1, 4, 4, 1, 1, 1], "setup": [True, True, False, False, False, False, False, False]},
        index=index,
    )
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data["close"] = np.arange(100.0, 108.0)
    features["atr"] = 1.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    features["setup"] = [True, False, True, False, False, False, False, False]
    assert not strategy.generate_signals(data, features, None, params).data["signal"].any()


if __name__ == "__main__":
    _self_check()
