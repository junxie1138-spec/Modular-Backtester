from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CorridorParams:
    anchor_bars: int = 20
    atr_bars: int = 18
    persistence_bars: int = 18
    band_atr: float = 0.5
    dominance_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[CorridorParams]):
    strategy_id = "gen_local_1791343192"

    @classmethod
    def params_type(cls) -> type[CorridorParams]:
        return CorridorParams

    def warmup_bars(self, params: CorridorParams) -> int:
        return max(params.anchor_bars + 1, params.atr_bars + 1) + params.persistence_bars

    def indicators(self, data: pd.DataFrame, params: CorridorParams) -> pd.DataFrame:
        if min(params.anchor_bars, params.atr_bars, params.persistence_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not np.isfinite(params.band_atr) or params.band_atr < 0:
            raise ValueError("band_atr must be finite and nonnegative.")
        if not 0 < params.dominance_threshold <= 1:
            raise ValueError("dominance_threshold must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["prior_high"] = data["high"].rolling(params.anchor_bars, min_periods=params.anchor_bars).max().shift(1)
        out["prior_low"] = data["low"].rolling(params.anchor_bars, min_periods=params.anchor_bars).min().shift(1)
        out["midpoint"] = (out["prior_high"] + out["prior_low"]) / 2
        out["upper"] = out["midpoint"] + params.band_atr * out["atr"]
        out["lower"] = out["midpoint"] - params.band_atr * out["atr"]
        valid = out["upper"].notna() & out["lower"].notna() & data["close"].notna()
        out["vote"] = (
            (data["close"] > out["upper"]).astype(float)
            - (data["close"] < out["lower"]).astype(float)
        ).where(valid)
        out["dominance"] = out["vote"].rolling(
            params.persistence_bars, min_periods=params.persistence_bars
        ).mean()
        out["entry_condition"] = (
            out["dominance"].notna()
            & (out["dominance"] >= params.dominance_threshold)
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CorridorParams,
    ) -> SignalFrame:
        condition = indicators["entry_condition"].fillna(False).astype(bool)
        df = pd.DataFrame(index=data.index)
        # Consecutive true bars confirm entry; the first false bar reverses it.
        df["signal"] = (condition & condition.shift(1, fill_value=False)).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    indicators = pd.DataFrame(
        {"entry_condition": [False, True, True, True, False, True, True, False]},
        index=index,
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, CorridorParams()).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert (result["size"] > 0).all()
