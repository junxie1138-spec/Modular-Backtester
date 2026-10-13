from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    breakout_z: float = 1.5
    compression_z: float = 0.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791290361"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.shift(1).rolling(6, min_periods=6).mean()
        out["std"] = close.shift(1).rolling(6, min_periods=6).std(ddof=0).replace(0.0, np.nan)
        out["z"] = (close - out["ma"]) / out["std"]
        out["compression"] = out["z"].abs().shift(1).rolling(2, min_periods=2).max()
        out["channel_high"] = data["high"].shift(1).rolling(3, min_periods=3).max()
        out["channel_low"] = data["low"].shift(1).rolling(3, min_periods=3).min()
        out["recent_volume"] = volume.shift(1).rolling(2, min_periods=2).mean()
        out["earlier_volume"] = volume.shift(3).rolling(2, min_periods=2).mean()
        out["previous_volume"] = volume.shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if not (0.0 <= params.compression_z < params.breakout_z < float("inf")):
            raise ValueError("Require finite breakout_z > compression_z >= 0")
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be an integer from 3 through 5")
        ready = indicators.notna().all(axis=1)
        release = (
            ready
            & indicators["compression"].le(params.compression_z)
            & indicators["recent_volume"].lt(indicators["earlier_volume"])
            & data["volume"].gt(indicators["previous_volume"])
            & data["volume"].gt(indicators["recent_volume"])
        )
        long_entry = release & indicators["z"].ge(params.breakout_z) & data["close"].gt(indicators["channel_high"])
        short_entry = release & indicators["z"].le(-params.breakout_z) & data["close"].lt(indicators["channel_low"])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position, entered_at = 0, -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i - entered_at >= params.hold_bars:
                    position = 0
            elif entry != 0:
                position, entered_at = int(entry), i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 200.0}, index=index)
    indicators = GeneratedStrategy.indicators(data, params)
    indicators.loc[:, :] = 1.0
    indicators["compression"] = 0.0
    indicators["recent_volume"] = 50.0
    indicators["earlier_volume"] = 100.0
    indicators["previous_volume"] = 100.0
    indicators["channel_high"] = 99.0
    indicators["channel_low"] = 101.0
    indicators["z"] = np.array([2, -2, -2, -2, 2, -2, 2, 2, 2, 2, 2], dtype=float)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0.0).all()
    data["close"] = np.arange(len(data), dtype=float) + 100.0
    calculated = GeneratedStrategy.indicators(data, params)
    assert calculated["compression"].first_valid_index() == index[8]
    assert GeneratedStrategy.warmup_bars(params) == 9


if __name__ == "__main__":
    _self_check()
