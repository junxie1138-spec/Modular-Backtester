from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    ma_window: int = 3
    z_threshold: float = 0.6
    dominance_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local_1791248916"

    @classmethod
    def params_type(cls) -> type[MomentumParams]:
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        return max(2, int(params.ma_window))

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        window = max(2, int(params.ma_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(window, min_periods=window).mean()
        out["price_std"] = close.rolling(window, min_periods=window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["price_std"].where(out["price_std"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["body"] = close - data["open"]
        out["gap_magnitude"] = (data["open"] - close.shift(1)).abs()
        out["range"] = data["high"] - data["low"]
        out["dominance"] = (np.sign(out["body"]) * (out["body"].abs() - out["gap_magnitude"]).clip(lower=0) / out["range"].where(out["range"] > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        valid = np.isfinite(indicators["z"]) & np.isfinite(indicators["dominance"])
        long_entry = valid & (indicators["z"] > params.z_threshold) & (indicators["dominance"] > params.dominance_threshold)
        short_entry = valid & (indicators["z"] < -params.z_threshold) & (indicators["dominance"] < -params.dominance_threshold)
        evidence = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(evidence):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"z": [1.0, np.nan, -1.0, -1.0, 0.0, 1.0], "dominance": [0.5, np.nan, -0.5, -0.5, 0.0, 0.5]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, MomentumParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (result.data["size"] > 0).all()
