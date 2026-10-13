from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 3
    energy_threshold: float = 0.55


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791265276"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return 8 + params.energy_window

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff().replace([np.inf, -np.inf], np.nan)
        for scale in (1, 2, 4):
            block = f"block_{scale}"
            detail = f"detail_{scale}"
            energy = f"energy_{scale}"
            out[block] = out["return"].rolling(scale, min_periods=scale).sum()
            out[detail] = (out[block] - out[block].shift(scale)) / np.sqrt(2.0 * scale)
            out[energy] = out[detail].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["approximation"] = out["return"].rolling(8, min_periods=8).sum() / np.sqrt(8.0)
        out["coarse_energy"] = out["approximation"].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["total_energy"] = out[["coarse_energy", "energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=4)
        out["coarse_share"] = out["coarse_energy"] / out["total_energy"].where(out["total_energy"] > 0)
        out["direction"] = out["approximation"].rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["score"] = (np.sign(out["direction"]) * out["coarse_share"]).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        score = indicators["score"].reindex(data.index).fillna(0.0)
        candidates = np.where(score >= params.energy_threshold, 1, np.where(score <= -params.energy_threshold, -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [np.nan, 0.8, 0.2, -0.8, -0.8, 0.0, 0.8, 0.8]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, WaveletParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    flat = GeneratedStrategy.indicators(data, WaveletParams())
    assert flat["score"].fillna(0).eq(0).all()
