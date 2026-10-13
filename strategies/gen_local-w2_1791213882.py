from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DisplacementMemoryParams:
    ma_bars: int = 40
    memory_bars: int = 10
    percentile_bars: int = 126
    percentile: float = 0.80
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[DisplacementMemoryParams]):
    strategy_id = "gen_local-w2_1791213882"

    @classmethod
    def params_type(cls) -> type[DisplacementMemoryParams]:
        return DisplacementMemoryParams

    @staticmethod
    def warmup_bars(params: DisplacementMemoryParams) -> int:
        return int(params.ma_bars + params.memory_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: DisplacementMemoryParams) -> pd.DataFrame:
        if min(params.ma_bars, params.memory_bars, params.percentile_bars) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.5 < params.percentile < 1.0:
            raise ValueError("percentile must lie strictly between 0.5 and 1")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["dispersion"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["abs_z"] = out["z"].abs()
        out["z_change"] = out["z"].diff()
        out["signed_memory"] = out["z"].rolling(params.memory_bars, min_periods=params.memory_bars).sum()
        out["absolute_memory"] = out["abs_z"].rolling(params.memory_bars, min_periods=params.memory_bars).sum()
        out["coherence"] = out["signed_memory"].abs() / out["absolute_memory"].replace(0.0, np.nan)
        out["displacement_threshold"] = out["abs_z"].rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.percentile).shift(1)
        out["coherence_threshold"] = out["coherence"].rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.percentile).shift(1)
        out["qualified"] = ((out["abs_z"] > out["displacement_threshold"]) & (out["coherence"] >= out["coherence_threshold"])).fillna(False)
        out["long_entry"] = (out["qualified"] & (out["z"] > 0.0) & (out["signed_memory"] > 0.0) & (out["z_change"] > 0.0)).fillna(False)
        out["short_entry"] = (out["qualified"] & (out["z"] < 0.0) & (out["signed_memory"] < 0.0) & (out["z_change"] < 0.0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DisplacementMemoryParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        direction = 0
        entry_bar = -1

        for i in range(len(data)):
            if direction != 0:
                if i - entry_bar == params.hold_bars:
                    direction = 0
                    continue
            elif longs[i] or shorts[i]:
                direction = 1 if longs[i] else -1
                entry_bar = i
            signal[i] = direction

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = DisplacementMemoryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=39, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"long_entry": False, "short_entry": False}, index=index)
    entries.loc[index[0], "long_entry"] = True
    entries.loc[index[2], "short_entry"] = True
    entries.loc[index[19], "short_entry"] = True
    result = strategy.generate_signals(data, entries, None, params).data
    expected = [0] + [1] * 18 + [0] + [-1] * 18 + [0]
    assert result["signal"].tolist() == expected
    assert (result["size"] > 0.0).all()
    flat_index = pd.date_range("2020-01-01", periods=strategy.warmup_bars(params) + 5, freq="B")
    flat = data.iloc[:1].reindex(flat_index).fillna(100.0)
    flat_signals = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert flat_signals["signal"].eq(0).all()
