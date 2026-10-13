from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    ma_window: int = 20
    entry_z: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1791303124"

    @classmethod
    def params_type(cls) -> type[ElasticParams]:
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        if params.ma_window < 2:
            raise ValueError("ma_window must be at least 2")
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if params.hold_bars not in (3, 4, 5):
            raise ValueError("hold_bars must be 3, 4, or 5")
        return int(max(200, params.ma_window + 5))

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        ma = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        scale = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).replace(0.0, np.nan)
        displacement = close - ma
        z = displacement / scale
        negative = (z < -0.5).astype(float).where(z.notna())
        positive = (z > 0.5).astype(float).where(z.notna())
        out = pd.DataFrame(index=data.index)
        out["ma"] = ma
        out["scale"] = scale
        out["displacement"] = displacement
        out["z"] = z
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["anchor_ratio"] = ma.diff(3).abs() / displacement.abs().replace(0.0, np.nan)
        out["negative_residence"] = negative.rolling(6, min_periods=6).sum()
        out["positive_residence"] = positive.rolling(6, min_periods=6).sum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        z = indicators["z"]
        stable = indicators["anchor_ratio"].le(0.20)
        long_entry = (
            stable
            & data["close"].gt(indicators["ma200"])
            & z.le(-params.entry_z)
            & indicators["negative_residence"].le(3)
        ).fillna(False)
        short_entry = (
            stable
            & data["close"].lt(indicators["ma200"])
            & z.ge(params.entry_z)
            & indicators["positive_residence"].le(3)
        ).fillna(False)
        entries = np.where(long_entry.to_numpy(), 1, np.where(short_entry.to_numpy(), -1, 0))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"z": 2.0, "ma200": 110.0, "anchor_ratio": 0.0, "negative_residence": 1.0, "positive_residence": 1.0}, index=index)
    derived.loc[index[0], ["z", "ma200"]] = [-2.0, 90.0]
    result = GeneratedStrategy.generate_signals(None, data, derived, None, ElasticParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    missing = derived * np.nan
    result = GeneratedStrategy.generate_signals(None, data, missing, None, ElasticParams()).data
    assert result["signal"].eq(0).all()
