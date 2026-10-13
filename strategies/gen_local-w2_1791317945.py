from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CurvatureParams:
    roc_bars: int = 5
    scale_bars: int = 40
    low_bars: int = 15
    acceleration_threshold: float = 0.8
    hold_bars: int = 18
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[CurvatureParams]):
    strategy_id = "gen_local-w2_1791317945"

    @classmethod
    def params_type(cls) -> type[CurvatureParams]:
        return CurvatureParams

    def warmup_bars(self, params: CurvatureParams) -> int:
        return int(max(params.roc_bars + params.scale_bars + 2, params.low_bars + 1))

    def indicators(self, data: pd.DataFrame, params: CurvatureParams) -> pd.DataFrame:
        if min(params.roc_bars, params.low_bars, params.hold_bars) < 1 or params.scale_bars < 2:
            raise ValueError("Lookbacks and holding period must be positive; scale_bars must be at least two.")
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive.")
        if not np.isfinite(params.min_size) or not 0 < params.min_size <= 1:
            raise ValueError("min_size must be in (0, 1].")
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_scale"] = out["acceleration"].rolling(params.scale_bars, min_periods=params.scale_bars).std(ddof=0).shift(1)
        out["prior_low"] = close.rolling(params.low_bars, min_periods=params.low_bars).min().shift(1)
        out["strength"] = (out["acceleration"] / out["acceleration_scale"].where(out["acceleration_scale"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["entry"] = ((close < out["prior_low"]) & (out["roc"] < 0) & (out["strength"] >= params.acceleration_threshold)).fillna(False)
        out["conviction"] = (out["strength"] / (2.0 * params.acceleration_threshold)).clip(lower=params.min_size, upper=1.0).fillna(params.min_size)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CurvatureParams) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(params.min_size).to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
            elif entries[i]:
                exit_bar = i + params.hold_bars
                entry_size = conviction[i]
            if exit_bar >= 0:
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = CurvatureParams()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    conviction = np.linspace(0.3, 1.0, len(data))
    features = pd.DataFrame({"entry": True, "conviction": conviction}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] + [1] * 18 + [0, 1]
    assert (result["size"].iloc[1:19] == conviction[0]).all()
    assert (result["size"].iloc[20:38] == conviction[19]).all()
    assert (result["size"] > 0).all()
