from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QueueParams:
    ma_bars: int = 20
    regime_bars: int = 10
    pressure_bars: int = 4
    z_band: float = 0.75
    capacity: float = 2.0
    migration_threshold: float = 0.35
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[QueueParams]):
    strategy_id = "gen_local-w2_1790963752"

    @classmethod
    def params_type(cls):
        return QueueParams

    def warmup_bars(self, params: QueueParams) -> int:
        return int(params.ma_bars + max(params.regime_bars, params.pressure_bars))

    def indicators(self, data: pd.DataFrame, params: QueueParams) -> pd.DataFrame:
        if (params.ma_bars < 2 or params.regime_bars < 1 or
                params.pressure_bars < 1 or params.hold_bars < 1 or
                not np.isfinite(params.z_band) or params.z_band <= 0 or
                not np.isfinite(params.capacity) or params.capacity <= 0 or
                not np.isfinite(params.migration_threshold) or
                params.migration_threshold <= 0):
            raise ValueError("Lookbacks and thresholds must be positive; ma_bars must be at least two.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["center"] = close.rolling(params.ma_bars).mean()
        out["dispersion"] = close.rolling(params.ma_bars).std(ddof=0).replace(0, np.nan)
        out["z"] = (close - out["center"]) / out["dispersion"]
        out["migration"] = (out["center"] - out["center"].shift(params.regime_bars)) / out["dispersion"]
        out["positive_load"] = (out["z"] - params.z_band).clip(lower=0).rolling(params.pressure_bars).sum()
        out["negative_load"] = (-out["z"] - params.z_band).clip(lower=0).rolling(params.pressure_bars).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1)
        out["regime"] = np.select(
            [valid & (out["migration"] > params.migration_threshold),
             valid & (out["migration"].abs() <= params.migration_threshold)],
            [1, 2], default=0,
        )
        out["load"] = np.select(
            [out["regime"].eq(1), out["regime"].eq(2)],
            [out["positive_load"], out["negative_load"]], default=0.0,
        )
        out["entry"] = (out["load"] >= params.capacity) & (out["load"].shift(1).fillna(0.0) < params.capacity)
        out["conviction"] = (0.25 + 0.70 * (out["load"] / params.capacity - 1.0)).clip(0.25, 0.95)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: QueueParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.25, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        exit_bar = -1
        entry_size = 0.25
        for i in range(len(data)):
            if i == exit_bar:
                continue
            if i > exit_bar and entries[i]:
                exit_bar = i + int(params.hold_bars)
                entry_size = conviction[i]
            if i < exit_bar:
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = QueueParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"entry": [True] * 12,
                        "conviction": np.linspace(0.3, 0.9, 12)}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert np.allclose(result["size"].iloc[1:5], 0.3)
    assert result["size"].iloc[6] > result["size"].iloc[1]
    assert result["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert not actual["entry"].any()
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
