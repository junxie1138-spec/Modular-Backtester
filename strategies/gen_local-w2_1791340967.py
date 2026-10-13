from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownWaveParams:
    peak_window: int = 20
    phase_lag: int = 4
    scale_window: int = 20
    threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[DrawdownWaveParams]):
    strategy_id = "gen_local-w2_1791340967"

    @classmethod
    def params_type(cls) -> type[DrawdownWaveParams]:
        return DrawdownWaveParams

    @staticmethod
    def warmup_bars(params: DrawdownWaveParams) -> int:
        return int(max(max(2, params.peak_window) + max(1, params.phase_lag), max(2, params.scale_window)))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownWaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["peak"] = close.rolling(max(2, params.peak_window), min_periods=max(2, params.peak_window)).max()
        out["drawdown"] = (out["peak"] - close) / out["peak"]
        out["lagged_drawdown"] = out["drawdown"].shift(max(1, params.phase_lag))
        out["range_fraction"] = (data["high"] - data["low"]).clip(lower=0) / close
        out["scale"] = out["range_fraction"].rolling(max(2, params.scale_window), min_periods=max(2, params.scale_window)).mean()
        out["wave"] = (out["drawdown"] - out["lagged_drawdown"]) / out["scale"].where(out["scale"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownWaveParams) -> SignalFrame:
        wave = indicators["wave"].replace([np.inf, -np.inf], np.nan)
        threshold = max(0.0, float(params.threshold))
        reversals = pd.Series(
            np.select([wave > threshold, wave < -threshold], [1.0, 0.0], default=np.nan),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = reversals.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    sample = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"wave": [np.nan, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(sample, derived, None, DrawdownWaveParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 1, 0]
    assert (result["size"] > 0).all()
