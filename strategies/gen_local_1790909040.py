from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalElasticParams:
    lookback: int = 8
    seasonal_amplitude: float = 0.75
    threshold: float = 0.6


class GeneratedStrategy(BaseStrategy[SeasonalElasticParams]):
    strategy_id = "gen_local_1790909040"

    @classmethod
    def params_type(cls) -> type[SeasonalElasticParams]:
        return SeasonalElasticParams

    def warmup_bars(self, params: SeasonalElasticParams) -> int:
        return params.lookback

    def indicators(
        self, data: pd.DataFrame, params: SeasonalElasticParams
    ) -> pd.DataFrame:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not np.isfinite(params.seasonal_amplitude) or params.seasonal_amplitude < 0:
            raise ValueError("seasonal_amplitude must be finite and nonnegative")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(params.lookback, min_periods=params.lookback).mean()
        out["std"] = close.rolling(params.lookback, min_periods=params.lookback).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["std"].replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        # Tuesday is maximum seasonal support; Friday is near minimum.
        out["seasonal"] = params.seasonal_amplitude * np.cos(
            2.0 * np.pi * (data.index.dayofweek.to_numpy() - 1.0) / 5.0
        )
        out["tension"] = out["seasonal"] - out["z"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalElasticParams,
    ) -> SignalFrame:
        tension = indicators["tension"]
        # Mirrored thresholds: retain the position throughout the neutral band.
        events = pd.Series(
            np.where(
                tension >= params.threshold,
                1.0,
                np.where(tension <= -params.threshold, 0.0, np.nan),
            ),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = events.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = SeasonalElasticParams()
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    probe = pd.DataFrame({"tension": [np.nan, 0.7, 0.2, -0.7, 0.8, -0.8]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, probe, None, p).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert (result["size"] > 0).all()
    flat = GeneratedStrategy.indicators(None, data, SeasonalElasticParams(lookback=2))
    assert flat["tension"].isna().all()
