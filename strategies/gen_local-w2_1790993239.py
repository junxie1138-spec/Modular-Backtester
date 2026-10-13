from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    roc_bars: int = 5
    percentile_bars: int = 63
    acceleration_quantile: float = 0.80
    volume_quantile: float = 0.40
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local-w2_1790993239"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    def warmup_bars(self, params: ParticipationParams) -> int:
        return int(params.roc_bars + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)
        lagged_close = close.shift(params.roc_bars).replace(0.0, np.nan)
        out["roc"] = (close / lagged_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_threshold"] = out["acceleration"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.acceleration_quantile).shift(1)
        out["volume_threshold"] = volume.rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.volume_quantile).shift(1)
        out["volume"] = volume
        bar_range = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["close_location"] = ((close - data["low"]) / bar_range).replace(
            [np.inf, -np.inf], np.nan
        )
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ParticipationParams,
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entries = (
            ready
            & (indicators["roc"] > 0.0)
            & (indicators["acceleration"] > 0.0)
            & (indicators["acceleration"] > indicators["acceleration_threshold"])
            & (indicators["volume"] > 0.0)
            & (indicators["volume"] < indicators["volume_threshold"])
            & (indicators["close_location"] > 0.5)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        holding = max(1, int(params.hold_bars))
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if entries[i]:
                signal[i] = 1
                exit_bar = i + holding
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
