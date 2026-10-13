from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    lookback: int = 18
    volume_lookback: int = 20
    participation_edge: float = 0.10
    efficiency_floor: float = 0.25


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1790976598"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    @staticmethod
    def warmup_bars(params: ParticipationParams) -> int:
        return int(params.volume_lookback + params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        if params.lookback < 2 or params.volume_lookback < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not np.isfinite(params.participation_edge) or params.participation_edge < 0:
            raise ValueError("participation_edge must be finite and nonnegative")
        if not np.isfinite(params.efficiency_floor) or not 0 <= params.efficiency_floor <= 1:
            raise ValueError("efficiency_floor must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["absolute_return"] = out["return"].abs()
        out["volume_reference"] = data["volume"].rolling(params.volume_lookback, min_periods=params.volume_lookback).mean().shift(1)
        out["relative_volume"] = data["volume"].where(data["volume"] >= 0) / out["volume_reference"].where(out["volume_reference"] > 0)
        out["weighted_return"] = out["return"] * out["relative_volume"]
        n = params.lookback
        out["return_sum"] = out["return"].rolling(n, min_periods=n).sum()
        out["path_length"] = out["absolute_return"].rolling(n, min_periods=n).sum()
        out["volume_sum"] = out["relative_volume"].rolling(n, min_periods=n).sum()
        out["weighted_return_sum"] = out["weighted_return"].rolling(n, min_periods=n).sum()
        out["ordinary_drift"] = out["return_sum"] / n
        out["weighted_drift"] = out["weighted_return_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["mean_absolute_return"] = out["path_length"] / n
        out["participation_contrast"] = (out["weighted_drift"] - out["ordinary_drift"]) / out["mean_absolute_return"].where(out["mean_absolute_return"] > 0)
        out["efficiency"] = out["return_sum"] / out["path_length"].where(out["path_length"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ParticipationParams) -> SignalFrame:
        participation = indicators["participation_contrast"] > params.participation_edge
        efficient_ascent = indicators["efficiency"] > params.efficiency_floor
        ready = np.isfinite(indicators[["participation_contrast", "efficiency"]]).all(axis=1)
        # The joint condition defines exposure; its reversal is the only exit.
        df = pd.DataFrame(index=data.index)
        df["signal"] = (ready & participation & efficient_ascent).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({"participation_contrast": [0.2, 0.0, 0.2, 0.2, np.nan], "efficiency": [0.5, 0.5, 0.0, 0.5, 0.5]}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, ParticipationParams()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
