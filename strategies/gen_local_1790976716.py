from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParticipationParams:
    scale_window: int = 40
    vote_window: int = 8
    threshold: float = 0.375
    noise_floor: float = 0.25
    spike_multiple: float = 3.0
    refractory_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapParticipationParams]):
    strategy_id = "gen_local_1790976716"

    @classmethod
    def params_type(cls):
        return GapParticipationParams

    def warmup_bars(self, params: GapParticipationParams) -> int:
        return int(params.scale_window + max(params.vote_window, params.refractory_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: GapParticipationParams) -> pd.DataFrame:
        if min(params.scale_window, params.vote_window, params.refractory_bars) < 1:
            raise ValueError("Window lengths must be positive.")
        if not 0 < params.threshold <= 1:
            raise ValueError("threshold must be in (0, 1].")
        if not 0 <= params.noise_floor < params.spike_multiple:
            raise ValueError("Require 0 <= noise_floor < spike_multiple.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        ratio = data["open"].where(data["open"] > 0).div(
            out["prior_close"].where(out["prior_close"] > 0)
        )
        out["gap"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        out["gap_scale"] = out["gap"].abs().rolling(
            params.scale_window, min_periods=params.scale_window
        ).mean().shift(1)
        out["ready"] = out["gap"].notna() & out["gap_scale"].gt(0)
        out["eligible"] = (
            out["ready"]
            & out["gap"].abs().ge(params.noise_floor * out["gap_scale"])
            & out["gap"].abs().le(params.spike_multiple * out["gap_scale"])
        )
        out["vote"] = np.sign(out["gap"]).where(out["eligible"], 0.0).where(out["ready"])
        out["participation"] = out["vote"].rolling(
            params.vote_window, min_periods=params.vote_window
        ).mean()
        out["spike"] = out["gap"].abs().gt(params.spike_multiple * out["gap_scale"])
        out["refractory"] = out["spike"].astype(float).rolling(
            params.refractory_bars, min_periods=params.refractory_bars
        ).max().fillna(1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapParticipationParams,
    ) -> SignalFrame:
        permitted = indicators["ready"] & indicators["refractory"].eq(0)
        score = indicators["participation"]
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[permitted & score.ge(params.threshold)] = 1
        direction.loc[permitted & score.le(-params.threshold)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(direction.to_numpy()):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapParticipationParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = pd.DataFrame(
        {"ready": True, "refractory": 0.0,
         "participation": [0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5]},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, features, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params)["participation"].isna().all()
