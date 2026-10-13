from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactParams:
    rank_window: int = 8
    impact_quantile: float = 0.875
    volume_ceiling: float = 0.5
    hold_bars: int = 8

    def __post_init__(self):
        if not 2 <= self.rank_window <= 9:
            raise ValueError("rank_window must be between 2 and 9")
        if not 0.5 < self.impact_quantile <= 1.0:
            raise ValueError("impact_quantile must be in (0.5, 1]")
        if not 0.0 < self.volume_ceiling <= 1.0:
            raise ValueError("volume_ceiling must be in (0, 1]")
        if not 5 <= self.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")


class GeneratedStrategy(BaseStrategy[ImpactParams]):
    strategy_id = "gen_local_1791260270"

    @classmethod
    def params_type(cls):
        return ImpactParams

    def warmup_bars(self, params: ImpactParams) -> int:
        return params.rank_window + 1

    def indicators(self, data: pd.DataFrame, params: ImpactParams) -> pd.DataFrame:
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].where(data["volume"] > 0).replace([np.inf, -np.inf], np.nan)
        impact = (returns / volume).replace([np.inf, -np.inf], np.nan)
        impact_rank = impact.abs().rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        volume_rank = volume.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return pd.DataFrame({
            "return": returns.fillna(0.0),
            "price_impact": impact.fillna(0.0),
            "impact_rank": impact_rank.fillna(0.0),
            "volume_rank": volume_rank.fillna(1.0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ImpactParams,
    ) -> SignalFrame:
        eligible = (
            (indicators["impact_rank"] >= params.impact_quantile)
            & (indicators["volume_rank"] <= params.volume_ceiling)
        )
        entries = np.where(eligible, -np.sign(indicators["return"]), 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position and i == exit_bar:
                position = 0
                continue
            if position == 0 and entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = ImpactParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0}, index=pd.date_range("2020-01-01", periods=24))
    indicators = pd.DataFrame({"return": -0.02, "impact_rank": 1.0, "volume_rank": 0.1}, index=data.index)
    indicators.loc[data.index[2:10], "return"] = 0.02
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist()[:11] == [0] + [1] * 8 + [0, -1]
    assert strategy.warmup_bars(params) <= 10
    flat = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data
    assert (flat["signal"] == 0).all()
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
