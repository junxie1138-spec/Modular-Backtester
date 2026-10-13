from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactWaveParams:
    volume_window: int = 20
    block_window: int = 10
    rank_window: int = 126
    persistence_window: int = 18
    entry_percentile: float = 0.58
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ImpactWaveParams]):
    strategy_id = "gen_local-w2_1791248184"

    @classmethod
    def params_type(cls) -> type[ImpactWaveParams]:
        return ImpactWaveParams

    @staticmethod
    def warmup_bars(params: ImpactWaveParams) -> int:
        return int(max(2, params.volume_window) + 2 * params.block_window + params.rank_window + params.persistence_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ImpactWaveParams) -> pd.DataFrame:
        if min(params.volume_window, params.block_window, params.rank_window, params.persistence_window) < 2:
            raise ValueError("All lookbacks must be at least two bars.")
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between 0.5 and 1.")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")

        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].where(data["volume"] > 0)
        volume_mean = volume.rolling(params.volume_window, min_periods=params.volume_window).mean()
        relative_volume = volume.div(volume_mean.where(volume_mean > 0))
        impact = returns.div(relative_volume.clip(lower=0.1))
        recent_impact = impact.rolling(params.block_window, min_periods=params.block_window).mean()
        preceding_impact = recent_impact.shift(params.block_window)
        impact_change = recent_impact - preceding_impact
        change_percentile = impact_change.rolling(params.rank_window, min_periods=params.rank_window).rank(method="average", pct=True)
        conviction = change_percentile.rolling(params.persistence_window, min_periods=params.persistence_window).mean()
        return pd.DataFrame({
            "returns": returns,
            "volume_mean": volume_mean,
            "relative_volume": relative_volume,
            "impact": impact,
            "recent_impact": recent_impact,
            "preceding_impact": preceding_impact,
            "impact_change": impact_change,
            "change_percentile": change_percentile,
            "conviction": conviction,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ImpactWaveParams) -> SignalFrame:
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan)
        # The same condition governs entry and reversal exit; there is no timed exit.
        entry_condition = conviction.ge(params.entry_percentile) & conviction.notna()
        strength = ((conviction - params.entry_percentile) / (1.0 - params.entry_percentile)).clip(0.0, 1.0).fillna(0.0)
        decision_size = params.min_size + (params.max_size - params.min_size) * strength
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        df["size"] = decision_size.shift(1).fillna(params.min_size).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ImpactWaveParams()
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"conviction": [np.nan, 0.7, 0.9, 0.4, 0.4]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert params.min_size < result["size"].iloc[2] < result["size"].iloc[3] <= params.max_size
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.indicators(data, params)["conviction"].isna().all()


if __name__ == "__main__":
    _self_check()
