from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    energy_window: int = 8
    participation_window: int = 8
    dominance_ratio: float = 1.5


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local-w2_1791318071"

    @classmethod
    def params_type(cls) -> type[MomentumParams]:
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        return max(2, params.energy_window, params.participation_window) + 1

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        energy_window = max(2, int(params.energy_window))
        participation_window = max(2, int(params.participation_window))
        close = data["close"].astype(float).where(data["close"] > 0)
        volume = data["volume"].astype(float).where(data["volume"] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        volume_change = np.log(volume).diff().replace([np.inf, -np.inf], np.nan)
        upside_squared = returns.clip(lower=0).pow(2)
        downside_squared = returns.clip(upper=0).pow(2)
        upside_energy = upside_squared.rolling(energy_window, min_periods=energy_window).mean()
        downside_energy = downside_squared.rolling(energy_window, min_periods=energy_window).mean()
        paired_returns = returns.where(volume_change.notna())
        paired_volume_change = volume_change.where(returns.notna())
        mean_return = paired_returns.rolling(participation_window, min_periods=participation_window).mean()
        mean_volume_change = paired_volume_change.rolling(participation_window, min_periods=participation_window).mean()
        mean_product = (paired_returns * paired_volume_change).rolling(participation_window, min_periods=participation_window).mean()
        covariance = mean_product - mean_return * mean_volume_change
        return pd.DataFrame({
            "returns": returns,
            "volume_change": volume_change,
            "upside_squared": upside_squared,
            "downside_squared": downside_squared,
            "upside_energy": upside_energy,
            "downside_energy": downside_energy,
            "mean_return": mean_return,
            "mean_volume_change": mean_volume_change,
            "mean_product": mean_product,
            "participation_covariance": covariance,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        upside = indicators["upside_energy"]
        downside = indicators["downside_energy"]
        covariance = indicators["participation_covariance"]
        valid = np.isfinite(upside) & np.isfinite(downside) & np.isfinite(covariance)
        volatility_agrees = (upside > params.dominance_ratio * downside) & (upside > 0)
        participation_agrees = covariance > 0
        # The position exits whenever the same entry conjunction becomes false.
        agreement = valid & volatility_agrees & participation_agrees
        df = pd.DataFrame(index=data.index)
        df["signal"] = agreement.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = MomentumParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({
        "upside_energy": [np.nan, 2.0, 1.0, 2.0, 2.0, 2.0],
        "downside_energy": [np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        "participation_covariance": [np.nan, 1.0, 1.0, -1.0, 1.0, 0.0],
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 1]
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) == 9
