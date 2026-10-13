from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ParticipationParams:
    fractional_order: float = 0.65
    fractional_window: int = 16
    regime_window: int = 4
    participation_threshold: float = 0.70
    entry_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[ParticipationParams]):
    strategy_id = "gen_local_1791274495"

    @classmethod
    def params_type(cls) -> type[ParticipationParams]:
        return ParticipationParams

    @staticmethod
    def warmup_bars(params: ParticipationParams) -> int:
        return int(params.fractional_window + params.regime_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ParticipationParams) -> pd.DataFrame:
        if params.fractional_window < 2 or params.regime_window < 2:
            raise ValueError("Indicator windows must be at least two.")
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one.")
        if not 0.0 < params.participation_threshold <= 1.0:
            raise ValueError("participation_threshold must be in (0, 1].")
        if not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0.0:
            raise ValueError("entry_threshold must be finite and positive.")

        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: fixed-width fractional filter; increase the window for longer memory.
        for lag in range(1, params.fractional_window):
            weight *= -(params.fractional_order - lag + 1.0) / lag
            fractional_close = fractional_close + weight * close.shift(lag)

        innovation = fractional_close.diff()
        window = params.regime_window
        absolute_mass = innovation.abs().rolling(window, min_periods=window).sum()
        squared_mass = innovation.pow(2).rolling(window, min_periods=window).sum()
        participation = absolute_mass.pow(2).div(
            window * squared_mass.where(squared_mass > 0.0)
        ).clip(0.0, 1.0)
        innovation_scale = squared_mass.div(window).pow(0.5)
        standardized_innovation = innovation.div(
            innovation_scale.where(innovation_scale > 0.0)
        ).replace([np.inf, -np.inf], np.nan)

        regime = pd.Series(np.nan, index=data.index, dtype=float)
        valid = participation.notna() & standardized_innovation.notna()
        regime.loc[valid] = np.where(
            participation.loc[valid] >= params.participation_threshold, 1.0, -1.0
        )
        forecast = (regime * standardized_innovation).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "innovation": innovation,
            "absolute_mass": absolute_mass,
            "squared_mass": squared_mass,
            "participation": participation,
            "innovation_scale": innovation_scale,
            "standardized_innovation": standardized_innovation,
            "regime": regime,
            "forecast": forecast,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ParticipationParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"].replace([np.inf, -np.inf], np.nan)
        entries = np.where(
            forecast >= params.entry_threshold,
            1,
            np.where(forecast <= -params.entry_threshold, -1, 0),
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[bar] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(index=index)
    forecast = pd.DataFrame({"forecast": [1.0, 1.0, np.nan, -1.0, -1.0, -1.0, 1.0, 1.0]}, index=index)
    params = ParticipationParams()
    result = GeneratedStrategy.generate_signals(data, forecast, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    reflected = GeneratedStrategy.generate_signals(data, -forecast, None, params).data
    assert reflected["signal"].tolist() == (-result["signal"]).tolist()
    assert result["size"].gt(0.0).all()
