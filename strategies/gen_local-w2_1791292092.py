from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    noise_window: int = 5
    gap_noise: float = 0.35
    acceleration_noise: float = 0.20


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = "gen_local-w2_1791292092"

    @classmethod
    def params_type(cls) -> type[GapRecoveryParams]:
        return GapRecoveryParams

    @staticmethod
    def warmup_bars(params: GapRecoveryParams) -> int:
        if not 2 <= params.noise_window <= 8:
            raise ValueError("noise_window must be between 2 and 8")
        if not np.isfinite(params.gap_noise) or params.gap_noise < 0:
            raise ValueError("gap_noise must be finite and nonnegative")
        if not np.isfinite(params.acceleration_noise) or params.acceleration_noise < 0:
            raise ValueError("acceleration_noise must be finite and nonnegative")
        return int(params.noise_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        result["previous_close"] = data["close"].shift(1)
        result["previous_low"] = data["low"].shift(1)
        denominator = result["previous_close"].where(result["previous_close"] > 0)
        result["gap"] = data["open"].div(denominator).sub(1)
        result["roc"] = data["close"].div(denominator).sub(1)
        result["acceleration"] = result["roc"].diff()
        result["noise"] = result["roc"].abs().rolling(
            params.noise_window, min_periods=params.noise_window
        ).mean().shift(1)
        result["breached_low"] = data["low"] < result["previous_low"]
        result["reclaimed_low"] = data["close"] > result["previous_low"]
        result["unfilled_gap"] = data["close"] < result["previous_close"]
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRecoveryParams,
    ) -> SignalFrame:
        valid = indicators[[
            "previous_close", "previous_low", "gap", "roc", "acceleration", "noise"
        ]].notna().all(axis=1)
        entry = (
            valid
            & (indicators["noise"] > 0)
            & (indicators["gap"] < -params.gap_noise * indicators["noise"])
            & indicators["breached_low"]
            & indicators["reclaimed_low"]
            & indicators["unfilled_gap"]
            & (indicators["acceleration"] > params.acceleration_noise * indicators["noise"])
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long exactly while the entry condition holds; its reversal exits.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    close = np.array([100., 101., 100., 99., 98., 97., 96., 95.9, 96., 96.])
    opening = np.array([100., 100., 101., 100., 99., 98., 97., 94., 95.9, 96.])
    data = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) + 0.2,
        "low": np.minimum(opening, close) - 0.2,
        "close": close,
        "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=len(close)))
    strategy = GeneratedStrategy()
    params = GapRecoveryParams()
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].tolist() == [0] * 8 + [1, 0]
    assert signals["size"].gt(0).all()
    assert strategy.warmup_bars(GapRecoveryParams(noise_window=8)) == 10


if __name__ == "__main__":
    _self_check()
