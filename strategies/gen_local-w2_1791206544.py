from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 12
    acceleration_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791206544"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(2, int(params.lookback)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        roc = close.pct_change(fill_method=None)
        acceleration = roc.diff()
        gap = opening.div(close.shift(1)).sub(1.0)
        session = close.div(opening).sub(1.0)
        gap_energy = gap.pow(2).rolling(window, min_periods=window).mean()
        session_energy = session.pow(2).rolling(window, min_periods=window).mean()
        scale = acceleration.pow(2).rolling(window, min_periods=window).mean().pow(0.5)
        normalized = acceleration.div(scale.where(scale > 0))
        total_energy = gap_energy + session_energy
        balance = (session_energy - gap_energy).div(total_energy.where(total_energy > 0))
        raw = pd.DataFrame({
            "roc": roc,
            "acceleration": acceleration,
            "gap_return": gap,
            "session_return": session,
            "gap_energy": gap_energy,
            "session_energy": session_energy,
            "acceleration_scale": scale,
            "normalized_acceleration": normalized,
            "energy_balance": balance,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        raw["ready"] = raw.notna().all(axis=1)
        return raw.fillna(0.0)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        threshold = max(0.0, float(params.acceleration_threshold))
        regime_polarity = np.where(indicators["energy_balance"] >= 0.0, 1, -1)
        score = indicators["normalized_acceleration"].to_numpy() * regime_polarity
        ready = indicators["ready"].to_numpy(dtype=bool)
        entry = np.where(ready & (score > threshold), 1,
                         np.where(ready & (score < -threshold), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, desired in enumerate(entry):
            if position == 0:
                position = int(desired)
            elif desired == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({
        "energy_balance": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
        "normalized_acceleration": [1.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0],
        "ready": True,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    derived = GeneratedStrategy.indicators(data, Params(lookback=2))
    assert not derived.isna().any().any()
    assert GeneratedStrategy.warmup_bars(Params(lookback=2)) == 4


if __name__ == "__main__":
    _self_check()
