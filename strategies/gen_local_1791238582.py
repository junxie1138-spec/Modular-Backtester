from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationBudgetParams:
    vol_window: int = 5
    acceleration_threshold: float = 0.05
    target_annual_vol: float = 0.12

    def __post_init__(self):
        if self.vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        if not np.isfinite(self.acceleration_threshold) or self.acceleration_threshold <= 0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(self.target_annual_vol) or self.target_annual_vol <= 0:
            raise ValueError("target_annual_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[AccelerationBudgetParams]):
    strategy_id = "gen_local_1791238582"

    @classmethod
    def params_type(cls):
        return AccelerationBudgetParams

    def warmup_bars(self, params):
        return int(max(200, params.vol_window + 2))

    def indicators(self, data, params):
        close = data["close"].astype(float).where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["roc"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["raw_acceleration"] = out["roc"].diff()
        out["daily_vol"] = out["roc"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=0)
        out["risk_roc"] = out["roc"] / out["daily_vol"].where(out["daily_vol"] > 1e-8)
        out["risk_acceleration"] = out["risk_roc"].diff()
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        score = indicators["risk_acceleration"]
        # Opposite directional acceleration exits; neutral readings retain exposure.
        # Divergence and the MA regime gate entries, not existing-position exits.
        direction = pd.Series(
            np.select(
                [score > params.acceleration_threshold,
                 score < -params.acceleration_threshold],
                [1, -1], default=0
            ), index=data.index, dtype=int
        )
        divergence = indicators["raw_acceleration"] * score < 0
        regime_ok = (
            ((direction == 1) & (data["close"] > indicators["ma200"]))
            | ((direction == -1) & (data["close"] < indicators["ma200"]))
        )
        entry = direction.where(divergence & regime_ok, 0)
        state = 0
        positions = np.zeros(len(data), dtype=int)
        for i, (vote, candidate) in enumerate(zip(direction.to_numpy(), entry.to_numpy())):
            if state != 0 and vote == -state:
                state = 0
            elif state == 0 and candidate != 0:
                state = int(candidate)
            positions[i] = state

        df = pd.DataFrame({"signal": positions}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = (
            params.target_annual_vol
            / indicators["annual_vol"].where(indicators["annual_vol"] > 1e-8)
        ).replace([np.inf, -np.inf], np.nan).clip(lower=0.01, upper=0.95)
        df["size"] = exposure.ffill().shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = AccelerationBudgetParams()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [101, 101, 99, 99, 99, 101, 101]}, index=index)
    derived = pd.DataFrame({
        "ma200": [100.0] * 7,
        "risk_acceleration": [0.2, 0.0, -0.2, -0.2, 0.0, 0.2, 0.0],
        "raw_acceleration": [-0.01, 0.0, 0.01, 0.01, 0.0, -0.01, 0.0],
        "annual_vol": [0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2],
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.isclose(result["size"].iloc[2], 0.3)
    assert np.isfinite(result["size"]).all() and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
