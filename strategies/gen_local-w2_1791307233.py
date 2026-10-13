from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.45
    frac_window: int = 40
    position_window: int = 63
    arm_level: float = 0.20
    reclaim_level: float = 0.55
    hold_bars: int = 18

    def __post_init__(self):
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError("frac_order must be between zero and one")
        if self.frac_window < 2 or self.position_window < 2 or self.hold_bars < 1:
            raise ValueError("Invalid window or holding period")
        if not 0.0 <= self.arm_level < self.reclaim_level <= 1.0:
            raise ValueError("Invalid hysteresis thresholds")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791307233"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.frac_window + params.position_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        # Fixed-width fractional differencing: w[k] = w[k-1]*(k-1-d)/k.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional
        prior = fractional.shift(1)
        out["prior_low"] = prior.rolling(
            params.position_window, min_periods=params.position_window
        ).min()
        out["prior_high"] = prior.rolling(
            params.position_window, min_periods=params.position_window
        ).max()
        out["span"] = out["prior_high"] - out["prior_low"]
        out["position"] = (
            (fractional - out["prior_low"]) / out["span"].where(out["span"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        armed = False
        confirmations = 0
        exit_at = -1
        for i, position in enumerate(indicators["position"].to_numpy()):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
                    armed = False
                    confirmations = 0
                continue
            if not np.isfinite(position):
                armed = False
                confirmations = 0
                continue
            if position <= params.arm_level:
                armed = True
            if armed and position >= params.reclaim_level:
                confirmations += 1
            else:
                confirmations = 0
            if confirmations == 2:
                signal[i] = 1
                exit_at = i + params.hold_bars
                armed = False
                confirmations = 0
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    index = pd.date_range("2020-01-01", periods=60, freq="B")
    data = pd.DataFrame(1.0, index=index, columns=["open", "high", "low", "close", "volume"])
    position = np.full(60, 0.60)
    position[0] = 0.10
    position[2] = 0.40
    position[5:23] = 0.10
    position[30] = 0.10
    features = pd.DataFrame({"position": position}, index=index)
    result = GeneratedStrategy().generate_signals(data, features, None, params).data
    expected = np.zeros(60, dtype=int)
    expected[5:23] = 1
    expected[33:51] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
