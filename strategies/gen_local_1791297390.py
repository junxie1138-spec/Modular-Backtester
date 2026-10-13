from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 7
    area_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local_1791297390"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return params.path_bars + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.path_bars
        returns = np.log(data["close"].where(data["close"] > 0)).diff()
        variation = returns.abs()
        travel = variation.rolling(n, min_periods=n).sum()
        net = returns.rolling(n, min_periods=n).sum()
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            wedges = returns.shift(lag) * variation - variation.shift(lag) * returns
            area += 0.5 * wedges.rolling(n - lag, min_periods=n - lag).sum()
        denominator = travel.pow(2).where(travel > 0)
        return pd.DataFrame({
            "efficiency": (net / travel.where(travel > 0)).replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "area": (area / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        efficiency = indicators["efficiency"]
        area = indicators["area"]
        eligible = (efficiency.abs() > 0.3) & (area.abs() > params.area_threshold) & (area * efficiency < 0)
        candidates = np.where(eligible, np.sign(efficiency), 0).astype(int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i, candidate in enumerate(candidates):
            if position and i == exit_at:
                position = 0
            elif position == 0 and candidate:
                position = int(candidate)
                exit_at = i + 4
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    indicators = pd.DataFrame({"efficiency": 0.6, "area": -0.1}, index=index)
    result = GeneratedStrategy().generate_signals(
        pd.DataFrame(index=index), indicators, None, SignatureParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1]


if __name__ == "__main__":
    _self_check()
