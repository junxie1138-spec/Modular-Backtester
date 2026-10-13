from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    shock_window: int = 4
    pressure_threshold: float = 1.25


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1791313131"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        return 41 + max(1, int(params.shock_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        window = max(1, int(params.shock_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["squared_return"] = out["return"].pow(2)
        out["signed_squared_return"] = out["return"] * out["return"].abs()
        out["baseline_variance"] = out["squared_return"].rolling(40, min_periods=40).mean().shift(window)
        out["shock_pressure"] = out["signed_squared_return"].rolling(window, min_periods=window).mean()
        denominator = out["baseline_variance"].where(out["baseline_variance"] > 0)
        out["normalized_pressure"] = (out["shock_pressure"] / denominator).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        pressure = indicators["normalized_pressure"]
        entry_condition = pressure.lt(-abs(float(params.pressure_threshold))).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Remain long only while the entry condition holds; its flip exits.
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"normalized_pressure": [np.nan, -2.0, -2.0, 0.0, -2.0, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, ShockwaveParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 1]
    assert result.data["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(ShockwaveParams()) == 45


if __name__ == "__main__":
    _self_check()
