from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SemivolatilityParams:
    risk_window: int = 20
    smoothing: int = 3
    threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[SemivolatilityParams]):
    strategy_id = "gen_local_1791333244"

    @classmethod
    def params_type(cls) -> type[SemivolatilityParams]:
        return SemivolatilityParams

    def warmup_bars(self, params: SemivolatilityParams) -> int:
        return params.risk_window + params.smoothing

    def indicators(self, data: pd.DataFrame, params: SemivolatilityParams) -> pd.DataFrame:
        if params.risk_window < 2 or params.smoothing < 1:
            raise ValueError("Lookback lengths must be positive; risk_window must be at least two.")
        if not np.isfinite(params.threshold) or not 0 < params.threshold < 1:
            raise ValueError("threshold must be strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["returns"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up_energy"] = out["returns"].clip(lower=0).pow(2)
        out["down_energy"] = out["returns"].clip(upper=0).pow(2)
        out["up_vol"] = np.sqrt(out["up_energy"].rolling(params.risk_window, min_periods=params.risk_window).mean())
        out["down_vol"] = np.sqrt(out["down_energy"].rolling(params.risk_window, min_periods=params.risk_window).mean())
        out["total_vol"] = out["up_vol"] + out["down_vol"]
        out["balance"] = (out["up_vol"] - out["down_vol"]) / out["total_vol"].where(out["total_vol"] > 0)
        out.loc[out["total_vol"].eq(0), "balance"] = 0.0
        out["strength"] = out["balance"].rolling(params.smoothing, min_periods=params.smoothing).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SemivolatilityParams) -> SignalFrame:
        strength = indicators["strength"].reindex(data.index)
        decisions = pd.Series(
            np.where(strength >= params.threshold, 1.0,
                     np.where(strength <= -params.threshold, 0.0, np.nan)),
            index=data.index,
        )
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SemivolatilityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"strength": [np.nan, 0.2, 0.0, -0.05, -0.2, 0.0, 0.2, 0.0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 1]
    assert result.data["size"].gt(0).all()
