from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 40
    strength_window: int = 18
    z_floor: float = 0.35


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791344560"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.ma_window + params.strength_window - 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.ma_window < 2 or params.strength_window < 1 or not np.isfinite(params.z_floor):
            raise ValueError("Invalid strategy parameters")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["regime_ma"] = close.rolling(200, min_periods=200).mean()
        out["trend_ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["trend_ma"]) / out["price_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["z_floor"] = out["z"].rolling(params.strength_window, min_periods=params.strength_window).quantile(0.25)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        condition = (
            indicators["z_floor"].notna()
            & indicators["regime_ma"].notna()
            & (indicators["z_floor"] > params.z_floor)
            & (data["close"] > indicators["regime_ma"])
        )
        df = pd.DataFrame(index=data.index)
        # Hold while the complete entry condition remains true; exit on its reversal.
        df["signal"] = condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    idx = pd.date_range("2020-01-01", periods=6)
    bars = pd.DataFrame({"close": [101.0, 101.0, 101.0, 101.0, 99.0, 101.0]}, index=idx)
    derived = pd.DataFrame({"regime_ma": 100.0, "z_floor": [np.nan, 0.5, 0.6, 0.2, 0.8, 0.9]}, index=idx)
    result = GeneratedStrategy.generate_signals(None, bars, derived, None, Params()).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.index.equals(bars.index) and (result["size"] > 0).all()
