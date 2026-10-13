from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureTrendParams:
    lookback: int = 9
    area_threshold: float = 0.04
    efficiency_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[SignatureTrendParams]):
    strategy_id = "gen_local_1790997526"

    @classmethod
    def params_type(cls) -> type[SignatureTrendParams]:
        return SignatureTrendParams

    def warmup_bars(self, params: SignatureTrendParams) -> int:
        return max(2, min(9, int(params.lookback))) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureTrendParams) -> pd.DataFrame:
        n = max(2, min(9, int(params.lookback)))
        log_close = np.log(data["close"].astype(float).where(data["close"] > 0.0))
        returns = log_close.diff().replace([np.inf, -np.inf], np.nan)
        net_move = returns.rolling(n, min_periods=n).sum()
        travel = returns.abs().rolling(n, min_periods=n).sum()

        # Second-order antisymmetric signature for increments (1, log_return).
        # Positive area means later returns exceed earlier returns.
        area = pd.Series(0.0, index=data.index)
        for lag in range(n):
            area = area + 0.5 * (n - 1 - 2 * lag) * returns.shift(lag)

        denominator = travel.where(travel > 0.0)
        result = pd.DataFrame(index=data.index)
        result["log_close"] = log_close
        result["log_return"] = returns
        result["net_move"] = net_move
        result["travel"] = travel
        result["signature_area"] = area
        result["area_strength"] = area / (n * denominator)
        result["directional_efficiency"] = net_move / denominator
        result["ready"] = area.notna() & net_move.notna() & denominator.notna()
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureTrendParams,
    ) -> SignalFrame:
        bullish = (
            indicators["ready"].astype(bool)
            & (indicators["net_move"] > 0.0)
            & (indicators["area_strength"] > params.area_threshold)
            & (indicators["directional_efficiency"] > params.efficiency_threshold)
        )
        df = pd.DataFrame(index=data.index)
        # Hold while the entry predicate remains true; its reversal is the only exit.
        df["signal"] = bullish.astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
